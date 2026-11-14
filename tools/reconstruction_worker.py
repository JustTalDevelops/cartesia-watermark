"""Keep the isolated reconstruction runtime alive between serialized uploads."""
import json
from pathlib import Path
import selectors
import subprocess
import threading
import time


class ReconstructionWorker:
    def __init__(self, python, script, boundary_safe=False):
        self.command = [str(python), str(script), "--worker"]
        if boundary_safe:
            self.command.append('--boundary-safe')
        self.process = None
        self.lock = threading.RLock()

    def _receive(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(self.command, 0)
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(remaining):
                raise subprocess.TimeoutExpired(self.command, remaining)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Reconstruction worker exited before returning a result")
        return json.loads(line)

    def run(self, source, output, preset, metadata, timeout):
        """Run one job; a timeout or broken worker is discarded before the next job."""
        with self.lock:
            deadline = time.monotonic() + timeout
            try:
                if self.process is None or self.process.poll() is not None:
                    self.close()
                    self.process = subprocess.Popen(
                        self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        text=True, bufsize=1,
                    )
                    if self._receive(deadline) != {"ready": True}:
                        raise RuntimeError("Reconstruction worker could not load the model")
                request = {"input": str(Path(source).resolve()), "output": str(Path(output).resolve()),
                           "preset": preset, "metadata": str(Path(metadata).resolve())}
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                reply = self._receive(deadline)
                if "error" in reply:
                    raise RuntimeError("Reconstruction failed: " + reply["error"])
                return reply["report"]
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()
