# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The repository owner wants to test speech reconstruction and the earlier audio editor on their own recordings.

## Product Purpose

Upload a recording, reconstruct its speech, compare original and processed audio, and download the resulting WAV. Offer the earlier V3 editor as Small edits.

## Operating Context

This is a basic private testing website hosted on the existing machine and accessed through Tailscale. The repository already contains Python inference and the trained checkpoint.

## Capabilities and Constraints

Voice quality takes priority over detector suppression. The listener-approved reconstruction preset is the default; stronger reconstruction is optional. Processing preserves sample rate, channel count and sample count. Removal can fail on some recordings. Inference uses preinstalled local models and needs no Cartesia API or external model download. Preserve the frozen V3 editor and its inference implementation for Small edits.

## Product Principles

- Keep the upload, playback and download flow simple.
- Favor voice fidelity when offering processing controls.
- Describe processing results honestly without claiming that detection has been checked.
- Keep recordings on this machine.
