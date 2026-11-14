from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from codec_content import ContentRecognizer
from codec_packed_fp16 import PackedLargeRecognizer,pack_linear_weights
from experiments.codec_profile_cache import configuration


@unittest.skipUnless(sys.version_info[:2]==(3,12),'Pinned codec environment uses Python 3.12')
class PackedRecognizerTests(unittest.TestCase):
    def test_packing_preserves_original_and_float_output_projection(self):
        import torch
        torch.set_num_threads(4)
        model=torch.nn.Module()
        model.projection=torch.nn.Linear(4,4)
        model.proj_out=torch.nn.Linear(4,3)
        packed,count=pack_linear_weights(model)
        self.assertEqual(count,1)
        self.assertIs(type(model.projection),torch.nn.Linear)
        self.assertIs(type(packed.proj_out),torch.nn.Linear)
        self.assertTrue(torch.equal(model.proj_out.weight,packed.proj_out.weight))
        self.assertEqual(packed.projection(torch.ones(1,4)).dtype,torch.float32)
        self.assertEqual(packed.projection._packed_params.dtype,torch.float16)

    def test_empty_or_unsupported_backend_cannot_silently_claim_packing(self):
        import torch
        with self.assertRaises(ValueError):pack_linear_weights(torch.nn.Identity())
        original=torch.backends.quantized.engine
        try:
            torch.backends.quantized.engine='qnnpack'
            with self.assertRaises(RuntimeError):pack_linear_weights(torch.nn.Linear(4,4))
        finally:
            torch.backends.quantized.engine=original

    def test_lazy_loading_cannot_change_cache_identity_or_repack_same_model(self):
        base={'guard_version':'prospective-v3','precision':'float32','large':{'model_id':'pinned fixture'}}
        def initialize(instance):
            instance.metadata=deepcopy(base)
            instance.large_model=None
        def load(instance):
            if instance.large_model is None:instance.large_model=SimpleNamespace(kind='float')
            return 'processor',instance.large_model
        packed=SimpleNamespace(kind='packed')
        with patch.object(ContentRecognizer,'__init__',initialize),patch.object(ContentRecognizer,'_large',load), \
                patch('codec_packed_fp16.pack_linear_weights',return_value=(packed,512)) as pack:
            recognizer=PackedLargeRecognizer()
            before=configuration(recognizer.metadata)
            self.assertNotEqual(before,configuration(base))
            self.assertEqual(recognizer._large(),('processor',packed))
            self.assertEqual(recognizer._large(),('processor',packed))
            self.assertEqual(configuration(recognizer.metadata),before)
            pack.assert_called_once()
            recognizer.large_model=None
            recognizer._large()
            self.assertEqual(pack.call_count,2)


if __name__=='__main__':unittest.main()
