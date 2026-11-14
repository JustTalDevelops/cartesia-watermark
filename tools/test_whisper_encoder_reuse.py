from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from experiments.whisper_encoder_reuse import prepare_original


class WhisperEncoderReuseTests(unittest.TestCase):
    def model(self):
        model = Mock()
        model.generation_config = SimpleNamespace(language='ar', lang_to_id={'<|ar|>': 7})
        model.detect_language.return_value = [7]
        return model

    def test_short_source_encodes_once_and_preserves_mask(self):
        model = self.model()
        features, mask = np.zeros((1, 128, 3000)), object()
        language, parameters, reused = prepare_original(model, {'input_features': features, 'attention_mask': mask})
        self.assertEqual(language, 'ar')
        self.assertTrue(reused)
        model.get_encoder.return_value.assert_called_once_with(features, return_dict=True)
        self.assertIs(parameters['encoder_outputs'], model.get_encoder.return_value.return_value)
        self.assertIs(parameters['attention_mask'], mask)
        self.assertNotIn('input_features', parameters)
        self.assertEqual(model.generation_config.language, 'ar')
        self.assertIsNone(model.detect_language.call_args.kwargs['generation_config'].language)

    def test_complete_long_features_keep_original_path(self):
        model = self.model()
        batch = {'input_features': np.zeros((1, 128, 3001)), 'attention_mask': object()}
        language, parameters, reused = prepare_original(model, batch)
        self.assertFalse(reused)
        self.assertEqual(language, 'ar')
        self.assertIs(parameters['input_features'], batch['input_features'])
        model.get_encoder.assert_not_called()

    def test_unknown_language_is_not_silently_assigned(self):
        model = self.model()
        model.detect_language.return_value = [8]
        language, _, _ = prepare_original(model, {'input_features': np.zeros((1, 128, 3000))})
        self.assertIsNone(language)


if __name__ == '__main__':
    unittest.main()
