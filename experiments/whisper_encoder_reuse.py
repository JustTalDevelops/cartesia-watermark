"""Prototype reuse of one FP32 encoder result for short-source language/text.

The frozen codec runtime does not import this helper. Inputs longer than the
model's 30-second encoder window retain the existing complete-input path.
The caller owns evaluation/inference mode and must consume returned features
immediately; this helper writes no source data or recognized text.
"""
import copy


def prepare_original(model, batch, reuse=True, maximum_frames=3000):
    configuration = copy.deepcopy(model.generation_config)
    configuration.language = None
    if reuse and batch['input_features'].shape[-1] <= maximum_frames:
        encoded = model.get_encoder()(batch['input_features'], return_dict=True)
        token = int(model.detect_language(encoder_outputs=encoded, generation_config=configuration)[0])
        parameters = {'encoder_outputs': encoded}
        if 'attention_mask' in batch:
            parameters['attention_mask'] = batch['attention_mask']
        reused = True
    else:
        token = int(model.detect_language(input_features=batch['input_features'], generation_config=configuration)[0])
        parameters = dict(batch)
        reused = False
    tag = next((name for name, value in configuration.lang_to_id.items() if value == token), '')
    language = tag[2:-2] if tag.startswith('<|') and tag.endswith('|>') else None
    return language, parameters, reused
