# Model origins and third-party notices

The release distinguishes independently trained heads/editors from fine-tuned upstream backbones. It does not contain Resemble Enhance's reconstruction weights or the original Cartesia detector.

## XLS-R

The `encoder_oracle_*.pt` research checkpoints adapt [Facebook/Meta's Wav2Vec2 XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m), credited to Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, and Michael Auli. The source model card identifies Apache-2.0 licensing; a copy is included as `XLSR_APACHE_2_0.txt`.

These checkpoints were modified by truncating the feature encoder used by this project and adapting feature-extractor/head weights to Cartesia watermark labels. They are not the unchanged upstream model and do not imply Meta endorsement. The independently trained convolutional heads use XLS-R features but do not bundle the upstream backbone.

## WavLM

The `finetune_best.pt` and `finetune_r2_best.pt` checkpoints adapt [Microsoft WavLM Base Plus](https://huggingface.co/microsoft/wavlm-base-plus). Credit: Microsoft and the WavLM authors listed in the source model card, including Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Shuo Ren, Yanmin Qian, Yao Qian, Michael Zeng, and Furu Wei.

At packaging, the model card's official-license link points to the [Microsoft UniSpeech Attribution-ShareAlike 3.0 Unported license](https://github.com/microsoft/UniSpeech/blob/main/LICENSE). The included WavLM fine-tuned weights are distributed under that same CC BY-SA 3.0 license; its full text is included as `WAVLM_CC_BY_SA_3_0.txt`.

Changes: the encoder is truncated for this project's feature extraction, its convolutional extractor/first transformer layer are fine-tuned on Cartesia speech, and a trained watermark classification head is added. These adaptations are by the cartesia-watermark project; they are not an official Microsoft model or endorsement. Independently trained WavLM feature heads do not bundle WavLM's backbone weights.

## AudioSeal

The `raw_oracle_*.pt` checkpoints start from the [Meta AudioSeal detector](https://huggingface.co/facebook/audioseal). AudioSeal is copyright Meta Platforms, Inc. and affiliates, under the MIT license reproduced as `AUDIOSEAL_MIT.txt`.

Changes: the watermark/message head is replaced with two Cartesia-label classes and the classifier is fine-tuned. These are research teachers for this project, not stock AudioSeal detectors and not Cartesia's own model. No upstream endorsement is implied.

## Package scope

These notices describe the included upstream-derived model components; they do not assign the rest of the repository a project-wide license. Original upstream code, separately downloaded backbone initialization files, and reconstruction assets retain their own terms. `models.json` identifies each packaged checkpoint and its loader. Training data and personal recordings are not included.
