# narabas: Japanese phoneme forced alignment tool

⚠ This is an experimental implementation. ⚠

narabas is a tool for Japanese forced alignment, based on the [Wav2Vec2](https://arxiv.org/abs/2006.11477) model. It comes with pre-trained models trained on the [ReazonSpeech corpus](https://research.reazon.jp/projects/ReazonSpeech/index.html). Using narabas, you can easily obtain Japanese phoneme alignments.

Phoneme sequences generated by [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) from text are used for training as grandtruth phoneme sequences. No distinction is made between voiced and voiceless vowels.

Usage:

```python
import narabas

narabas.load_model()

segments = model.align(
    "tests/fixtures/meian_1413.wav",
    "a n e m u s u m e n o ts u i k o w a",
)

for (start, end, phoneme) in segments:
    print(f"{start:.3f} {end:.3f} {phoneme}")
```

Output:

```text
0.220 0.300 a
0.300 0.320 n
0.320 0.400 e
0.400 0.440 m
0.440 0.480 u
0.480 0.520 s
0.520 0.580 u
0.580 0.620 m
0.620 0.680 e
0.680 0.720 n
0.720 0.860 o
0.860 0.880 ts
0.880 0.980 u
0.980 1.060 i
1.060 1.100 k
1.100 1.160 o
1.160 1.220 w
1.220 1.620 a
```


## About pretrained model

The pre-trained model is based on [WAV2VEC2_ASR_LARGE_LV60K_960H](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H.html) and trained using the ReazonSpeech corpus.

The model was trained on the author's handmade PC with a single RTX 3090, so the number of steps is probably not sufficient and no extensive hyperparameter exploration has been done (740k steps trained). When used as a naive phoneme identifier, the phoneme error rate is about 6% (using the [Mozilla Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) validation split as the validation set, with accuracy involving EOS tokens / BOS tokens).

If you have huge computing power, it would be great if you could provide the model you have trained on.

## Training

    $ poetry install
    $ poetry shell
    $ python trainer.py fit
    $ python export-onnx.py

## Development

    $ pytest