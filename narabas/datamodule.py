from typing import List
import pyopenjtalk

import datasets
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from narabas.symbols import phoneme_to_id, BOS, EOS


def text_to_phn_ids(text: str) -> List[int]:
    phns = pyopenjtalk.g2p(text)
    if phns == "":
        return []
    phns = phns.split(" ")
    # Do not distinguish between voiced and voiceless vowels
    phns = [
        phn.lower()
        if phn in {"A", "I", "U", "E", "O", "[PAD]", "[BOS]", "[EOS]"}
        else phn
        for phn in phns
    ]
    phn_ids = [phoneme_to_id[phn] for phn in phns]
    return [BOS, *phn_ids, EOS]


def _drop_long_audio(example, threshold_sec=15.0):
    length_sec = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return length_sec < threshold_sec


def _drop_phns_error(example):
    # some texts are failed to convert to phonemes; drop them
    return len(example["phn"]) > 0


def _preprocess(item, sampling_rate=16_000):
    wav, sr_ = item["audio"]["array"], item["audio"]["sampling_rate"]
    if sr_ != sampling_rate:
        wav = torchaudio.functional.resample(wav, sr_, sampling_rate)
    phn_ids = torch.tensor(text_to_phn_ids(item["transcription"]))

    return {
        "wav": wav,
        "phn": phn_ids,
    }


def _collate_fn(items):
    assert items[0]["phn"][0] == BOS, f"{items[0]['phn'][0]}"

    wav_lens = torch.tensor([len(item["wav"]) for item in items])
    wavs = [
        torch.from_numpy(item["wav"]).float()
        if type(item["wav"]).__module__ == "numpy"
        else item["wav"]
        for item in items
    ]
    wav = pad_sequence(wavs, batch_first=True)
    phn_lens = torch.tensor([len(item["phn"]) for item in items])
    phn = pad_sequence(
        [item["phn"] for item in items], padding_value=-100, batch_first=True
    )
    return {
        "wav": wav,
        "wav_lens": wav_lens,
        "phn": phn,
        "phn_lens": phn_lens,
    }


class NarabasDataModule(LightningDataModule):
    def __init__(self, name: str, batch_size: int, num_workers: int):
        super().__init__()
        self.save_hyperparameters()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        ds = (
            load_dataset(
                "reazon-research/reazonspeech",
                name=self.name,
                split=datasets.Split.TRAIN,
                num_proc=self.num_workers,
            )
            .with_format("torch")
            .filter(_drop_long_audio)
            .map(_preprocess)
            .filter(_drop_phns_error)
        )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            name="ja",
            split=datasets.Split.VALIDATION,
            num_proc=self.num_workers,
        )
        ds.set_format("torch")
        ds = (
            ds.filter(_drop_long_audio)
            .rename_column("sentence", "transcription")
            .map(_preprocess)
        )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
        )
