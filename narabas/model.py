import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models.wav2vec2.model import wav2vec2_model

from narabas.symbols import id_to_phoneme, phoneme_symbols, PAD, BOS, EOS


class Narabas(LightningModule):
    def __init__(self, dim_out=len(phoneme_symbols), load_base_parameters=True):
        super().__init__()
        self.save_hyperparameters()

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
        bundle._params["aux_num_out"] = dim_out
        model = wav2vec2_model(**bundle._params)

        if load_base_parameters:
            # Do almost what bundle.get_model() does, but we need to specify `strict=False` to load the state dict because we want use different `aux_num_out` from that of the saved state.
            # Is their any better way to do this?
            state_dict = bundle._get_state_dict(None)
            del state_dict["aux.weight"]
            del state_dict["aux.bias"]
            model.load_state_dict(state_dict, strict=False)
            torch.nn.init.xavier_uniform_(model.aux.weight)
            torch.nn.init.zeros_(model.aux.bias)

        model.feature_extractor.requires_grad_(False)
        self.wav2vec2 = model
        self.hop_length = np.prod(
            [l[-1] for l in bundle._params["extractor_conv_layer_config"]]
        )  # 320
        self.sample_rate = bundle._sample_rate

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor = None) -> torch.Tensor:
        # wav: (B, T)
        return self.wav2vec2(wav, wav_lens)

    def training_step(self, batch, batch_idx):
        x, x_lens, y, y_lens = (
            batch["wav"],
            batch["wav_lens"],
            batch["phn"],
            batch["phn_lens"],
        )
        y_hat, y_hat_lens = self(x, x_lens)

        log_probs = F.log_softmax(y_hat, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (B, T, C) -> (T, B, C)

        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=y,
            input_lengths=y_hat_lens,
            target_lengths=y_lens,
            blank=PAD,
            reduction="mean",
            zero_infinity=True,
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_lens, y, y_lens = (
            batch["wav"],
            batch["wav_lens"],
            batch["phn"],
            batch["phn_lens"],
        )
        y_hat, y_hat_lens = self(x, x_lens)

        log_probs = F.log_softmax(y_hat, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (B, T, C) -> (T, B, C)

        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=y,
            input_lengths=y_hat_lens,
            target_lengths=y_lens,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        self.log("val_loss", loss, prog_bar=True)

        predictions = []
        references = []
        for n in range(y_hat.size(0)):
            _, hyp_per_frame = torch.max(y_hat[n], dim=1)
            hyp_ids = self._ctc_decode(hyp_per_frame.cpu().numpy())
            ref_ids = y[n, : y_lens[n].item()].cpu().numpy()

            hyp = [id_to_phoneme[i] for i in hyp_ids]
            ref = [id_to_phoneme[i] for i in ref_ids]
            hyp_str = " ".join(hyp)
            ref_str = " ".join(ref)
            predictions.append(hyp_str)
            references.append(ref_str)

        return {"loss": loss, "predictions": predictions, "references": references}

    def _ctc_decode(self, hyp_per_frame: np.ndarray):
        return [
            cur
            for cur, prev in zip(hyp_per_frame, [None, *hyp_per_frame[:-1]])
            if cur != prev and cur != 0
        ]

    def validation_epoch_end(self, validation_step_outputs):
        predictions = []
        references = []
        for batch in validation_step_outputs:
            predictions.extend(batch["predictions"])
            references.extend(batch["references"])

        text = ""
        for _, hyp, ref in zip(range(10), predictions, references):
            text += f"REF: {ref}\n\n"
            text += f"HYP: {hyp}\n\n"
            text += "----\n"

        # Calculate the phoneme error rate by computing the word error rate for a space-separated sequence of phonemes.
        phoneme_error_rate = jiwer.wer(references, predictions)
        self.log(
            "val_phoneme_error_rate",
            phoneme_error_rate,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("hp_metric", phoneme_error_rate)
        self.logger.experiment.add_text("val", text, self.global_step)
