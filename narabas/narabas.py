import os
from posixpath import expanduser
import numpy as np
import torch
import torchaudio
from narabas.symbols import BOS, EOS, PAD, id_to_phoneme, phoneme_to_id
import onnxruntime as ort


class Narabas:
    def __init__(self, device="cpu", model_dir=expanduser("~/.cache/narabas")) -> None:
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        self.model_dir = model_dir
        self.sess = ort.InferenceSession(self.prepare_model(), providers=providers)

        meta = self.sess.get_modelmeta().custom_metadata_map
        self.sample_rate = int(meta["sample_rate"])
        self.hop_length = int(meta["hop_length"])
        self.hop_length_sec = self.hop_length / self.sample_rate

    def prepare_model(self):
        os.makedirs(self.model_dir, exist_ok=True, mode=0o755)
        dest_path = os.path.join(self.model_dir, "narabas-v0.onnx")
        if not os.path.exists(dest_path):
            torch.hub.download_url_to_file(
                "https://github.com/darashi/narabas-models/releases/download/v0/narabas-v0.onnx",
                dest_path,
                progress=True,
            )
            print("model saved to", dest_path)
        return dest_path

    def _ctc_decode(self, hyp_per_frame: np.ndarray):
        return [
            cur
            for cur, prev in zip(hyp_per_frame, [None, *hyp_per_frame[:-1]])
            if cur != prev and cur != 0
        ]

    def load_audio(self, audio_path: str):
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            raise ValueError("Only mono audio is supported")

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Since our model requires a batch dimension,
        # treat the unneeded audio channel dimension (1) as the batch dimension
        return wav

    def transcribe(self, audio_path: str):
        wav = self.load_audio(audio_path)
        (y_hat,) = self.sess.run(["output"], {"input": wav.numpy()})
        hyp_per_frame = np.argmax(y_hat.squeeze(0), axis=-1)
        phn_ids = self._ctc_decode(hyp_per_frame)
        phn = " ".join([id_to_phoneme[phn_id] for phn_id in phn_ids])

        return phn

    def align(self, audio_path: str, phns: str):
        phn_ids = [phoneme_to_id[phn] for phn in phns.split()]
        phn_ids = [BOS, *phn_ids, EOS]

        wav = self.load_audio(audio_path)
        (y_hat,) = self.sess.run(["output"], {"input": wav.numpy()})
        y_hat = torch.from_numpy(y_hat)
        emission = torch.log_softmax(y_hat.squeeze(0), dim=-1)  # (frames, phns)

        num_frames = emission.shape[0]
        num_tokens = len(phn_ids)

        likelihood = np.full((num_tokens + 1,), -np.Inf)  # (tokens,)
        likelihood[0] = 0

        path = np.zeros((num_frames, num_tokens + 1), dtype=np.int32)

        # NOTE are the indices t, i handled correctly at both ends? And do we really need [BOS] and [EOS]?
        for t in range(0, num_frames):
            for i in range(1, num_tokens + 1):
                stay = likelihood[i] + emission[t, PAD]
                move = likelihood[i - 1] + emission[t, phn_ids[i - 1]]
                if stay > move:
                    path[t][i] = 0
                else:
                    path[t][i] = 1

                likelihood[i] = np.max([stay, move])

        # do backtracing
        alignment = []
        t = num_frames - 1
        i = num_tokens
        while t >= 0:
            if path[t][i] == 1:
                i -= 1
                alignment.append((t, i))
            t -= 1
        alignment = alignment[-2::-1]  # drop [BOS] and reverse

        segments = []
        for (t, i), (t_next, _i_next) in zip(alignment, alignment[1:]):
            start = t * self.hop_length_sec
            end = t_next * self.hop_length_sec
            token = id_to_phoneme[phn_ids[i]]
            segments.append((start, end, token))

        return segments
