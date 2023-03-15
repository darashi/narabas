from glob import glob

import onnx
import torch

from narabas.model import Narabas

# TODO fix sorting
state_dict_path = sorted(glob("./lightning_logs/version_*/checkpoints/epoch=*.ckpt"))[-1]
print(f"Loading state dict from {state_dict_path}")
model = Narabas.load_from_checkpoint(state_dict_path)

dest_path = "narabas.onnx"

model.eval()

dummy_input = torch.randn(1, 16000)

torch.onnx.export(
    model=model,
    args=dummy_input,
    f=dest_path,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    },
    do_constant_folding=True,
)
print(f"Exported to {dest_path}")

onnx_model = onnx.load(dest_path)

# add metadata
print("Adding metadata ...")
onnx.helper.set_model_props(
    onnx_model,
    {
        "sample_rate": str(model.sample_rate),
        "hop_length": str(model.hop_length),
    },
)

# check model
onnx.checker.check_model(onnx_model)

onnx.save(onnx_model, dest_path)

size = onnx_model.ByteSize()
print("Done! model size: {:.1f} MB".format(size / (1024 * 1024)))
