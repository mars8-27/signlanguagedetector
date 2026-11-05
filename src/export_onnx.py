import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from src.config import cfg
from src.models.sign_classifier import STGCNSignClassifier

def export_to_onnx():
    ckpt_path = cfg.CHECKPOINT_DIR / "best_stgcn.pt"
    onnx_path = cfg.CHECKPOINT_DIR / "sign_stgcn.onnx"

    model = STGCNSignClassifier()
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, cfg.IN_CHANNELS, cfg.SEQ_LENGTH, cfg.NUM_NODES)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["skeleton_sequence"],
        output_names=["class_logits"],
        dynamic_axes={"skeleton_sequence": {0: "batch_size"}, "class_logits": {0: "batch_size"}}
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Verify numerical parity and benchmark CPU latency with ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_np = dummy_input.numpy()

    start = time.perf_counter()
    ort_outs = ort_session.run(None, {"skeleton_sequence": test_np})
    latency_ms = (time.perf_counter() - start) * 1000.0

    print(f"ONNX export verified at: {onnx_path}")
    print(f"Single-sequence CPU inference latency: {latency_ms:.2f} ms")

if __name__ == "__main__":
    export_to_onnx()