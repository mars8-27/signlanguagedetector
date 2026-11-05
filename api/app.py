import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from src.config import cfg
from src.extract_landmarks import normalize_skeleton_frame
from src.models.sign_classifier import STGCNSignClassifier

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
ONNX_PATH = cfg.CHECKPOINT_DIR / "sign_stgcn.onnx"
PT_PATH = cfg.CHECKPOINT_DIR / "best_stgcn.pt"

ort_session = None
pt_model = None

def get_engine():
    global ort_session, pt_model
    if ort_session is not None:
        return "onnx"
    if pt_model is not None:
        return "pytorch"

    if HAS_ORT and ONNX_PATH.exists():
        try:
            ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
            print(f"[INFO] Engine Loaded: ONNX Runtime ({ONNX_PATH.name})")
            return "onnx"
        except Exception as e:
            print(f"[WARN] Failed to load ONNX: {e}")

    if PT_PATH.exists():
        try:
            model = STGCNSignClassifier()
            state = torch.load(PT_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.eval()
            pt_model = model
            print(f"[INFO] Engine Loaded: PyTorch ST-GCN ({PT_PATH.name})")
            return "pytorch"
        except Exception as e:
            print(f"[WARN] Failed to load PyTorch: {e}")

    return "none"

get_engine()

app = FastAPI(title="Sign Language ST-GCN Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def serve_index():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(str(FRONTEND_FILE))

@app.get("/health")
def health_status():
    return {"status": "healthy", "engine": get_engine(), "classes": cfg.CLASSES}

def run_inference(sequence_ctv: np.ndarray) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    engine = get_engine()

    if engine == "onnx" and ort_session is not None:
        input_tensor = np.expand_dims(sequence_ctv, axis=0).astype(np.float32)
        logits = ort_session.run(None, {"skeleton_sequence": input_tensor})[0][0]
    elif engine == "pytorch" and pt_model is not None:
        tensor_in = torch.from_numpy(sequence_ctv).unsqueeze(0).float()
        with torch.no_grad():
            logits = pt_model(tensor_in)[0].numpy()
    else:
        return "MODEL_NOT_FOUND", 0.0, 0.0

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    top_idx = int(np.argmax(probs))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    sign = cfg.CLASSES[top_idx] if top_idx < len(cfg.CLASSES) else "UNKNOWN"
    return sign, round(float(probs[top_idx]), 4), round(latency_ms, 2)

@app.websocket("/ws/translate")
async def websocket_sign_stream(websocket: WebSocket):
    await websocket.accept()
    frame_buffer = deque(maxlen=cfg.SEQ_LENGTH)

    try:
        while True:
            data = await websocket.receive_json()
            raw_landmarks = np.array(data.get("landmarks", []), dtype=np.float32)

            if raw_landmarks.shape != (cfg.NUM_NODES, cfg.IN_CHANNELS):
                continue

            norm_landmarks = normalize_skeleton_frame(raw_landmarks)
            frame_buffer.append(norm_landmarks)

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            if len(frame_buffer) < cfg.SEQ_LENGTH:
                await websocket.send_json({
                    "predicted_sign": f"BUFFERING ({len(frame_buffer)}/{cfg.SEQ_LENGTH})",
                    "confidence": 0.0,
                    "latency_ms": 0.0,
                    "buffer_fill": len(frame_buffer)
                })
            else:
                seq_tvc = np.stack(frame_buffer, axis=0)
                seq_ctv = np.transpose(seq_tvc, (2, 0, 1))
                sign, conf, latency = run_inference(seq_ctv)

                await websocket.send_json({
                    "predicted_sign": sign,
                    "confidence": conf,
                    "latency_ms": latency,
                    "buffer_fill": len(frame_buffer)
                })
    except (WebSocketDisconnect, ConnectionResetError, RuntimeError):
        pass
    finally:
        frame_buffer.clear()