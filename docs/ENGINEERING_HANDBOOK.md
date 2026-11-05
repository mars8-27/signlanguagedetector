# Engineering Handbook

This handbook documents the implementation in this repository. It is intended to help developers understand, run, debug, extend, and discuss the project.

> **Scope:** This is a fixed-vocabulary gesture classifier prototype. Its included dataset is procedurally synthesized. The repository does not establish real-world ASL accuracy or continuous sign-language translation.

## 1. System overview

The browser captures webcam frames and runs MediaPipe Holistic locally. JavaScript selects 53 landmarks and sends x/y/z coordinates to FastAPI over a WebSocket. The backend normalizes each frame, accumulates a 30-frame rolling window, and runs a custom Spatio-Temporal Graph Convolutional Network (ST-GCN). ONNX Runtime CPU is preferred when the ONNX artifact is available; otherwise the backend attempts to load the PyTorch checkpoint.

The API returns a predicted class, softmax confidence, inference latency, and buffer fill.

## 2. Data contract

### Landmark layout

| Node indices | Content |
|---|---|
| 0 | Nose |
| 1–2 | Left and right mouth landmarks |
| 3–4 | Left and right shoulders |
| 5–6 | Left and right elbows |
| 7–8 | Left and right wrists |
| 9–10 | Left and right hips |
| 11–31 | 21 left-hand landmarks |
| 32–52 | 21 right-hand landmarks |

The exact pose source indices are defined in `src/graph/skeleton.py`. Hand blocks retain MediaPipe hand-landmark ordering.

### Tensor dimensions

| Stage | Shape |
|---|---|
| One frame | `(53, 3)` |
| Sequence | `(30, 53, 3)` |
| One model input | `(1, 3, 30, 53)` |
| Model output | `(1, 10)` |

The central invariant is **53 ordered nodes, three coordinate channels, and 30 frames**. Changing it requires coordinated updates to frontend extraction, normalization, graph topology, training data, and model configuration.

## 3. Runtime flow

1. Browser requests webcam permission through `getUserMedia`.
2. MediaPipe Holistic extracts pose and hand landmarks in the browser.
3. The frontend selects 11 pose landmarks and both 21-point hands.
4. It sends a `landmarks` array to `/ws/translate`.
5. FastAPI converts the array to float32 and checks for shape `(53, 3)`.
6. The backend applies shoulder-centred normalization.
7. A deque retains the latest 30 accepted frames.
8. The window is stacked and transposed to `(1, 3, 30, 53)`.
9. ONNX Runtime or PyTorch produces ten logits.
10. Softmax and argmax produce a label and confidence.
11. The API sends the prediction response to the browser.

Reported latency is measured inside the inference function. It excludes webcam capture, MediaPipe processing, network transit, and browser rendering.

## 4. File-by-file guide

### Configuration and graph

- **`src/config.py`**: shared configuration for node count (53), input channels (3), sequence length (30), class labels, training hyperparameters, and data/checkpoint paths. Treat it as the source of truth for dimensions and class order.
- **`src/graph/skeleton.py`**: maps MediaPipe pose indices to the selected 11 pose nodes and defines anatomical graph edges.
- **`src/graph/adjacency.py`**: builds a two-partition adjacency tensor `(2, 53, 53)`: identity/self connections and degree-normalized neighbor connections. The neighbor matrix uses symmetric normalization `D^(-1/2) A D^(-1/2)`.

### Model

- **`src/models/st_gcn_block.py`**: implements spatial graph convolution, temporal convolution, batch normalization, SiLU activation, dropout, residual paths, and trainable edge-importance weights.
- **`src/models/sign_classifier.py`**: defines the classifier stages with channels 3→64→64→128→256, temporal downsampling in later blocks, global average pooling, and a final 10-logit linear layer.

### Data and training

- **`src/extract_landmarks.py`**: shared shoulder-centred normalization and offline video extraction. Normalization subtracts the shoulder midpoint and divides by shoulder distance. Live extraction is performed in frontend JavaScript.
- **`generate_dataset.py`**: main procedural generator. It creates 100 samples per configured class, 30 frames per sample, 53 nodes, and 3 coordinates. It synthesizes simplified hand geometry and class-specific wrist/hand trajectories, adds coordinate jitter, normalizes, and saves `X.npy` and `y.npy`.
- **`src/dataset.py`**: PyTorch dataset wrapper and augmentation (Y-axis rotation and Gaussian noise). It also contains a separate fallback synthetic generator used if processed arrays are absent. This fallback uses simpler class-dependent sinusoidal patterns and is not the same algorithm as the root generator.
- **`src/train.py`**: loads processed arrays or uses the fallback generator, performs a stratified 80/20 split, trains with cross-entropy and label smoothing, AdamW, cosine learning-rate scheduling, and gradient clipping, then saves the best model state dictionary to `checkpoints/best_stgcn.pt`. It does not save optimizer/scheduler state for exact training resumption.
- **`src/export_onnx.py`**: loads the best PyTorch state, exports ONNX opset 17 with input `(1, 3, 30, 53)`, checks the graph, and creates an ONNX Runtime CPU session for a sample inference. Keep `sign_stgcn.onnx` and `sign_stgcn.onnx.data` together.

### API and frontend

- **`api/app.py`**: selects the inference engine, serves `/` and `/health`, accepts WebSocket frames at `/ws/translate`, maintains the rolling window, runs inference, and returns the prediction payload. ONNX Runtime is preferred, with PyTorch as fallback.
- **`api/schemas.py`**: defines Pydantic frame and response shapes. The current WebSocket handler parses JSON directly and does not use these models to enforce its incoming payload schema.
- **`frontend/index.html`**: browser UI, webcam access, MediaPipe Holistic integration, landmark selection, canvas visualization, WebSocket client, and prediction display. The canvas currently draws landmark points, not a full anatomical bone overlay.

### Tests and deployment

- **`tests/test_pipeline.py`**: tests adjacency shape/symmetry, normalization invariance, and model output shape. It does not currently cover WebSocket integration, browser behavior, ONNX parity, Docker packaging, or real-world recognition quality.
- **`Dockerfile`**: builds a Python 3.11 slim image, installs system libraries and requirements, copies source/API/frontend/checkpoints, and starts Uvicorn on port 8000.
- **`requirements.txt`**: specifies minimum dependency versions rather than a fully locked environment. Pin a tested dependency set for reproducible builds.
- **`pytest.ini`**: adds the repository root to pytest's Python path. Keep it UTF-8 without a byte-order mark.

## 5. Model intuition

A sign gesture has spatial and temporal structure. Spatial structure describes how joints relate; temporal structure describes how they move. A graph represents joints as nodes and anatomical connections as edges. Spatial graph convolution aggregates information from connected nodes, while temporal convolution learns motion patterns across the 30-frame sequence. Residual connections provide shortcut paths through the network.

The model returns logits. The API applies softmax to obtain class probabilities and selects the largest probability.

## 6. Training and inference commands

Python 3.11 is the recommended baseline because the Docker image uses Python 3.11.

```bash
python -m pip install -r requirements.txt
python generate_dataset.py
python -m src.train
python -m src.export_onnx
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

The repository already includes processed data and model artifacts. Dataset generation, training, and export are only needed to reproduce those stages. Run tests with `python -m pytest`.

## 7. Review findings and limitations

| Finding | Recommendation |
|---|---|
| Dataset is synthetic | Do not claim real-signer recognition accuracy. Collect consented real data and evaluate on held-out signers. |
| WebSocket payload validation is basic | Validate nested values, finite coordinates, and missing landmarks; return structured errors. |
| Pydantic schemas are not used by the current handler | Apply them or equivalent explicit validation to the WebSocket contract. |
| Browser WebSocket scheme must match deployment | Use `wss://` when the page is served over HTTPS. |
| Dependency versions are minimum-bounded | Pin a tested set for reproducible installs. |
| No license has been selected | Do not claim MIT or unrestricted reuse until a license file is added. |

The full pytest suite was not verified in the review environment: pytest first rejected the BOM-prefixed configuration; bypassing it then exposed a missing MediaPipe dependency during collection. A full run after installing all requirements is still needed.

## 8. Evaluation requirements

1. Collect real, consented recordings from multiple signers.
2. Define a repeatable recording protocol and document camera/lighting conditions.
3. Split by signer so test signers are not present in training.
4. Report per-class precision, recall, F1, confusion matrix, and confidence calibration.
5. Evaluate occlusion, missing hands, camera position, and varied signing speeds.
6. Compare ONNX and PyTorch predictions on identical inputs.

## 9. Interview-ready explanation

**30-second overview:** “I built a webcam-based fixed-vocabulary sign-gesture classifier. MediaPipe Holistic extracts body and hand landmarks in the browser. The browser streams 53 three-dimensional landmarks per frame to a FastAPI WebSocket. The backend normalizes coordinates, maintains a 30-frame sequence, and passes a `(1, 3, 30, 53)` tensor into a Spatio-Temporal Graph Convolutional Network. The model predicts one of ten configured labels. The service prefers ONNX Runtime CPU inference and falls back to a PyTorch checkpoint.”

**Common questions:**

1. **Why landmarks instead of raw images?** Landmarks provide a compact structured representation and reduce dependence on background pixels, although landmark quality still depends on pose estimation.
2. **Why ST-GCN?** It models anatomical relationships as a graph and movement over time with temporal convolution.
3. **What does adjacency represent?** Self-connections and anatomical neighbor relationships between the 53 nodes.
4. **Why shoulder normalization?** It centres the skeleton at the shoulder midpoint and scales by shoulder distance, reducing translation and apparent-size variation.
5. **Why ONNX?** It provides a portable inference representation, used here with ONNX Runtime's CPU provider.
6. **Does validation prove ASL accuracy?** No. The supplied data is synthetic; real signer-separated evaluation is required.
7. **Is this a complete translator?** No. It predicts one of ten labels for a rolling window, not continuous grammar-aware sentences.
8. **What would you improve first?** Strengthen payload and missing-landmark handling, add API and ONNX tests, and evaluate on real multi-signer data.

## 10. Mental model

Remember five layers: **perception** (MediaPipe landmarks), **representation** (normalized 53-node sequence), **learning** (ST-GCN), **serving** (FastAPI plus ONNX/PyTorch), and **presentation** (browser prediction display). Preserve the data contract: 53 ordered nodes, three coordinates, 30 frames, and model input `(B, 3, 30, 53)`.
