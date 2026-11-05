import cv2
import numpy as np
import mediapipe as mp
from src.config import cfg
from src.graph.skeleton import POSE_INDICES

def normalize_skeleton_frame(landmarks_53x3: np.ndarray) -> np.ndarray:
    """
    Centers coordinates at the mid-shoulder origin and scales by shoulder span.
    Input shape: (53, 3) -> Output shape: (53, 3)
    """
    left_shoulder = landmarks_53x3[3]   # Mapped index 3 in POSE_INDICES
    right_shoulder = landmarks_53x3[4]  # Mapped index 4 in POSE_INDICES
    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
    normalized = (landmarks_53x3 - mid_shoulder) / shoulder_dist
    return normalized.astype(np.float32)

def extract_53_nodes_from_holistic(results) -> np.ndarray:
    """Extracts (53, 3) array from MediaPipe Holistic result object."""
    frame_nodes = np.zeros((cfg.NUM_NODES, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, mp_idx in enumerate(POSE_INDICES):
            lm = results.pose_landmarks.landmark[mp_idx]
            frame_nodes[i] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_nodes[11 + i] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_nodes[32 + i] = [lm.x, lm.y, lm.z]

    return normalize_skeleton_frame(frame_nodes)

def extract_video_to_tensor(video_path: str, seq_len: int = cfg.SEQ_LENGTH) -> np.ndarray:
    """
    Reads an RGB video file, extracts normalized skeletal coordinates, and
    resamples temporally to fixed shape (C=3, T=seq_len, V=53).
    """
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    frames_list = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            frames_list.append(extract_53_nodes_from_holistic(results))

    cap.release()

    if not frames_list:
        return np.zeros((cfg.IN_CHANNELS, seq_len, cfg.NUM_NODES), dtype=np.float32)

    raw_seq = np.stack(frames_list, axis=0)  # (T_raw, V, C)
    # Temporal linear resampling to fixed T=seq_len
    indices = np.linspace(0, len(raw_seq) - 1, seq_len).astype(int)
    sampled_seq = raw_seq[indices]           # (T, V, C)

    # Transpose to PyTorch ST-GCN standard format: (C, T, V)
    return np.transpose(sampled_seq, (2, 0, 1)).astype(np.float32)