import numpy as np
from pathlib import Path
from src.config import cfg
from src.extract_landmarks import normalize_skeleton_frame

NUM_SAMPLES_PER_CLASS = 100
SEQ_LEN = cfg.SEQ_LENGTH     # 30 frames
NUM_NODES = cfg.NUM_NODES    # 53 landmarks

def build_hand(wrist_pos, palm_dir, finger_states, hand_scale=0.08):
    """
    Synthesizes 21 MediaPipe hand landmarks based on wrist position and finger states.
    finger_states: list of 5 floats [thumb, index, middle, ring, pinky] (1.0 = extended, 0.0 = curled)
    """
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist_pos

    # Relative base angles for fingers around palm
    finger_angles = [-0.35, -0.15, 0.0, 0.15, 0.32]
    finger_lengths = [0.8, 1.0, 1.05, 0.95, 0.75]

    for f_idx in range(5):
        ext = finger_states[f_idx]
        angle = finger_angles[f_idx]
        length = finger_lengths[f_idx] * hand_scale

        base_idx = 1 + f_idx * 4
        # Root knuckle (MCP)
        dx = length * 0.35 * np.sin(angle)
        dy = -length * 0.35 * np.cos(angle)
        hand[base_idx] = wrist_pos + [dx, dy * palm_dir, 0.0]

        # Knuckle segments extending outward or curling inwards
        for seg in range(1, 4):
            frac = (seg + 1) / 4.0
            if ext > 0.5:
                # Extended outward
                seg_x = length * frac * np.sin(angle)
                seg_y = -length * frac * np.cos(angle) * palm_dir
                seg_z = -0.01 * seg
            else:
                # Curled into palm
                seg_x = length * 0.35 * np.sin(angle)
                seg_y = (-length * 0.35 + (length * 0.3 * frac)) * palm_dir
                seg_z = -0.02 * seg
            hand[base_idx + seg] = wrist_pos + [seg_x, seg_y, seg_z]

    return hand

def synthesize_sequence(class_idx: int) -> np.ndarray:
    """Generates a (30, 53, 3) sequence of MediaPipe normalized skeleton landmarks."""
    # Anthropometric variations per person
    shoulder_w = np.random.uniform(0.22, 0.26)
    torso_h = np.random.uniform(0.38, 0.44)
    speed_factor = np.random.uniform(0.85, 1.15)
    noise_lvl = 0.004

    seq = np.zeros((SEQ_LEN, NUM_NODES, 3), dtype=np.float32)

    # Base body landmark anchors
    nose = np.array([0.50, 0.22, 0.0], dtype=np.float32)
    mouth_l = np.array([0.52, 0.30, -0.02], dtype=np.float32)
    mouth_r = np.array([0.48, 0.30, -0.02], dtype=np.float32)
    sh_r = np.array([0.50 - shoulder_w / 2, 0.42, 0.0], dtype=np.float32)
    sh_l = np.array([0.50 + shoulder_w / 2, 0.42, 0.0], dtype=np.float32)
    hip_r = np.array([0.50 - shoulder_w / 2.5, 0.42 + torso_h, 0.0], dtype=np.float32)
    hip_l = np.array([0.50 + shoulder_w / 2.5, 0.42 + torso_h, 0.0], dtype=np.float32)

    for t in range(SEQ_LEN):
        progress = (t / (SEQ_LEN - 1)) * speed_factor
        progress = np.clip(progress, 0.0, 1.0)

        # Default resting hand positions
        rw = np.array([0.35, 0.78, 0.0], dtype=np.float32)
        lw = np.array([0.65, 0.78, 0.0], dtype=np.float32)
        r_fingers = [0.0, 0.0, 0.0, 0.0, 0.0]
        l_fingers = [0.0, 0.0, 0.0, 0.0, 0.0]
        palm_r, palm_l = 1.0, 1.0

        # Class-specific kinematic trajectory formulations
        if class_idx == 0:  # HELLO (Right hand salutes from temple outward)
            rw = np.array([0.38 - 0.18 * progress, 0.24 - 0.06 * progress, -0.05])
            r_fingers = [1.0, 1.0, 1.0, 1.0, 1.0]

        elif class_idx == 1:  # THANK_YOU (Right hand moves from chin forward/down)
            rw = np.array([0.50, 0.32 + 0.18 * progress, -0.05 - 0.20 * progress])
            r_fingers = [1.0, 1.0, 1.0, 1.0, 1.0]

        elif class_idx == 2:  # HELP (Left palm flat, right thumbs-up on top, both rise)
            rise = 0.12 * progress
            lw = np.array([0.52, 0.54 - rise, -0.08])
            rw = np.array([0.50, 0.48 - rise, -0.08])
            l_fingers = [1.0, 1.0, 1.0, 1.0, 1.0]
            r_fingers = [1.0, 0.0, 0.0, 0.0, 0.0]

        elif class_idx == 3:  # DOCTOR (Right hand taps left wrist twice)
            lw = np.array([0.55, 0.52, -0.10])
            tap = 0.04 * np.abs(np.sin(progress * 4 * np.pi))
            rw = np.array([0.55, 0.52 - tap, -0.10])
            l_fingers = [1.0, 1.0, 1.0, 1.0, 1.0]
            r_fingers = [0.0, 1.0, 1.0, 0.0, 0.0]

        elif class_idx == 4:  # PAIN (Index fingers jab toward each other twice)
            jab = 0.05 * np.sin(progress * 4 * np.pi)
            rw = np.array([0.40 + jab, 0.48, -0.05])
            lw = np.array([0.60 - jab, 0.48, -0.05])
            r_fingers = [0.0, 1.0, 0.0, 0.0, 0.0]
            l_fingers = [0.0, 1.0, 0.0, 0.0, 0.0]

        elif class_idx == 5:  # WATER (W shape taps chin twice)
            tap_z = 0.03 * np.sin(progress * 4 * np.pi)
            rw = np.array([0.50, 0.32, -0.05 + tap_z])
            r_fingers = [0.0, 1.0, 1.0, 1.0, 0.0]

        elif class_idx == 6:  # YES (Fist nods up and down twice)
            nod = 0.05 * np.sin(progress * 4 * np.pi)
            rw = np.array([0.38, 0.48 + nod, -0.05])
            r_fingers = [0.0, 0.0, 0.0, 0.0, 0.0]

        elif class_idx == 7:  # NO (Beak pinching motion twice)
            pinch = 0.5 + 0.5 * np.cos(progress * 4 * np.pi)
            rw = np.array([0.42, 0.46, -0.05])
            r_fingers = [pinch, pinch, pinch, 0.0, 0.0]

        elif class_idx == 8:  # PLEASE (Circular rub over chest)
            cx, cy = 0.50, 0.48
            rw = np.array([cx + 0.07 * np.cos(progress * 2 * np.pi),
                           cy + 0.07 * np.sin(progress * 2 * np.pi), -0.04])
            r_fingers = [1.0, 1.0, 1.0, 1.0, 1.0]

        elif class_idx == 9:  # EMERGENCY (Rapid lateral shake of E-hand)
            shake = 0.06 * np.sin(progress * 6 * np.pi)
            rw = np.array([0.40 + shake, 0.46, -0.05])
            r_fingers = [0.2, 0.2, 0.2, 0.2, 0.2]

        # Calculate natural elbow positions connecting shoulders to wrists
        el_r = (sh_r + rw) / 2.0 + np.array([-0.06, 0.04, 0.02])
        el_l = (sh_l + lw) / 2.0 + np.array([0.06, 0.04, 0.02])

        # Assign the 11 pose points
        seq[t, 0] = nose
        seq[t, 1] = mouth_l
        seq[t, 2] = mouth_r
        seq[t, 3] = sh_l
        seq[t, 4] = sh_r
        seq[t, 5] = el_l
        seq[t, 6] = el_r
        seq[t, 7] = lw
        seq[t, 8] = rw
        seq[t, 9] = hip_l
        seq[t, 10] = hip_r

        # Assign Left Hand (11..31) and Right Hand (32..52)
        seq[t, 11:32] = build_hand(lw, palm_l, l_fingers)
        seq[t, 32:53] = build_hand(rw, palm_r, r_fingers)

        # Add Gaussian spatial jitter
        seq[t] += np.random.normal(0, noise_lvl, seq[t].shape)

    # Normalize each frame identically to the real inference pipeline
    norm_seq = np.zeros_like(seq)
    for t in range(SEQ_LEN):
        norm_seq[t] = normalize_skeleton_frame(seq[t])

    # Transpose from (T, V, C) to (C, T, V)
    return np.transpose(norm_seq, (2, 0, 1))

def generate_all():
    print(f"[START] Synthesizing {NUM_SAMPLES_PER_CLASS * len(cfg.CLASSES)} realistic ASL kinematic sequences...")
    all_x = []
    all_y = []

    for c_idx, c_name in enumerate(cfg.CLASSES):
        print(f" -> Generating class [{c_idx:02d}]: {c_name} ({NUM_SAMPLES_PER_CLASS} samples)")
        for _ in range(NUM_SAMPLES_PER_CLASS):
            sample = synthesize_sequence(c_idx)
            all_x.append(sample)
            all_y.append(c_idx)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    X = np.stack(all_x, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)

    np.save(cfg.DATA_DIR / "X.npy", X)
    np.save(cfg.DATA_DIR / "y.npy", y)
    print(f"\n[DONE] Dataset saved successfully!")
    print(f"       X shape: {X.shape} (Batch, Channels, Frames, Nodes)")
    print(f"       y shape: {y.shape}")

if __name__ == "__main__":
    generate_all()