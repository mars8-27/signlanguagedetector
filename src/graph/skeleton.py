from typing import List, Tuple

# Selected indices from MediaPipe Pose (33 total landmarks):
# 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
# 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip, 9: mouth_left, 10: mouth_right
POSE_INDICES: List[int] = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]

def get_anatomical_edges() -> List[Tuple[int, int]]:
    """
    Returns undirected bone pairs (u, v) across the 53-node upper-body and hand graph.
    Node indexing:
      0..10  : Upper Body Pose (11 nodes)
      11..31 : Left Hand (21 nodes)
      32..52 : Right Hand (21 nodes)
    """
    edges: List[Tuple[int, int]] = []

    # 1. Upper Body Pose Connections
    pose_bones = [
        (0, 1), (0, 2), (1, 2),           # Nose & mouth triangle
        (3, 4),                           # Shoulder span (left_shoulder=3, right_shoulder=4)
        (3, 5), (5, 7),                   # Left arm (shoulder -> elbow -> wrist)
        (4, 6), (6, 8),                   # Right arm (shoulder -> elbow -> wrist)
        (3, 9), (4, 10), (9, 10),         # Torso box (shoulders to hips)
        (7, 11),                          # Left pose wrist (7) to Left Hand wrist root (11)
        (8, 32)                           # Right pose wrist (8) to Right Hand wrist root (32)
    ]
    edges.extend(pose_bones)

    # 2. Hand Bone Topology (identical relative layout for Left and Right hands)
    hand_finger_chains = [
        [0, 1, 2, 3, 4],       # Thumb
        [0, 5, 6, 7, 8],       # Index finger
        [0, 9, 10, 11, 12],    # Middle finger
        [0, 13, 14, 15, 16],   # Ring finger
        [0, 17, 18, 19, 20]    # Pinky finger
    ]

    for offset in (11, 32):  # 11 = Left Hand start, 32 = Right Hand start
        for chain in hand_finger_chains:
            for i in range(len(chain) - 1):
                edges.append((offset + chain[i], offset + chain[i + 1]))
        # Palm knuckle cross-links for structural rigidity
        for k in (5, 9, 13):
            edges.append((offset + k, offset + k + 4))

    return edges