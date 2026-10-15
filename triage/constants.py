"""Frozen constants shared by every module. Changing anything here is a CONTRACT change
(see CONTRACTS.md) and must be agreed by all four team members."""

# Class order matters: index in the model's output vector.
CLASSES = ["RED", "YELLOW", "GREEN"]
LABEL2ID = {name: i for i, name in enumerate(CLASSES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
RED_ID = LABEL2ID["RED"]
NUM_CLASSES = len(CLASSES)

# Pose: COCO 17 keypoints, each (x, y, confidence), coordinates normalised to [0, 1].
NUM_KEYPOINTS = 17
POSE_DIM = NUM_KEYPOINTS * 3  # = 51 per frame

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Manifest schema (CSV). One row per clip.
MANIFEST_COLUMNS = [
    "clip_id", "video_path", "pose_path", "label", "split",
    "age", "sex", "arrival_time", "source", "annotator",
]
SPLITS = ["train", "val", "test"]
SOURCES = ["synthetic", "mock", "ntu", "ucf", "hospital"]
