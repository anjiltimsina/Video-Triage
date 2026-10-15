# CONTRACTS (frozen - change only if ALL four agree)

These are the interfaces between your four parts. As long as everyone respects them, anyone
can swap in a better implementation (or real data) without breaking anyone else.

## 1. Clip tensor
`video`: float32 `(T, 3, H, W)`, values in [0, 1].
- On disk: `uint8 (T0, H, W, 3)` `.npy` (T0 = about 150 for a real 5 s clip at 30 FPS).
- `T` = `data.num_frames` (default **8**, because `timesformer-base-finetuned-k400` expects 8). The Dataset samples T frames uniformly from the T0 available.
- `H = W = data.image_size` (**224** for the real model, 64 for CPU tests).

## 2. Pose tensor
`pose`: float32 `(T, 17, 3)` = (x, y, confidence), x/y normalised to [0, 1], COCO-17 keypoint order (see `triage/constants.py`).
- The model flattens each frame to **51** values. (The proposal says 51 in Stage 4 and 34 in Section 5.3; we use 51 because confidence is included.)
- MediaPipe Pose gives 33 landmarks, not 17. Use **YOLOv8n-pose** (native COCO-17) or map MediaPipe down to 17.

## 3. Labels
Class order everywhere: `RED=0, YELLOW=1, GREEN=2`.
Outcome mapping: ICU / HDU / in-hospital death -> RED, ward admission -> YELLOW, ED discharge -> GREEN.

## 4. Manifest (CSV, one row per clip)
`clip_id, video_path, pose_path, label, split, age, sex, arrival_time, source, annotator`
- `video_path` / `pose_path` are relative to the manifest's folder.
- `label` in {RED, YELLOW, GREEN}; `split` in {train, val, test}; `source` in {synthetic, mock, ntu, ucf, hospital}.
- **Split by patient, never by clip** (all clips of one patient must be in one split) once real data arrives.
- Validate any new manifest with `triage.data.manifest.validate_manifest(path)` (`scripts/run_pipeline.py` does this automatically before training).

## 5. Model API
```python
model(video, pose)         -> {"logits": (B,3), "pose_logits": (B,3)}      # training
model.predict(video, pose) -> {"probs": (B,3), "calibrated_probs": (B,3)}  # inference, order RED/YEL/GRN
model.explain(video, pose) -> heatmaps (B, T, H, W) in [0, 1]              # GradCAM (stub for now)
```
`calibrated_probs` = softmax(logits / T) with T fitted on the validation set.

## 6. Serving API
| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model type, tau |
| `POST /score` `{patient_id, video_path, pose_path?}` | returns `{patient_id, p_red, p_yel, p_grn, predicted, ts, alert_id}` |
| `GET /alerts` | all alerts |
| `POST /alerts/{id}/decision` `{decision: confirm\|dismiss, nurse_id}` | nurse decision, written to the audit log |
| `WS /ws/alerts` | live push of alerts: `{alert_id, patient_id, p_red, p_yel, p_grn, ts, status, heatmap_url}` |

An alert is only created when `p_red > tau`, starts as `pending`, and **nothing clinical happens until a nurse confirms**.

## 7. Config
One YAML per experiment in `configs/`. Switching data = changing `data.manifest` (and optionally `data.sources`). Nothing else.
