# Video-Based AI Triage for Emergency Departments in Nepal

BE Major Project, Tribhuvan University IOE Pulchowk (IPCV/AIML). Decision-support system that scores
ED patients as RED / YELLOW / GREEN from CCTV video (TimeSformer + pose stream), with calibrated
probabilities, nurse-confirmed alerts and GradCAM explanations.

> **How to use this file:** it is the team's progress tracker. When you finish something, tick its box
> `[x]`, add one line to the **Work log** at the bottom (date, who, what) and commit.

---

## 1. Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --source synthetic      # data -> train -> calibrate -> evaluate (about 10 s on CPU)
python -m pytest -q                                     # 23 tests
./scripts/serve.sh                                      # API on :8000 (stub model, or TRIAGE_CHECKPOINT=runs/synthetic/best.pt)
```

Outputs land in `runs/synthetic/`: `best.pt` (checkpoint + temperature), `metrics.json`, `history.csv`.
Evaluate a checkpoint on any split: `python scripts/evaluate.py --checkpoint runs/synthetic/best.pt --split test`.

## 2. What each file does

| Path | What it does | Owner |
|---|---|---|
| `CONTRACTS.md` | Frozen interfaces: tensor shapes, manifest, model API, serving API | all |
| `configs/default.yaml` | Every setting. Copy it per experiment; override with `--set a.b=value` | all |
| `triage/constants.py` | Class order, pose size, manifest columns | all |
| `triage/data/synthetic.py` | Makes fake class-dependent clips + manifest so we can build before real data exists | Aayush B |
| `triage/data/manifest.py` | Loads/validates the manifest, counts clips per split and class | Aayush B |
| `triage/data/dataset.py` | PyTorch `Dataset` driven by the manifest (frame sampling, resize, [0,1]) | Aayush B |
| `triage/models/backbones.py` | `tiny` 3D-CNN (CPU tests) and real `timesformer`; frozen / last4 / full modes | Anjil |
| `triage/models/fusion.py` | Video features + pose MLP (51 -> 64) -> 3-class head; `predict()`, `explain()` stub | Anjil |
| `triage/models/losses.py` | Focal loss + pose loss + consistency loss (eq. 3 of the proposal) | Anjil |
| `triage/models/calibration.py` | Temperature scaling fitted on the validation set | Anjil |
| `triage/models/stub.py` | Random-output model with the same interface (Day-1 stub) | Anjil |
| `triage/eval/metrics.py` | Macro AUC, RED sensitivity/specificity, bootstrap 95% CI, ECE, subgroups, tau selection | Aayush P |
| `triage/training.py` | Train loop: oversample RED, best-by-val-AUC checkpoint, calibrate, evaluate | Anjil |
| `triage/serving/app.py` | FastAPI `/score`, nurse `/decision`, WebSocket alerts, audit log | Ayush Jha |
| `scripts/*.py` | `make_synthetic`, `train`, `evaluate`, `run_pipeline`, `serve.sh` | all |
| `tests/` | 23 tests covering data, model, metrics, API, pipeline and the data-swap | all |

## 3. Team split

| Person | Owns |
|---|---|
| **Aayush Bhandari** | Camera capture, MOG2 clip segmentation, preprocessing, de-identification, Dataset, NTU/UCF loaders, mock-ED recording, CVAT annotation, ethics (NHRC/IRC) paperwork |
| **Aayush Pandey** | YOLOv8 person detection + pose extraction, pose encoder pretraining (NTU), GradCAM, evaluation suite and plots |
| **Anjil Timsina** | TimeSformer + fusion model, losses, 3-phase training, calibration, I3D baseline |
| **Ayush Kumar Jha** | Repo/CI/Docker, FastAPI, React dashboard, nurse confirm/dismiss, 5-minute re-scoring, audit log |

---

## 4. Progress tracker

### Phase 0 - Foundation
- [x] Repo skeleton, `.gitignore`, requirements
- [x] `CONTRACTS.md` written
- [x] Proposal inconsistencies resolved (pose dim 51, 17 keypoints, frame count 8)
- [ ] Git repo created on GitHub/GitLab, all 4 members cloned and ran the quick start
- [ ] CI (run `pytest` on every push) - Jha

### Phase 1 - Stubs and scaffolding
- [x] Synthetic data generator + manifest + validation
- [x] `TriageDataset` driven by manifest
- [x] Stub model + real fusion model (tiny backbone) with the same interface
- [x] Config-driven training loop runs end to end
- [x] Metrics functions (AUC, sensitivity, CI, ECE) tested
- [x] FastAPI `/score`, WebSocket, nurse decision, audit log (tested)
- [ ] React dashboard showing fake alerts - Jha

### Phase 2 - Real components on proxy data
- [ ] YOLOv8-pose extraction script writing `(T,17,3)` `.npy` per clip - Aayush P
- [ ] NTU RGB+D / UCF-Crime loaders + label mapping to RED/YELLOW/GREEN - Aayush B
- [ ] MOG2 clip segmentation on real video + face blurring - Aayush B
- [ ] Mock-ED recording session on campus (100-200 clips) + CVAT annotation - Aayush B
- [ ] Switch backbone to `timesformer` and train Phase 1 (frozen) on a GPU - Anjil
- [ ] Pose MLP pretraining on NTU - Aayush P
- [ ] API loads a real checkpoint (works already for tiny; test with TimeSformer) - Jha
- [ ] 5-minute re-scoring scheduler - Jha

### Phase 3 - Explainability, calibration, baselines
- [ ] GradCAM (or attention rollout) replacing `explain()` stub - Aayush P
- [ ] Heatmap viewer in dashboard - Jha
- [ ] I3D baseline trained and compared - Anjil
- [ ] Reliability plots + subgroup report on real splits - Aayush P
- [ ] Cohen's Kappa >= 0.80 on 20% double-annotated subset - Aayush B

### Phase 4 - Hardening
- [ ] Latency < 2 s on RTX 3060 measured - Anjil + Jha
- [ ] Docker + docker-compose deployment - Jha
- [ ] Failure handling (camera drop, empty clip, low pose confidence) - Aayush B + P
- [x] Data-swap dry run on synthetic "hospital" manifest (automated test)

### Phase 5 - Real data (after NHRC/IRC approval)
- [ ] Ethics application submitted (takes 3-6 months, **start now**) - Aayush B
- [ ] Cameras + 10 TB NAS installed at PAHS
- [ ] Hospital manifest built (split by patient) and validated
- [ ] Baseline evaluation on test split **before** any fine-tuning
- [ ] Phase 2 and Phase 3 fine-tuning, tau + temperature re-tuned on real validation set
- [ ] Shadow-mode run (alerts shown, nobody acts on them)
- [ ] Final numbers vs targets (macro AUC > 0.80, RED sens >= 0.85 if >= 300 RED cases)

### Phase 6 - Wrap-up
- [ ] Final benchmark tables, report, SOPs, paper draft, demo video, defence slides

---

## 5. Training phases (from the proposal)

| Phase | Config settings | LR | When |
|---|---|---|---|
| 1 Frozen backbone | `model.backbone=timesformer model.train_mode=frozen` | 1e-3 | Month 2 |
| 2 Domain fine-tune | `model.train_mode=last4 train.init_checkpoint=<phase1 best.pt>` | 1e-5 | Months 3-4 |
| 3 Full fine-tune | `model.train_mode=full train.oversample_red=5 train.init_checkpoint=<phase2 best.pt>` | 1e-6 | Month 5 |

Real-model settings: `data.image_size=224 data.num_frames=8`, a GPU with >= 16 GB VRAM for training.

## 6. Honest status and known caveats

- Verified on this machine (CPU only): all 23 tests pass and the pipeline trains on synthetic data.
  Synthetic results (test AUC about 0.9) only prove the plumbing works; they say nothing about real performance.
- The `timesformer` backbone wrapper was checked with a small randomly initialised TimeSformer (shapes and
  freeze modes), but **not** with the real pretrained weights or on a GPU. That is your first GPU task.
- `explain()` is a stub (uniform heatmap) until GradCAM is implemented.
- With very few RED clips the metrics are noisy (the test split of the synthetic set has only 3 RED).
- Loss design choices to mention in the report: pose loss is cross-entropy (proposal said BCE) and the
  consistency loss is our own definition (MSE between the two heads' probabilities).
- RED is weighted twice if you use both `class_weights` and `oversample_red`; tune them together.

## 7. Work log

| Date | Who | What |
|---|---|---|
| 2026-10-15 | Anjil + Claude | Built repo skeleton, contracts, synthetic data, model, losses, calibration, metrics, training loop, API stub, 23 tests. Pipeline verified on CPU. |
