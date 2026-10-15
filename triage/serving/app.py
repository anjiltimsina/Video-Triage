"""FastAPI inference + alert service.   (Contract: CONTRACTS.md)

    GET  /health
    POST /score                      score one clip -> probabilities (+ creates a PENDING alert if p_RED > tau)
    GET  /alerts                     all alerts (newest last)
    POST /alerts/{id}/decision       nurse CONFIRMS or DISMISSES an alert  -> written to the audit log
    WS   /ws/alerts                  live push of new alerts to the dashboard

Safety design (matches the proposal): the system is DECISION SUPPORT ONLY. An alert is just a
"pending" record until a nurse confirms or dismisses it, and every decision is logged.

Run:  uvicorn "triage.serving.app:create_app" --factory --reload
OWNER: Ayush Kumar Jha.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..constants import ID2LABEL, RED_ID
from ..data.dataset import _sample_indices
from ..models.stub import StubTriageModel


class ScoreRequest(BaseModel):
    patient_id: str
    video_path: str                  # .npy, uint8 (T, H, W, 3)
    pose_path: str | None = None     # .npy, float32 (T, 17, 3); zeros are used if omitted


class DecisionRequest(BaseModel):
    decision: str                    # "confirm" | "dismiss"
    nurse_id: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_inputs(req: ScoreRequest, num_frames: int, image_size: int | None):
    import cv2
    video = np.load(req.video_path)
    video = video[_sample_indices(len(video), num_frames)]
    if image_size and video.shape[1] != image_size:
        video = np.stack([cv2.resize(f, (image_size, image_size)) for f in video])
    pose = np.load(req.pose_path) if req.pose_path else np.zeros((num_frames, 17, 3), dtype=np.float32)
    pose = pose[_sample_indices(len(pose), num_frames)]
    v = torch.from_numpy(video).float().div(255.0).permute(0, 3, 1, 2)[None]
    p = torch.from_numpy(pose).float()[None]
    return v, p


def create_app(model=None, checkpoint: str | None = None, tau: float | None = None,
               audit_log: str | None = None, num_frames: int = 8, image_size: int | None = 64) -> FastAPI:
    """`model` can be injected (tests). Otherwise load `checkpoint`, else fall back to the stub."""
    if model is None:
        checkpoint = checkpoint or os.environ.get("TRIAGE_CHECKPOINT")
        if checkpoint:
            from ..training import load_checkpoint
            model, ckpt = load_checkpoint(checkpoint)
            num_frames = ckpt["config"]["data"]["num_frames"]
            image_size = ckpt["config"]["data"]["image_size"]
            tau = tau if tau is not None else ckpt["config"]["serving"]["tau"]
        else:
            model = StubTriageModel()
    tau = 0.5 if tau is None else tau
    audit_path = Path(audit_log or os.environ.get("TRIAGE_AUDIT_LOG", "runs/audit_log.jsonl"))
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Video-Based AI Triage (decision support)")
    alerts: dict[str, dict] = {}
    sockets: set[WebSocket] = set()

    def audit(event: str, **fields) -> None:
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "event": event, **fields}) + "\n")

    async def broadcast(message: dict) -> None:
        for ws in list(sockets):
            try:
                await ws.send_json(message)
            except Exception:
                sockets.discard(ws)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": type(model).__name__, "tau": tau}

    @app.post("/score")
    async def score(req: ScoreRequest):
        try:
            video, pose = await run_in_threadpool(_load_inputs, req, num_frames, image_size)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"file not found: {e.filename}")
        pred = await run_in_threadpool(model.predict, video, pose)
        p = pred["calibrated_probs"][0].tolist()
        result = {
            "patient_id": req.patient_id, "p_red": p[0], "p_yel": p[1], "p_grn": p[2],
            "predicted": ID2LABEL[int(np.argmax(p))], "ts": _now(), "alert_id": None,
        }
        if p[RED_ID] > tau:
            alert_id = uuid.uuid4().hex[:12]
            alerts[alert_id] = {**result, "alert_id": alert_id, "status": "pending",
                                "decision_by": None, "heatmap_url": None}
            result["alert_id"] = alert_id
            audit("alert_created", alert_id=alert_id, patient_id=req.patient_id, p_red=p[0], tau=tau)
            await broadcast(alerts[alert_id])
        return result

    @app.get("/alerts")
    def list_alerts():
        return list(alerts.values())

    @app.post("/alerts/{alert_id}/decision")
    async def decide(alert_id: str, req: DecisionRequest):
        if alert_id not in alerts:
            raise HTTPException(status_code=404, detail="unknown alert")
        if req.decision not in ("confirm", "dismiss"):
            raise HTTPException(status_code=422, detail="decision must be 'confirm' or 'dismiss'")
        alert = alerts[alert_id]
        alert["status"] = "confirmed" if req.decision == "confirm" else "dismissed"
        alert["decision_by"] = req.nurse_id
        audit("nurse_decision", alert_id=alert_id, decision=req.decision, nurse_id=req.nurse_id)
        await broadcast(alert)
        return alert

    @app.websocket("/ws/alerts")
    async def ws_alerts(ws: WebSocket):
        await ws.accept()
        sockets.add(ws)
        try:
            while True:
                await ws.receive_text()      # keep the connection open; clients send nothing
        except WebSocketDisconnect:
            sockets.discard(ws)

    return app
