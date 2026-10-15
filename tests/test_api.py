import json

import numpy as np
import torch
from fastapi.testclient import TestClient

from triage.serving.app import create_app


class FixedModel:
    """Always says p_RED = 0.9, so the alert path is deterministic."""

    def predict(self, video, pose):
        p = torch.tensor([[0.9, 0.07, 0.03]]).repeat(video.shape[0], 1)
        return {"probs": p, "calibrated_probs": p}


def _clip(tmp_path):
    np.save(tmp_path / "v.npy", np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8))
    np.save(tmp_path / "p.npy", np.random.rand(10, 17, 3).astype(np.float32))
    return str(tmp_path / "v.npy"), str(tmp_path / "p.npy")


def test_score_creates_pending_alert_and_audit_confirmation(tmp_path):
    audit = tmp_path / "audit.jsonl"
    app = create_app(model=FixedModel(), tau=0.5, audit_log=str(audit), num_frames=8, image_size=32)
    v, p = _clip(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/score", json={"patient_id": "P1", "video_path": v, "pose_path": p}).json()
        assert r["predicted"] == "RED" and r["alert_id"]
        assert client.get("/alerts").json()[0]["status"] == "pending"        # nothing happens automatically
        d = client.post(f"/alerts/{r['alert_id']}/decision",
                        json={"decision": "confirm", "nurse_id": "nurse7"}).json()
        assert d["status"] == "confirmed" and d["decision_by"] == "nurse7"
    events = [json.loads(line)["event"] for line in audit.read_text().splitlines()]
    assert events == ["alert_created", "nurse_decision"]


def test_no_alert_below_tau_and_bad_inputs(tmp_path):
    app = create_app(model=FixedModel(), tau=0.95, audit_log=str(tmp_path / "a.jsonl"),
                     num_frames=8, image_size=32)
    v, p = _clip(tmp_path)
    with TestClient(app) as client:
        r = client.post("/score", json={"patient_id": "P2", "video_path": v, "pose_path": p}).json()
        assert r["alert_id"] is None and client.get("/alerts").json() == []
        assert client.post("/score", json={"patient_id": "P3", "video_path": "/nope.npy"}).status_code == 404
        assert client.post("/alerts/zzz/decision", json={"decision": "confirm", "nurse_id": "n"}).status_code == 404


def test_websocket_receives_new_alert(tmp_path):
    app = create_app(model=FixedModel(), tau=0.5, audit_log=str(tmp_path / "a.jsonl"),
                     num_frames=8, image_size=32)
    v, p = _clip(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as ws:
            client.post("/score", json={"patient_id": "P9", "video_path": v, "pose_path": p})
            msg = ws.receive_json()
            assert msg["patient_id"] == "P9" and msg["status"] == "pending"


def test_stub_fallback_runs(tmp_path):
    app = create_app(audit_log=str(tmp_path / "a.jsonl"), num_frames=8, image_size=32)
    v, p = _clip(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["model"] == "StubTriageModel"
        r = client.post("/score", json={"patient_id": "S", "video_path": v}).json()
        assert abs(r["p_red"] + r["p_yel"] + r["p_grn"] - 1.0) < 1e-5
