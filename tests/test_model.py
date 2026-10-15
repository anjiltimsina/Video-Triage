import torch

from triage.models.calibration import fit_temperature
from triage.models.fusion import FusionTriageModel
from triage.models.losses import FocalLoss, TriageLoss
from triage.models.stub import StubTriageModel


def _batch(b=4, t=8, s=32):
    return torch.rand(b, t, 3, s, s), torch.rand(b, t, 17, 3), torch.tensor([0, 1, 2, 0])


def test_forward_shapes_and_predict():
    model = FusionTriageModel("tiny")
    video, pose, _ = _batch()
    out = model(video, pose)
    assert out["logits"].shape == (4, 3) and out["pose_logits"].shape == (4, 3)
    pred = model.predict(video, pose)
    assert torch.allclose(pred["probs"].sum(1), torch.ones(4), atol=1e-5)
    assert model.explain(video, pose).shape == (4, 8, 32, 32)


def test_loss_backward_updates_weights():
    model = FusionTriageModel("tiny")
    video, pose, y = _batch()
    loss, parts = TriageLoss(class_weights=[5.0, 1.5, 1.0])(model(video, pose), y)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"focal", "pose", "consistency"}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_focal_loss_downweights_easy_examples():
    easy = torch.tensor([[10.0, -10.0, -10.0]])
    hard = torch.tensor([[-1.0, 1.0, 0.0]])
    y = torch.tensor([0])
    f = FocalLoss(gamma=2.0)
    assert f(easy, y) < f(hard, y)


def test_frozen_mode_freezes_backbone():
    model = FusionTriageModel("tiny")
    model.set_train_mode("frozen")
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_temperature_scaling_reduces_overconfidence():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (500,))
    logits = torch.randn(500, 3) * 6          # wildly overconfident, uninformative logits
    assert fit_temperature(logits, labels) > 1.5


def test_stub_has_same_interface():
    stub = StubTriageModel()
    video, pose, _ = _batch()
    pred = stub.predict(video, pose)
    assert pred["calibrated_probs"].shape == (4, 3) and stub.explain(video, pose).shape == (4, 8, 32, 32)
