import torch

from src.training.losses import HeightAwareCombinedLoss


def test_height_aware_log_loss_masks_nodata_before_logarithm():
    prediction = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32, requires_grad=True
    )
    target = torch.tensor(
        [[[[1.0, -5.0], [float("nan"), 8.0]]]], dtype=torch.float32
    )
    valid_mask = torch.tensor([[[[True, False], [False, True]]]])
    semantic = torch.zeros((1, 2, 2), dtype=torch.long)
    criterion = HeightAwareCombinedLoss(
        beta=1.0,
        grad_weight=0.5,
        log_weight=0.5,
        height_weights=(1.0, 1.0, 2.0, 3.0, 4.0),
        building_weight=1.5,
    )

    loss, components = criterion(
        prediction, target, valid_mask, semantic
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0, 1] == 0
    assert prediction.grad[0, 0, 1, 0] == 0
