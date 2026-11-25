import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.model import AddSubLanguageModel, SparseTop1Experts


def _config(**overrides):
    values = dict(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
    )
    values.update(overrides)
    return AddSubConfig(**values)


def test_loss_free_bias_is_opt_in_and_keeps_default_state_dict_unchanged():
    baseline = AddSubLanguageModel(_config())
    assert not any("loss_free_router_bias" in key for key in baseline.state_dict())

    enabled = AddSubLanguageModel(_config(moe_loss_free_bias=True))
    assert "blocks.0.experts.loss_free_router_bias" in enabled.state_dict()
    assert enabled.parameter_count() == baseline.parameter_count()


def test_loss_free_bias_updates_underloaded_experts_and_clamps():
    experts = SparseTop1Experts(
        _config(
            moe_loss_free_bias=True,
            moe_bias_update_rate=0.25,
            moe_bias_max_abs=0.4,
        )
    )
    experts.train()
    first = experts.update_loss_free_bias(torch.tensor([1.0, 0.0]))
    assert first
    torch.testing.assert_close(
        experts.loss_free_router_bias,
        torch.tensor([-0.25, 0.25]),
    )
    for _ in range(4):
        experts.update_loss_free_bias(torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        experts.loss_free_router_bias,
        torch.tensor([-0.4, 0.4]),
    )


def test_loss_free_bias_rejects_invalid_loads_and_eval_updates():
    experts = SparseTop1Experts(_config(moe_loss_free_bias=True)).train()
    for load in (
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([-1.0, 2.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0]),
    ):
        with pytest.raises(ValueError):
            experts.update_loss_free_bias(load)

    experts.eval()
    with pytest.raises(RuntimeError, match="training mode"):
        experts.update_loss_free_bias(torch.tensor([1.0, 0.0]))


def test_loss_free_router_bias_changes_selection_but_not_probability_logits():
    torch.manual_seed(23)
    experts = SparseTop1Experts(
        _config(moe_loss_free_bias=True, moe_bias_update_rate=0.01)
    ).eval()
    with torch.no_grad():
        experts.router.weight.zero_()
        experts.router.bias.zero_()
        experts.loss_free_router_bias.copy_(torch.tensor([2.0, -2.0]))
    inputs = torch.randn(1, 3, 32)
    # Equal raw router logits imply equal selected-expert probability even
    # though the non-gradient bias deterministically chooses expert zero.
    raw_logits = experts.router(inputs).float()
    torch.testing.assert_close(torch.softmax(raw_logits, dim=-1), torch.full((1, 3, 2), 0.5))
    _, _, _, load = experts(inputs)
    torch.testing.assert_close(load, torch.tensor([1.0, 0.0]))


def test_loss_free_bias_updates_once_through_model_training_hook():
    model = AddSubLanguageModel(
        _config(moe_loss_free_bias=True, moe_bias_update_rate=0.05)
    ).train()
    before = model.blocks[0].experts.loss_free_router_bias.clone()
    changed = model.update_loss_free_router_bias(torch.tensor([[1.0, 0.0]]))
    assert changed
    assert not torch.equal(before, model.blocks[0].experts.loss_free_router_bias)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moe_loss_free_bias": 1},
        {"moe_bias_update_rate": True},
        {"moe_bias_update_rate": 0.0},
        {"moe_bias_update_rate": float("inf")},
        {"moe_bias_max_abs": -1.0},
    ],
)
def test_loss_free_bias_config_rejects_invalid_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _config(**kwargs).validate()


def test_loss_free_bias_config_round_trips():
    config = _config(moe_loss_free_bias=True, moe_bias_update_rate=0.02, moe_bias_max_abs=0.7)
    restored = AddSubConfig.from_dict(config.to_dict())
    assert restored == config
