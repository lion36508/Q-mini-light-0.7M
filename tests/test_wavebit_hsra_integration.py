from __future__ import annotations

import numpy as np
import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.wavebit_hsra import (
    CapacitorCharge, HolographicRotationMemory, groupwise_scale,
    holographic_state_step_int, integer_wavebit_projection, wavebit_quantize,
)


def tiny_config() -> AddSubConfig:
    return AddSubConfig(
        vocab_size=264, d_model=16, d_hidden=24, n_layers=2, n_experts=2,
        kernel_size=3, max_context=32, dropout=0.0, seed=17,
    )


def options() -> dict:
    return {"group_size": 8, "rotation": 1, "gamma": 4, "ewgs_strength": 0.5}


def test_wavebit_has_no_additional_model_parameters_or_state_keys():
    config = tiny_config()
    baseline = AddSubLanguageModel(config, seed=17)
    wavebit = AddSubLanguageModel(config, seed=17, wavebit_options=options())
    assert baseline.parameter_count() == wavebit.parameter_count()
    assert set(baseline.state_dict()) == set(wavebit.state_dict())


def test_wavebit_full_forward_backward_is_finite():
    model = AddSubLanguageModel(tiny_config(), seed=17, wavebit_options=options()).train()
    tokens = torch.randint(0, 264, (2, 9))
    result = model(tokens)
    loss = result["logits"].float().square().mean() + result["aux_loss"] * 0.01
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert model.parameter_count() > 0


def test_wavebit_incremental_cache_matches_full_sequence():
    torch.manual_seed(29)
    model = AddSubLanguageModel(tiny_config(), seed=29, wavebit_options=options()).eval()
    tokens = torch.randint(0, 264, (2, 11))
    with torch.inference_mode():
        expected = model(tokens)["logits"]
        caches = model.init_cache(batch_size=tokens.shape[0])
        pieces = []
        for index in range(tokens.shape[1]):
            logits, caches = model.forward_step(tokens[:, index], caches)
            pieces.append(logits.unsqueeze(1))
        actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_groupwise_scale_is_independent_for_each_block():
    weight = torch.cat([torch.ones(2, 32), torch.full((2, 32), 4.0)], dim=1)
    scale = groupwise_scale(weight, group_size=32)
    assert scale.shape == weight.shape
    assert torch.all(scale[:, 32:] > scale[:, :32])


def test_wavebit_ewgs_preserves_nonzero_gradient_at_zero():
    weight = torch.zeros(3, 8, requires_grad=True)
    quantized, polarity, phase, scale = wavebit_quantize(
        weight, group_size=4, ewgs_strength=0.5
    )
    (quantized.sum() + polarity.sum() + phase.sum()).backward()
    assert torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(weight.grad).item() == weight.numel()
    assert torch.count_nonzero(quantized).item() == 0
    assert torch.all(scale > 0)


def test_holographic_memory_is_parameter_free_and_fixed_width():
    memory = HolographicRotationMemory(width=16, rotation=3, gamma=4)
    inputs = torch.randn(2, 13, 16)
    contexts, state = memory(inputs)
    assert contexts.shape == inputs.shape
    assert state.shape == (2, 16)
    assert len(list(memory.parameters())) == 0
    assert torch.count_nonzero(contexts[:, 0]) == 0


def test_integer_projection_matches_seven_effective_states():
    activations = np.asarray([[7, -5, 3, 2]], dtype=np.int16)
    codes = np.asarray([[0, 1, 2, 4], [4, 3, 1, 0]], dtype=np.uint8)
    context = np.asarray([[-1, 0, 1, 1]], dtype=np.int8)
    actual = integer_wavebit_projection(activations, codes, context)
    polarity = (-1, -1, 0, 1, 1)
    phase = (1, 0, 0, 0, 1)
    expected = np.zeros((1, 2), dtype=np.int64)
    for out_index in range(codes.shape[0]):
        for feature in range(codes.shape[1]):
            p = polarity[int(codes[out_index, feature])]
            g = phase[int(codes[out_index, feature])]
            c = int(context[0, feature])
            x = int(activations[0, feature])
            if p == 0:
                continue
            term = -x if p < 0 else x
            copies = 1 if not g else (1 if c < 0 else (2 if c == 0 else 4))
            for _ in range(copies):
                expected[0, out_index] += term
    np.testing.assert_array_equal(actual, expected)


def test_integer_rotation_uses_shift_decay_and_previous_state_context():
    state = np.asarray([[16, -16, 32, -32]], dtype=np.int32)
    context, updated = holographic_state_step_int(
        state, np.zeros_like(state), rotation=0, gamma=4, threshold=1
    )
    np.testing.assert_array_equal(context, [[1, -1, 1, -1]])
    np.testing.assert_array_equal(updated, [[15, -15, 30, -30]])


def test_capacitor_charge_changes_code_only_after_threshold_and_saturates():
    controller = CapacitorCharge(np.asarray([0, 4, 2], dtype=np.uint8), threshold=10)
    np.testing.assert_array_equal(
        controller.step(np.asarray([-15, 15, 6], dtype=np.int32)),
        np.asarray([0, 4, 2], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        controller.step(np.asarray([0, 0, 5], dtype=np.int32)),
        np.asarray([0, 4, 3], dtype=np.uint8),
    )


def test_invalid_wavebit_configuration_is_rejected():
    with pytest.raises(ValueError):
        AddSubLanguageModel(tiny_config(), wavebit_options={"gamma": 0})
