import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini.native import NativeKernel
from qmini.quantization import pack_quaternary, unpack_quaternary


@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("clang++")), reason="C++ compiler missing")
def test_native_projection_matches_reference_on_odd_dimensions(tmp_path):
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    library = tmp_path / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )

    weight = torch.tensor([
        [-4., -1., 1., 4., 4., 1., -1.],
        [1., 4., -4., -1., -1., -4., 4.],
        [2., -3., 1., -2., 4., -1., 3.],
    ])
    packed, scales = pack_quaternary(weight)
    dequantized = unpack_quaternary(packed, scales, in_features=7)
    x = np.array([
        [.2, -.4, .7, 1., -.1, .3, .5],
        [-.9, .1, .4, -.2, .7, -.5, .3],
    ], dtype=np.float32)
    x_scale = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.
    x_q = np.rint(x / x_scale[:, None]).clip(-127, 127).astype(np.int8)
    bias = np.array([.25, -.5, .1], dtype=np.float32)

    output = NativeKernel(library).linear(x_q, packed, scales, x_scale, bias)
    expected = x_q.astype(np.float32) @ dequantized.T
    expected = expected * x_scale[:, None] + bias
    assert output.shape == (2, 3)
    assert np.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_native_wrapper_rejects_incompatible_shapes(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    native = NativeKernel(library)
    assert isinstance(native.uses_neon, bool)
    with pytest.raises(ValueError, match="one scale per input row"):
        native.linear(
            np.ones((2, 4), dtype=np.int8),
            np.zeros((3, 1), dtype=np.uint8),
            np.ones(3, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )


@pytest.fixture(scope="module")
def compiled_native_library(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path_factory.mktemp("native-runtime") / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    return library


def test_qlinear_uses_native_kernel_and_invalidates_packed_cache(
    compiled_native_library, monkeypatch
):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QLinear
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    kernel = NativeKernel(compiled_native_library)
    calls = {"count": 0}
    original = kernel.linear

    def counted_linear(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    kernel.linear = counted_linear
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    monkeypatch.setenv("QMINI_NATIVE_FORCE_SCALAR", "1")

    layer = QLinear(7, 5).eval()
    values = torch.randn(2, 3, 7, dtype=torch.float32)
    with torch.inference_mode():
        actual = layer(values)
        expected = F.linear(
            fake_quantize_int8(values),
            quantize_quaternary(layer.weight),
            layer.bias,
        )
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert calls["count"] == 1
        first_key = layer._qmini_native_packed_key

        # A changed master weight must regenerate the packed row data.
        layer.weight.add_(0.05)
        actual_after_update = layer(values)
        expected_after_update = F.linear(
            fake_quantize_int8(values),
            quantize_quaternary(layer.weight),
            layer.bias,
        )
        assert torch.allclose(
            actual_after_update, expected_after_update, atol=2e-5, rtol=2e-5
        )
        assert calls["count"] == 2
        assert layer._qmini_native_packed_key != first_key


def test_tied_embedding_output_uses_native_projection(
    compiled_native_library, monkeypatch
):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QEmbedding
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    kernel = NativeKernel(compiled_native_library)
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    monkeypatch.setenv("QMINI_NATIVE_FORCE_SCALAR", "1")
    embedding = QEmbedding(37, 9).eval()
    hidden = torch.randn(2, 4, 9)
    with torch.inference_mode():
        actual = embedding.project(hidden)
        expected = F.linear(
            fake_quantize_int8(hidden),
            quantize_quaternary(embedding.weight),
        )
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert hasattr(embedding, "_qmini_native_packed_weight")


def test_pytorch_fallback_is_used_without_native_library(monkeypatch):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QLinear
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    monkeypatch.setattr(native_module, "get_native_kernel", lambda: None)
    layer = QLinear(4, 3).eval()
    values = torch.randn(2, 4)
    with torch.inference_mode():
        actual = layer(values)
        expected = F.linear(
            fake_quantize_int8(values), quantize_quaternary(layer.weight), layer.bias
        )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not hasattr(layer, "_qmini_native_packed_weight")


def test_native_loader_retries_after_library_is_built(
    tmp_path, monkeypatch
):
    import qmini.native as native_module

    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path / "late-built.so"

    monkeypatch.setenv("QMINI_NATIVE_LIBRARY", str(library))
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setattr(native_module, "_NATIVE_CACHE_KEY", None)
    assert native_module.get_native_kernel() is None

    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    assert native_module.get_native_kernel() is not None
    assert native_module.native_kernel_status()["available"] is True


def test_native_wrapper_rejects_silent_dtype_narrowing(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    packed = np.zeros((2, 1), dtype=np.uint8)
    scales = np.ones(2, dtype=np.float32)
    activation_scales = np.ones(1, dtype=np.float32)

    with pytest.raises(TypeError, match="already be int8"):
        native.linear(
            np.ones((1, 4), dtype=np.int64),
            packed,
            scales,
            activation_scales,
        )
    with pytest.raises(TypeError, match="already be uint8"):
        native.linear(
            np.ones((1, 4), dtype=np.int8),
            np.zeros((2, 1), dtype=np.int16),
            scales,
            activation_scales,
        )


def test_outdated_native_library_fails_with_rebuild_instruction(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    old_source = tmp_path / "old_native.cpp"
    old_library = tmp_path / "old_native.so"
    old_source.write_text(
        'extern "C" int qmini_linear_i8() { return 0; }',
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(old_source), "-o", str(old_library)],
        check=True,
    )
    with pytest.raises(RuntimeError, match="Native library is outdated"):
        NativeKernel(old_library)


def test_scalar_native_kernel_is_opt_in_on_non_neon_platform(
    compiled_native_library, monkeypatch
):
    import qmini.native as native_module
    from qmini.model import QLinear

    kernel = NativeKernel(compiled_native_library)
    if kernel.uses_neon:
        pytest.skip("NEON is a default-supported native backend.")
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    monkeypatch.delenv("QMINI_NATIVE_FORCE_SCALAR", raising=False)

    layer = QLinear(7, 5).eval()
    values = torch.randn(2, 3, 7, dtype=torch.float32)
    with torch.inference_mode():
        output = layer(values)
    assert torch.isfinite(output).all()
    assert not hasattr(layer, "_qmini_native_packed_weight")
    assert native_module.native_kernel_should_be_used(kernel) is False

    monkeypatch.setenv("QMINI_NATIVE_FORCE_SCALAR", "1")
    with torch.inference_mode():
        output_forced = layer(values)
    assert torch.allclose(output_forced, output, atol=2e-5, rtol=2e-5)
    assert hasattr(layer, "_qmini_native_packed_weight")
    assert native_module.native_kernel_should_be_used(kernel) is True


def test_neon_native_kernel_is_selected_without_force_flag(
    compiled_native_library, monkeypatch
):
    import qmini.native as native_module

    kernel = NativeKernel(compiled_native_library)
    if not kernel.uses_neon:
        pytest.skip("This runner has no NEON backend.")
    monkeypatch.delenv("QMINI_NATIVE_FORCE_SCALAR", raising=False)
    assert native_module.native_kernel_should_be_used(kernel) is True



def test_native_q2_v2_zero_scale_row_matches_python_reference(compiled_native_library):
    from qmini.quantization import pack_quaternary, unpack_quaternary

    native = NativeKernel(compiled_native_library)
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, -0.5, 0.0, 0.5, 2.0],
    ])
    packed, scales = pack_quaternary(weight)
    assert scales[0] == 0.0
    assert scales[1] > 0.0
    x = np.array([[0.25, -0.5, 1.0, 0.75, -1.0]], dtype=np.float32)
    x_scale = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.0
    x_q = np.rint(x / x_scale[:, None]).clip(-127, 127).astype(np.int8)

    actual = native.linear(
        x_q, packed, scales, x_scale.astype(np.float32), q2_version=2
    )
    dequantized = unpack_quaternary(packed, scales, 5, format_version=2)
    expected = (x_q.astype(np.float32) @ dequantized.T) * x_scale[:, None]
    assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert actual[0, 0] == 0.0


def test_native_q2_v1_legacy_projection_uses_legacy_codebook(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    # Four legacy symbols [-2,-1,+1,+2] with a single positive v1 scale.
    packed = np.array([[0b11100100]], dtype=np.uint8)
    scales = np.array([0.25], dtype=np.float32)
    x_q = np.array([[1, 2, 3, 4]], dtype=np.int8)
    x_scale = np.array([0.1], dtype=np.float32)
    actual = native.linear(
        x_q, packed, scales, x_scale, q2_version=1
    )
    legacy_weight = np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float32) * 0.25
    expected = (x_q.astype(np.float32) @ legacy_weight.T) * 0.1
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_native_q2_v2_rejects_negative_scale_but_accepts_zero(compiled_native_library):
    native = NativeKernel(compiled_native_library)
    x_q = np.ones((1, 4), dtype=np.int8)
    packed = np.zeros((2, 1), dtype=np.uint8)
    x_scale = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        native.linear(
            x_q, packed, np.array([0.0, -1.0], dtype=np.float32),
            x_scale, q2_version=2,
        )
    out = native.linear(
        x_q, packed, np.array([0.0, 0.0], dtype=np.float32),
        x_scale, q2_version=2,
    )
    assert np.array_equal(out, np.zeros((1, 2), dtype=np.float32))
