from __future__ import annotations

import ast
import itertools
import math
import os
import subprocess
from pathlib import Path

import pytest

from qmini_addsub.oxcore import OxCoreError, OxCoreNative, StaticArena


@pytest.fixture(scope="session")
def oxcore(tmp_path_factory):
    configured = os.environ.get("QMINI_OXCORE_LIBRARY")
    if configured:
        library = Path(configured)
        assert library.is_file(), library
    else:
        root = Path(__file__).resolve().parents[1]
        library = tmp_path_factory.mktemp("oxcore-native") / "libqmini_oxcore.so"
        subprocess.run(
            ["bash", str(root / "scripts" / "build_oxcore.sh"), str(library)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    return OxCoreNative(library)


def test_oxcore_loads_as_standalone_c_abi(oxcore):
    assert oxcore.abi_version == 1
    assert oxcore.version == "OxCore 0.1.0"


def test_mirror63_all_250_codes_round_trip_and_exactly_six_escapes(oxcore):
    codes = set()
    for q in itertools.product(range(-2, 3), repeat=3):
        for octave in (0, 1):
            code = oxcore.mirror63_pack(q, octave)
            assert code not in codes
            codes.add(code)
            assert oxcore.mirror63_unpack(code) == (q, octave)

    assert len(codes) == 250
    assert set(range(256)) - codes == {
        0x3F, 0x40, 0x7F, 0xBF, 0xC0, 0xFF
    }
    assert oxcore.mirror63_pack((0, 0, 0), 0) == 0x00
    assert oxcore.mirror63_pack((0, 0, 0), 1) == 0x80
    assert oxcore.mirror63_unpack(0x3F) is None


def test_mirror63_rejects_all_reserved_non_skip_codes(oxcore):
    for code in (0x40, 0x7F, 0xBF, 0xC0, 0xFF):
        with pytest.raises(OxCoreError, match="reserved"):
            oxcore.mirror63_unpack(code)

    with pytest.raises(ValueError):
        oxcore.mirror63_pack((3, 0, 0), 0)
    with pytest.raises(ValueError):
        oxcore.mirror63_pack((0, 0, 0), 2)


def test_mirror63_dot_returns_exact_deferred_half_scale_numerator(oxcore):
    first = oxcore.mirror63_pack((2, -1, 0), octave=1)
    second = oxcore.mirror63_pack((-2, 2, -1), octave=0)
    accumulator = oxcore.mirror63_dot_i8(
        [first, second, 0x3F],
        [3, -4, 9, -128, 127, -3, 100, 100, 100],
    )
    # The represented real dot is accumulator / 2 = 266.5.
    assert accumulator == 533


def test_mirror63_dot_fails_closed_on_reserved_byte(oxcore):
    with pytest.raises(OxCoreError, match="reserved"):
        oxcore.mirror63_dot_i8([0x7F], [1, 2, 3])


def test_mirror63_python_bridge_rejects_implicit_integer_wrapping(oxcore):
    code = oxcore.mirror63_pack((1, 0, -1))
    with pytest.raises(ValueError, match="int8 range"):
        oxcore.mirror63_dot_i8([code], [128, 0, 0])
    with pytest.raises(TypeError, match="integers"):
        oxcore.mirror63_dot_i8([code], [1.5, 0, 0])
    with pytest.raises(ValueError, match="uint8 range"):
        oxcore.mirror63_dot_i8([256], [0, 0, 0])


def test_fwht_is_unnormalized_and_invertible_up_to_length(oxcore):
    values = [3, -1, 4, 2, 0, 7, -5, 9]
    transformed = oxcore.fwht_i64(values)
    restored = oxcore.fwht_i64(transformed)
    assert restored == [len(values) * value for value in values]


def test_fwht_rejects_non_power_of_two_and_preflights_overflow(oxcore):
    with pytest.raises(OxCoreError, match="invalid argument"):
        oxcore.fwht_i64([1, 2, 3])
    with pytest.raises(OxCoreError, match="overflow"):
        oxcore.fwht_i64([2**63 - 1, 0])


def test_sinkhorn4_returns_nonnegative_doubly_stochastic_matrix(oxcore):
    logits = [
        [0.0, 0.2, -0.1, 0.4],
        [0.3, -0.2, 0.1, 0.0],
        [-0.1, 0.4, 0.0, 0.2],
        [0.2, 0.0, 0.3, -0.2],
    ]
    matrix = oxcore.sinkhorn4(logits, iterations=40)
    assert all(math.isfinite(value) and value >= 0.0 for row in matrix for value in row)
    for row in matrix:
        assert sum(row) == pytest.approx(1.0, abs=1e-9)
    for column in range(4):
        assert sum(matrix[row][column] for row in range(4)) == pytest.approx(
            1.0, abs=1e-9
        )


def test_sinkhorn4_rejects_invalid_shape_and_nonfinite_values(oxcore):
    with pytest.raises(ValueError, match="4 by 4"):
        oxcore.sinkhorn4([[0.0]])
    with pytest.raises(OxCoreError, match="numeric"):
        oxcore.sinkhorn4([[float("nan")] * 4 for _ in range(4)])


def test_ngpt_l2_normalization_and_zero_vector(oxcore):
    normalized = oxcore.normalize_f32([3.0, 4.0])
    assert math.sqrt(sum(value * value for value in normalized)) == pytest.approx(
        1.0, abs=1e-6
    )
    assert oxcore.normalize_f32([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_ngpt_normalization_rejects_nonfinite_input(oxcore):
    with pytest.raises(OxCoreError, match="numeric"):
        oxcore.normalize_f32([1.0, float("inf")])


def test_static_arena_alignment_marks_rewind_and_high_water(oxcore):
    arena = StaticArena(oxcore, 512)
    first = arena.alloc(13, 64)
    assert first % 64 == 0
    mark = arena.mark()
    second = arena.alloc(31, 128)
    assert second % 128 == 0
    high_water = arena.high_water
    assert arena.used == high_water
    arena.rewind(mark)
    assert arena.used == mark
    assert arena.high_water == high_water
    arena.reset()
    assert arena.used == 0
    assert arena.high_water == high_water


def test_static_arena_rejects_capacity_and_alignment_errors(oxcore):
    arena = StaticArena(oxcore, 64)
    with pytest.raises(OxCoreError, match="capacity"):
        arena.alloc(1024)
    with pytest.raises(OxCoreError, match="invalid argument"):
        arena.alloc(4, 3)
    with pytest.raises(OxCoreError, match="invalid argument"):
        arena.rewind(arena.capacity + 1)


def test_tensor_view_checks_dtype_rank_strides_and_storage_bounds(oxcore):
    values = [1, 2, 3, 4, 5, 6]
    assert oxcore.validate_tensor_view(values, "i32", (2, 3))
    # A reversed length-3 view starts at element 2 and walks backwards.
    assert oxcore.validate_tensor_view(
        values, "i32", (3,), strides_bytes=(-4,), offset_bytes=8
    )
    assert not oxcore.validate_tensor_view(values, "i32", (7,))
    assert not oxcore.validate_tensor_view(
        values, "i32", (2,), strides_bytes=(2,)
    )
    assert oxcore.validate_tensor_view([], "i32", (0,))


def test_tensor_view_rejects_shape_product_overflow(oxcore):
    with pytest.raises(OxCoreError, match="overflow"):
        oxcore.validate_tensor_view([1], "i64", (2**63, 2))


def test_oxcore_python_package_has_no_ml_framework_imports():
    root = Path(__file__).resolve().parents[1] / "qmini_addsub" / "oxcore"
    forbidden = {"torch", "tensorflow", "transformers", "jax", "triton"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert not (names & forbidden), (path, names & forbidden)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, (path, node.module)
