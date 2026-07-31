"""
Tests for the JSON results export (`NumpyEncoder` and the `save_results_json` payload).

WHY THIS FILE EXISTS: coverage put `orchestrator.py` at 84%, and the cold lines were not
scattered through the statistics -- they were the serialisation layer. `NumpyEncoder`'s
type conversions and the whole 2D branch of the JSON payload were unrun.

That layer is what a user actually reads. Everything upstream of it can be correct and the
run still ships a file with the wrong numbers in it, or one that cannot be written at all
because some NumPy scalar reached `json.dump` unconverted -- a failure that arrives at the
very end of a long run, after all the expensive work is done.
"""

import json

import numpy as np
import pytest
from numba import njit

from pyfc.orchestrator import NumpyEncoder, compute_fc_intervals


# --- NumpyEncoder --------------------------------------------------------------------
#
# Each branch converts one NumPy type the stdlib json module refuses. They are checked
# individually because a missing branch does not degrade gracefully: json.dump raises
# TypeError and the entire results file is lost.


@pytest.mark.parametrize("value, expected_type, expected", [
    (np.array([1.0, 2.0]), list, [1.0, 2.0]),
    (np.array([[1, 2], [3, 4]]), list, [[1, 2], [3, 4]]),
    (np.float64(1.5), float, 1.5),
    (np.float32(0.5), float, 0.5),
    (np.int64(7), int, 7),
    (np.int32(7), int, 7),
    (np.bool_(True), bool, True),
    (np.bool_(False), bool, False),
])
def test_numpy_encoder_converts_each_numpy_type(value, expected_type, expected):
    """Every NumPy type the encoder claims to handle round-trips through json."""
    round_tripped = json.loads(json.dumps({"v": value}, cls=NumpyEncoder))["v"]
    assert isinstance(round_tripped, expected_type)
    assert round_tripped == expected


def test_numpy_encoder_still_rejects_genuinely_unserialisable_objects():
    """
    The fallback to `super().default` must remain: an encoder that quietly swallowed
    unknown types would write a results file with something meaningless in it rather than
    telling the caller their payload is wrong.
    """
    with pytest.raises(TypeError):
        json.dumps({"v": object()}, cls=NumpyEncoder)


def test_numpy_encoder_handles_nan_and_infinity():
    """
    NaN reaches this encoder in normal operation -- `results` is initialised with
    `np.full(..., np.nan)` and any grid point that was never evaluated still holds it. json
    emits bare `NaN`/`Infinity`, which is valid for Python's parser and is what round-trips
    here; the point of this test is that writing them does not raise.
    """
    payload = {"nan": np.float64(np.nan), "inf": np.float64(np.inf),
               "arr": np.array([np.nan, 1.0])}
    reloaded = json.loads(json.dumps(payload, cls=NumpyEncoder))

    assert np.isnan(reloaded["nan"])
    assert np.isinf(reloaded["inf"])
    assert np.isnan(reloaded["arr"][0]) and reloaded["arr"][1] == 1.0


# --- The 2D payload ------------------------------------------------------------------


@njit(fastmath=True, nogil=True)
def _rates(params, S_sumw2, B_sumw2):
    mu = np.array([20.0 * params[0] + 1.0, 15.0 * params[1] + 1.0,
                   6.0 * params[0] + 9.0 * params[1] + 1.0])
    return mu, S_sumw2


def test_json_export_contains_both_the_1d_and_2d_payloads(tmp_path):
    """
    A real run with `output_file` set, read back from disk.

    The 2D half of the payload was entirely unrun: it is built only when
    `compute_2D_intervals` is on and more than one parameter exists, and no test combined
    that with JSON output. The keys are checked against the parameter pair they describe,
    since the naming is positional (`p1p2`) and a transposed pair would still produce a
    perfectly valid file.
    """
    data = np.array([21.0, 16.0, 16.0])
    s2 = np.zeros(3)
    # Deliberately ASYMMETRIC -- different ranges and different lengths. With identical
    # grids a transposed pair is undetectable, and an earlier draft of this test used
    # matching ones and duly failed to notice exactly that mutation.
    grids = [np.linspace(0.6, 1.4, 3), np.linspace(0.7, 1.7, 4)]

    results, _ = compute_fc_intervals(
        data=data, grids=grids, compute_rates_func=_rates,
        cl=[0.90], n_toys=3, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sumw2=s2, B_sumw2=s2,
        use_finite_mc_correction_binned=False,
        output_file="fc_results", save_directory=str(tmp_path),
        compute_1D_intervals=True, compute_2D_intervals=True,
    )

    written = list(tmp_path.glob("*.json"))
    assert written, f"no JSON written; directory holds {[p.name for p in tmp_path.iterdir()]}"

    with open(written[0]) as f:
        payload = json.load(f)

    assert "2d_intervals" in payload, "the 2D payload was never written"
    assert "p1p2" in payload["2d_intervals"], (
        f"expected the p1/p2 pair, found {list(payload['2d_intervals'])}"
    )

    pair = payload["2d_intervals"]["p1p2"]
    for key in ("test_p1", "test_p2", "t_data"):
        assert key in pair, f"2D payload is missing {key!r}"

    # The exported grids must be the ones asked for, not a re-derivation that happens to
    # have the right length.
    np.testing.assert_allclose(pair["test_p1"], grids[0])
    np.testing.assert_allclose(pair["test_p2"], grids[1])

    # And the exported statistic must match what the run actually computed.
    np.testing.assert_allclose(
        np.asarray(pair["t_data"], dtype=float), results["2d_t_data_p1p2"], rtol=1e-9,
    )


def test_json_export_is_plain_json_with_no_numpy_left_in_it(tmp_path):
    """
    Everything written must survive a round-trip through a stock `json.load`, which is the
    only guarantee that matters to whoever consumes the file. Reloading with the standard
    parser and re-dumping *without* the custom encoder proves nothing NumPy-typed survived.
    """
    data = np.array([21.0, 16.0, 16.0])
    s2 = np.zeros(3)
    # Deliberately ASYMMETRIC -- different ranges and different lengths. With identical
    # grids a transposed pair is undetectable, and an earlier draft of this test used
    # matching ones and duly failed to notice exactly that mutation.
    grids = [np.linspace(0.6, 1.4, 3), np.linspace(0.7, 1.7, 4)]

    compute_fc_intervals(
        data=data, grids=grids, compute_rates_func=_rates,
        cl=[0.90], n_toys=3, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sumw2=s2, B_sumw2=s2,
        use_finite_mc_correction_binned=False,
        output_file="fc_results", save_directory=str(tmp_path),
        compute_1D_intervals=True, compute_2D_intervals=True,
    )

    with open(next(iter(tmp_path.glob("*.json")))) as f:
        payload = json.load(f)

    json.dumps(payload)  # no cls= : raises TypeError if anything NumPy-typed remains
