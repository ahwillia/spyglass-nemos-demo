"""
Spyglass -> pynapple -> NeMoS: an integrated neural-analysis pipeline.

This script demonstrates the canonical way to feed data managed by Spyglass
(https://github.com/LorenFrankLab/spyglass) into NeMoS
(https://github.com/flatironinstitute/nemos) to fit a Generalized Linear Model
(GLM) of neural firing.

THE KEY IDEA
------------
The two packages share two interchange layers:

    Spyglass  --(NWB on disk)-->  pynapple  --(TsGroup/Tsd)-->  NeMoS

  * Spyglass stores every analysis result as an NWB file and exposes
    `(Table & key).fetch_pynapple()`, which is literally:

        return [pynapple.load_file(path) for path in analysis_nwb_files]

    i.e. it hands you pynapple objects directly.
  * NeMoS consumes pynapple objects natively: a `TsGroup` of spike times, a
    `Tsd`/`TsdFrame` of a behavioral covariate, and an `IntervalSet` of valid
    epochs are exactly its expected inputs.

So the "integration" is just: pull pynapple objects out of Spyglass, bin them,
build features with a NeMoS basis, and fit a GLM. The seam between the two
ecosystems is a single function that returns `(units, feature, epoch)`.

WHAT THIS FILE DOES
-------------------
It defines that seam twice, returning *identical pynapple types* either way:

  * `load_session_from_spyglass(...)` -- the real call pattern against a
    configured Spyglass/DataJoint database. Guarded by a try/import so the file
    still runs without Spyglass installed.
  * `make_synthetic_session(...)`     -- a self-contained simulator producing
    head-direction-tuned neurons, so the downstream NeMoS pipeline is fully
    runnable on any machine with just `pynapple` + `nemos`.

Everything downstream of that seam (`fit_glm`, tuning-curve recovery) is
byte-for-byte identical regardless of where the data came from -- which is the
whole point.

Run:
    python integrated_pipeline.py            # synthetic data (always works)
    python integrated_pipeline.py --spyglass # pull from a live Spyglass DB
    python integrated_pipeline.py --plot     # also save tuning_curves.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pynapple as nap
import nemos as nmo


# --------------------------------------------------------------------------- #
# The interchange object: what crosses the Spyglass -> NeMoS boundary.
# These are all native pynapple types, so both ecosystems understand them.
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    units: nap.TsGroup        # spike times, one entry per neuron
    feature: nap.Tsd          # behavioral covariate (here: head direction, rad)
    epoch: nap.IntervalSet    # valid time interval(s) for the analysis
    feature_name: str = "head_direction"


def basis_features(basis, x, time_support=None):
    """Evaluate a NeMoS basis on a pynapple time series, version-robustly.

    Why this wrapper exists: NeMoS's basis `compute_features` validates its
    input by calling `.astype(float)` on it *before* converting pynapple ->
    array. Current pynapple time-series objects no longer subclass numpy and
    have no `.astype`, so passing a `Tsd` straight in raises AttributeError on
    NeMoS >= ~0.2.5 (older NeMoS accepted the `Tsd` directly).

    The portable fix: hand NeMoS a plain numpy array, then re-wrap the result
    as a pynapple `TsdFrame` so the time axis stays attached for the GLM. This
    works on every NeMoS/pynapple combination, old or new.
    """
    feats = np.asarray(basis.compute_features(np.asarray(x)))
    if time_support is not None and hasattr(x, "t"):
        return nap.TsdFrame(t=x.t, d=feats, time_support=time_support)
    return feats


def empirical_tuning(units, feature, ep, n_bins, lo=0.0, hi=2 * np.pi):
    """Empirical tuning curves as a (n_bins, n_neurons) array, version-robustly.

    pynapple renamed `compute_1d_tuning_curves(group=, nb_bins=, minmax=)` to
    `compute_tuning_curves(data, features, bins=, range=)`, and the new one
    returns an xarray of shape (n_neurons, n_bins) instead of (n_bins,
    n_neurons). This helper handles either and always returns (n_bins, n_neurons).
    """
    try:  # newer pynapple
        tc = nap.compute_tuning_curves(units, feature, bins=n_bins, range=(lo, hi), epochs=ep)
    except (AttributeError, TypeError):  # older pynapple
        tc = nap.compute_1d_tuning_curves(
            group=units, feature=feature, nb_bins=n_bins, ep=ep, minmax=(lo, hi)
        )
    arr = np.asarray(tc)
    if arr.shape[0] != n_bins and arr.shape[-1] == n_bins:
        arr = arr.T  # normalize to (n_bins, n_neurons)
    return arr


# --------------------------------------------------------------------------- #
# SEAM A -- real Spyglass. This is the only Spyglass-specific code.
# --------------------------------------------------------------------------- #
def load_session_from_spyglass(
    nwb_copy_file_name: str,
    interval_list_name: str,
    sorter_key: dict | None = None,
) -> Session:
    """Pull a session out of a live Spyglass/DataJoint database as pynapple objects.

    This mirrors the real Spyglass API. It is not executed in the synthetic demo;
    it documents exactly how the bridge looks against actual infrastructure.

    The pattern is always the same three steps:
      1. Restrict a Spyglass table with a `key` dict.
      2. Call `.fetch_pynapple()` (or `.fetch_nwb()`), which returns pynapple
         NWBFile objects -- one per analysis NWB file in the restriction.
      3. Index those objects to get a TsGroup / Tsd / IntervalSet.
    """
    # Imported lazily so the file runs without Spyglass installed.
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.position import PositionOutput

    # --- spikes: SpikeSortingOutput merge table -> pynapple TsGroup ----------
    spikes_key = {"nwb_file_name": nwb_copy_file_name, **(sorter_key or {})}
    # fetch_pynapple() returns a list of pynapple NWBFile objects.
    nwb = (SortedSpikesGroup & spikes_key).fetch_pynapple()[0]
    units: nap.TsGroup = nwb["units"]  # spike times as a TsGroup

    # --- behavior: PositionOutput -> head direction as a pynapple Tsd --------
    pos_nwb = (PositionOutput & {"nwb_file_name": nwb_copy_file_name}).fetch_pynapple()[0]
    # Spyglass position NWB exposes orientation/head-direction columns.
    head_dir: nap.Tsd = pos_nwb["head_orientation"]

    # --- valid epoch: from the Spyglass IntervalList -------------------------
    from spyglass.common import IntervalList

    valid_times = (
        IntervalList
        & {"nwb_file_name": nwb_copy_file_name, "interval_list_name": interval_list_name}
    ).fetch1("valid_times")
    epoch = nap.IntervalSet(start=valid_times[:, 0], end=valid_times[:, 1])

    # Restrict everything to the analysis epoch -- standard pynapple idiom.
    units = units.restrict(epoch)
    head_dir = head_dir.restrict(epoch)
    return Session(units=units, feature=head_dir, epoch=epoch)


# --------------------------------------------------------------------------- #
# SEAM B -- synthetic stand-in. Returns the SAME pynapple types as Seam A,
# so nothing downstream can tell the difference.
# --------------------------------------------------------------------------- #
def make_synthetic_session(
    n_neurons: int = 8,
    duration_s: float = 1200.0,
    dt: float = 0.01,
    seed: int = 1,
) -> Session:
    """Simulate head-direction cells and emit pynapple objects.

    Each neuron has a von-Mises tuning curve over head direction; spikes are
    drawn as an inhomogeneous Poisson process. This is the same data shape
    Spyglass would hand back, so it exercises the real integration code.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt)
    epoch = nap.IntervalSet(start=0.0, end=duration_s)

    # A smooth, drifting head-direction trajectory wrapped to [0, 2*pi).
    angle = np.cumsum(rng.normal(0, 0.3, size=t.size)) * dt * 6.0
    angle = np.mod(angle, 2 * np.pi)
    feature = nap.Tsd(t=t, d=angle, time_support=epoch)

    # Each neuron: preferred direction + von-Mises tuning -> Poisson spikes.
    pref_dirs = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    kappa = 4.0          # tuning sharpness
    peak_hz = 25.0       # peak firing rate
    base_hz = 1.0        # baseline

    spike_dict = {}
    for i, mu in enumerate(pref_dirs):
        rate_hz = base_hz + peak_hz * np.exp(kappa * (np.cos(angle - mu) - 1.0))
        n_spk = rng.poisson(rate_hz * dt)            # spikes per bin
        # Place spikes uniformly within each bin they occur in.
        idx = np.repeat(np.arange(t.size), n_spk)
        spk_t = t[idx] + rng.uniform(0, dt, size=idx.size)
        spike_dict[i] = nap.Ts(np.sort(spk_t))

    units = nap.TsGroup(spike_dict, time_support=epoch)
    units.set_info(pref_dir=pref_dirs)  # carry ground truth for comparison
    return Session(units=units, feature=feature, epoch=epoch)


# --------------------------------------------------------------------------- #
# DOWNSTREAM -- pure NeMoS. Identical no matter which seam produced `session`.
# --------------------------------------------------------------------------- #
def fit_glm(session: Session, bin_size: float = 0.01):
    """Fit a population Poisson GLM predicting spike counts from head direction.

    Steps, all on pynapple objects so epochs/gaps are respected automatically:
      1. Bin spikes:            TsGroup.count    -> TsdFrame (time x neurons)
      2. Align the covariate:   Tsd.bin_average  -> Tsd      (time,)
      3. Build features:        a cyclic B-spline basis over the angle
      4. Fit:                   PopulationGLM (Poisson observation model)
    """
    ep = session.epoch

    # 1. Spike counts per bin, per neuron.
    counts = session.units.count(bin_size, ep)               # TsdFrame (T, N)

    # 2. Down-sample the covariate onto the same bins.
    feat = session.feature.bin_average(bin_size, ep)         # Tsd (T,)

    # 3. Feature design. Head direction is circular -> cyclic B-spline basis.
    #    `basis_features` keeps the result a pynapple TsdFrame (see its docstring
    #    for the NeMoS/pynapple version note).
    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8)
    X = basis_features(basis, feat, time_support=ep)         # TsdFrame (T, 8)

    # 4. One GLM jointly over all neurons (shared design matrix).
    glm = nmo.glm.PopulationGLM(
        observation_model=nmo.observation_models.PoissonObservations(),
        regularizer="Ridge",
        regularizer_strength=1e-4,
    ).fit(X, counts)

    pseudo_r2 = float(glm.score(X, counts, score_type="pseudo-r2-McFadden"))
    return glm, basis, X, counts, feat, pseudo_r2


def compare_tuning(session: Session, glm, basis, bin_size: float = 0.01, n_bins: int = 60):
    """Compare GLM-predicted tuning to the empirical tuning curve per neuron.

    Returns a dict of correlation coefficients; >0.9 means the GLM recovered
    the head-direction tuning the cells actually have.
    """
    ep = session.epoch

    # Empirical tuning: mean rate as a function of head direction.
    emp = empirical_tuning(session.units, session.feature, ep, n_bins)  # (n_bins, n_neurons)

    # Model-predicted tuning: feed a sweep of angles through basis -> GLM.
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    Xsweep = basis_features(basis, angles)                  # ndarray (n_bins, 8)
    pred_rate = np.asarray(glm.predict(Xsweep)) / bin_size  # counts/bin -> Hz

    corrs = {}
    for j in range(emp.shape[1]):
        c = np.corrcoef(emp[:, j], pred_rate[:, j])[0, 1]
        corrs[j] = float(c)
    return corrs, angles, emp, pred_rate


def plot_tuning(angles, emp, pred, corrs, out_path="tuning_curves.png"):
    """Save a grid of per-neuron tuning curves: empirical vs. GLM prediction."""
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, don't open a window
    import matplotlib.pyplot as plt

    n = emp.shape[1]
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    deg = np.rad2deg(angles)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow),
                             sharex=True, subplot_kw={})
    for j, ax in enumerate(np.atleast_1d(axes).ravel()):
        if j >= n:
            ax.axis("off")
            continue
        ax.plot(deg, emp[:, j], color="0.5", lw=2, label="empirical")
        ax.plot(deg, pred[:, j], color="C3", lw=2, ls="--", label="GLM")
        ax.set_title(f"neuron {j}  (r={corrs[j]:+.2f})", fontsize=9)
        ax.set_xticks([0, 180, 360])
    fig.supxlabel("head direction (deg)")
    fig.supylabel("firing rate (Hz)")
    axes.ravel()[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Spyglass -> pynapple -> NeMoS GLM: recovered tuning curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spyglass", action="store_true",
        help="Pull data from a live Spyglass DB instead of the synthetic simulator.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save tuning_curves.png comparing empirical vs. GLM-predicted tuning.",
    )
    args = parser.parse_args()

    if args.spyglass:
        print(">> Loading session from Spyglass (live DataJoint database)...")
        # Fill these in for your database:
        session = load_session_from_spyglass(
            nwb_copy_file_name="my_session_.nwb",
            interval_list_name="pos 0 valid times",
        )
        source = "Spyglass"
    else:
        print(">> No live database; generating a synthetic head-direction session.")
        print("   (Identical pynapple types to Spyglass's fetch_pynapple output.)")
        session = make_synthetic_session()
        source = "synthetic"

    print(f"\nSession source : {source}")
    print(f"  units        : {len(session.units)} neurons (pynapple {type(session.units).__name__})")
    print(f"  feature      : {session.feature_name} (pynapple {type(session.feature).__name__})")
    print(f"  epoch        : {session.epoch.tot_length():.0f} s "
          f"(pynapple {type(session.epoch).__name__})")

    print("\n>> Fitting NeMoS PopulationGLM (Poisson) on head direction...")
    glm, basis, X, counts, feat, pseudo_r2 = fit_glm(session)
    print(f"  design matrix: {X.shape} (CyclicBSplineEval features)")
    print(f"  spike counts : {counts.shape}")
    print(f"  McFadden pseudo-R^2 (population): {pseudo_r2:.3f}")

    print("\n>> Recovering tuning curves (GLM prediction vs. empirical)...")
    corrs, angles, emp, pred = compare_tuning(session, glm, basis)
    for j, c in corrs.items():
        extra = ""
        if "pref_dir" in session.units.metadata_columns:
            extra = f"  [true pref {np.rad2deg(session.units.get_info('pref_dir')[j]):5.0f} deg]"
        print(f"  neuron {j}: empirical-vs-GLM tuning corr = {c:+.3f}{extra}")

    mean_c = float(np.mean(list(corrs.values())))
    print(f"\n  mean tuning correlation across neurons: {mean_c:+.3f}")
    if mean_c > 0.9:
        print("  -> GLM recovered the cells' tuning. Pipeline works end to end.")

    if args.plot:
        print("\n>> Plotting tuning curves...")
        plot_tuning(angles, emp, pred, corrs)


if __name__ == "__main__":
    main()
