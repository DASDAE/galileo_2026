# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "galileo-2026 @ git+https://github.com/DASDAE/galileo_2026@main",
#     "dascore @ git+https://github.com/DASDAE/dascore@dev",
#     "marimo>=0.24",
#     "matplotlib>=3.10",
#     "numba",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quantifying the quiet-band registration and optical-loss trend

    The 5 Hz DAS record contains two very different channel populations: fiber inside the inventory's coupled borehole intervals and fiber carried between them. The same optical path also crosses several components with measured discrete-event loss.

    **Questions:** how large and consistent is the low-variability contrast used to register the DAS path's borehole intervals, and is residual variability on the intervening cable monotonically associated with accumulated listed connector and splice loss?

    The first question is descriptive, not an independent coupling test. The inventory documentation says these DAS intervals were positioned from quiet bands and refined against low per-channel variance in a quiet ten-minute record. This notebook quantifies that defining contrast in the shipped data; it cannot use the same phenomenon to prove its mechanical cause.
    """)
    return


@app.cell
def _():
    from itertools import pairwise

    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from galileo_2026 import get_data_path, get_inventory_path

    plt.rcParams["figure.figsize"] = (11, 5)
    return dc, get_data_path, get_inventory_path, np, pairwise, pd, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A robust quiet-time metric

    We use the first three minutes in the archive, ending more than 30 seconds before the first isolated charge. Rather than force the one-minute files across their timing boundaries, we difference each file separately and then join the differences. A first difference rejects slow phase drift; its median absolute deviation, divided by the normal-consistency constant, measures robust scatter in the 0.2-second increments. It is not converted to a per-sample standard deviation, which would require an assumption about temporal independence. The result includes real fiber motion as well as measurement noise, so it is best read as **short-period channel variability**, not as a calibrated interrogator noise floor.
    """)
    return


@app.cell
def _(dc, get_data_path, get_inventory_path, np):
    inventory = dc.inventory(get_inventory_path())
    _quiet_spool = (
        dc.spool(get_data_path())
        .update()
        .attach_inventory(inventory)
        .select(
            tag="DAS_LF",
            time=("2026-08-06T10:09:54", "2026-08-06T10:12:54"),
        )
        .enrich()
    )
    quiet_parts = list(_quiet_spool)
    assert len(quiet_parts) == 3
    assert all(part.dims == ("time", "distance") for part in quiet_parts)
    _distance = quiet_parts[0].get_array("distance")
    assert all(
        np.array_equal(part.get_array("distance"), _distance)
        for part in quiet_parts[1:]
    )
    _quiet_spool.get_contents()[["time_min", "time_max", "distance_max"]]
    return inventory, quiet_parts


@app.cell
def _(np, pd, quiet_parts):
    # Convert every part before differencing so the metric has strain units.
    _strain_parts = [part.radians_to_strain() for part in quiet_parts]
    _differences = np.concatenate(
        [np.diff(part.data, axis=0) for part in _strain_parts], axis=0
    )
    _centered = _differences - np.median(_differences, axis=0)
    # The normal-consistency constant makes this a robust standard deviation
    # of the increments without assuming adjacent samples are independent.
    difference_scatter = (
        np.median(np.abs(_centered), axis=0) / 0.6744897501960817 * 1e9
    )

    _reference = quiet_parts[0]
    instrument_distance = np.asarray(_reference.get_array("distance"))
    _borehole = np.asarray(_reference.get_array("borehole")).astype(str)
    coupling = np.asarray(_reference.get_array("coupling")).astype(str)
    _hole_depth = np.asarray(_reference.get_array("hole_depth"), dtype=float)
    _coupled = coupling == "outside_borehole_casing"
    assert np.array_equal(_coupled, _borehole != "")

    _rows = []
    for _name in sorted(set(_borehole) - {""}):
        _inside = _coupled & (_borehole == _name)
        _start = instrument_distance[_inside].min()
        _stop = instrument_distance[_inside].max()
        _flank = (coupling == "") & (
            (
                (instrument_distance >= _start - 15)
                & (instrument_distance < _start)
            )
            | (
                (instrument_distance > _stop)
                & (instrument_distance <= _stop + 15)
            )
        )
        assert _flank.any()
        _inside_noise = np.median(difference_scatter[_inside])
        _flank_noise = np.median(difference_scatter[_flank])
        _rows.append(
            {
                "borehole": _name,
                "cable": (
                    "vertical"
                    if _name in {"S100", "S120", "S140", "S160", "S180"}
                    else "inclined"
                ),
                "distance_min": _start,
                "distance_max": _stop,
                "inside_nstrain": _inside_noise,
                "flank_nstrain": _flank_noise,
                "inside/flank": _inside_noise / _flank_noise,
            }
        )
    hole_summary = pd.DataFrame(_rows)

    _depth_rows = []
    for _low, _high in ((0, 5), (5, 15), (15, 25), (25, 32)):
        _mask = _coupled & (_hole_depth >= _low) & (_hole_depth < _high)
        _depth_rows.append(
            {
                "depth_bin_m": f"{_low}-{_high}",
                "channels": int(_mask.sum()),
                "median_nstrain": np.median(difference_scatter[_mask]),
            }
        )
    depth_summary = pd.DataFrame(_depth_rows)
    hole_summary.round(4)
    return (
        coupling,
        depth_summary,
        difference_scatter,
        hole_summary,
        instrument_distance,
    )


@app.cell
def _(difference_scatter, hole_summary, instrument_distance, np, plt):
    _fig, _axes = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": (2, 1)}
    )
    _axes[0].semilogy(
        instrument_distance,
        difference_scatter,
        color="0.35",
        lw=0.8,
        label="all channels",
    )
    for _row in hole_summary.itertuples():
        _axes[0].axvspan(
            _row.distance_min,
            _row.distance_max,
            color="C0" if _row.cable == "vertical" else "C2",
            alpha=0.18,
        )
        _axes[0].text(
            (_row.distance_min + _row.distance_max) / 2,
            0.00045,
            _row.borehole,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _axes[0].set(
        xlim=(0, 1660),
        ylim=(3e-4, 300),
        xlabel="DAS instrument distance [m]",
        ylabel="robust 0.2 s difference scatter [nanostrain]",
        title="Blue: vertical-hole cable; green: inclined-hole cable",
    )

    _colors = np.where(hole_summary["cable"] == "vertical", "C0", "C2")
    _axes[1].bar(
        hole_summary["borehole"],
        hole_summary["inside/flank"],
        color=_colors,
    )
    _axes[1].axhline(1, color="0.2", ls="--", lw=1)
    _axes[1].set_yscale("log")
    _axes[1].set(
        ylabel="inside / adjacent cable",
        title="Every inventory interval is quieter than its local flanks",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each blue or green band is selected by the inventory coupling code `outside_borehole_casing` and coincides with a deep trough. That alignment is expected: the DAS path positions were picked from quiet bands and then checked by minimizing mean per-channel variance in a quiet record. The installation metadata states that the borehole intervals are cemented, which supplies a plausible mechanical interpretation, but this analysis is not independent evidence for that mechanism. Its contribution is to quantify how strong and consistent the registered contrast is with a local comparison; using one global background would additionally confuse position with installation.

    ## Does accumulated splice loss explain the rest?

    The inventory places every measured connector and splice loss on optical-path distance. The acquisition's `distance_map` converts those positions back to the DAS instrument axis. We summarize only uncoupled cable, exclude five metres on either side of every event, stop at the inventoried fiber end, and compare each eligible inter-event segment with cumulative listed loss. This is a deliberately narrow test: because cumulative loss increases with distance, it tests only whether the segment medians have a monotonic trend along the listed discrete-event loss sequence. It does not estimate distributed attenuation or isolate the local effect of an individual event.
    """)
    return


@app.cell
def _(
    coupling,
    difference_scatter,
    instrument_distance,
    inventory,
    np,
    pairwise,
    pd,
    stats,
):
    _fiber_array = inventory.networks[0].fiber_arrays[0]
    _path = next(
        item
        for item in _fiber_array.optical_paths
        if item.location_code == "04"
    )
    _acquisition = next(
        item
        for item in _fiber_array.acquisitions
        if item.location_code == "04" and item.code == "MSF"
    )
    _distance_map = _acquisition.distance_map
    _instrument_control = np.asarray(_distance_map.instrument_distance)
    _path_control = np.asarray(_distance_map.distance)

    _loss_components = sorted(
        (
            item
            for item in _path.optical_components
            if getattr(item, "loss_db", None) is not None
            and _path_control.min() <= item.distance_min <= _path_control.max()
        ),
        key=lambda item: item.distance_min,
    )
    loss_positions = np.interp(
        [item.distance_min for item in _loss_components],
        _path_control,
        _instrument_control,
    )
    _loss_values = np.asarray([item.loss_db for item in _loss_components])
    _terminator = next(
        item
        for item in _path.optical_components
        if item.object_type == "Terminator"
    )
    fiber_end = float(
        np.interp(
            _terminator.distance_min,
            _path_control,
            _instrument_control,
        )
    )

    _interior_losses = [
        position
        for position in loss_positions
        if _instrument_control[0] < position < fiber_end
    ]
    _boundaries = np.asarray(
        [_instrument_control[0], *_interior_losses, fiber_end]
    )
    _segment_rows = []
    for _start, _stop in pairwise(_boundaries):
        _mask = (
            (instrument_distance >= _start + 5)
            & (instrument_distance <= _stop - 5)
            & (coupling == "")
        )
        if _mask.sum() < 5:
            continue
        _cumulative = _loss_values[loss_positions <= _start + 1e-9].sum()
        _segment_rows.append(
            {
                "distance_mid": (_start + _stop) / 2,
                "channels": int(_mask.sum()),
                "cumulative_loss_db": _cumulative,
                "median_nstrain": np.median(difference_scatter[_mask]),
            }
        )
    loss_summary = pd.DataFrame(_segment_rows)
    loss_rho, loss_pvalue = stats.spearmanr(
        loss_summary["cumulative_loss_db"],
        loss_summary["median_nstrain"],
    )
    loss_segment_count = len(loss_summary)
    loss_summary.round(3)
    return (
        fiber_end,
        loss_positions,
        loss_pvalue,
        loss_rho,
        loss_segment_count,
        loss_summary,
    )


@app.cell
def _(
    difference_scatter,
    fiber_end,
    instrument_distance,
    loss_positions,
    loss_rho,
    loss_segment_count,
    loss_summary,
    plt,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _valid = instrument_distance <= fiber_end
    _axes[0].semilogy(
        instrument_distance[_valid],
        difference_scatter[_valid],
        color="0.35",
        lw=0.8,
    )
    for _position in loss_positions:
        _axes[0].axvline(_position, color="C3", alpha=0.55, lw=1)
    _axes[0].set(
        xlabel="DAS instrument distance [m]",
        ylabel="robust 0.2 s difference scatter [nanostrain]",
        title="Red: connector or splice with measured loss",
    )

    _axes[1].scatter(
        loss_summary["cumulative_loss_db"],
        loss_summary["median_nstrain"],
        s=50,
    )
    for _row in loss_summary.itertuples():
        _axes[1].annotate(
            f"{_row.distance_mid:.0f} m",
            (_row.cumulative_loss_db, _row.median_nstrain),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    _axes[1].set_yscale("log")
    _axes[1].set(
        xlabel="cumulative listed loss [dB]",
        ylabel="median uncoupled-cable difference scatter [nanostrain]",
        title=f"{loss_segment_count} segments: Spearman rho = {loss_rho:.2f}",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    depth_summary,
    hole_summary,
    loss_pvalue,
    loss_rho,
    loss_segment_count,
    mo,
):
    _median_ratio = hole_summary["inside/flank"].median()
    _minimum = hole_summary["inside/flank"].min()
    _maximum = hole_summary["inside/flank"].max()
    _vertical = hole_summary.loc[
        hole_summary["cable"] == "vertical", "inside_nstrain"
    ].median()
    _inclined = hole_summary.loc[
        hole_summary["cable"] == "inclined", "inside_nstrain"
    ].median()
    _depth_spread = (
        depth_summary["median_nstrain"].max()
        / depth_summary["median_nstrain"].min()
    )
    _hole_count = len(hole_summary)
    _median_quieter = 1 / _median_ratio
    _minimum_quieter = 1 / _maximum
    _maximum_quieter = 1 / _minimum
    mo.md(rf"""
    ## Results

    Across all {_hole_count} holes, the median coupled-to-flank ratio is **{_median_ratio:.3f}**: the inventory-labelled fiber is **{_median_quieter:.1f} times quieter** than its adjacent unlabelled cable. Every hole agrees; individual ratios run from {_minimum:.3f} to {_maximum:.3f}, or {_minimum_quieter:.1f} to {_maximum_quieter:.1f} times quieter inside. The median 0.2-second difference scatter within the vertical- and inclined-hole cables is {_vertical:.3f} and {_inclined:.3f} nanostrain, respectively, and the four depth bins differ by only a factor of {_depth_spread:.2f}. This quantifies a strong, consistent registration feature across two cable designs and {_hole_count} installations, but part of that consistency is selection by construction.

    Across the {loss_segment_count} eligible uncoupled segments, there is no detected monotonic association between difference scatter and cumulative listed discrete-event loss: Spearman $\rho={loss_rho:.2f}$ with $p={loss_pvalue:.2f}$. This low-power result is also a distance-trend test because cumulative loss only increases down the link. It does not show that optical loss is irrelevant, test distributed attenuation, or estimate local before/after changes at individual events.

    Cemented attachment remains a plausible explanation for the registered quiet bands, but testing it causally would require independently positioned intervals or an independent dataset and registration procedure. The coupled level contains the measurement background plus whatever quiet rock strain remains. The practical distinction still matters: an apparent spatial anomaly on unlabelled cable is not automatically ground deformation, and cumulative listed event loss alone does not provide a correction for it in this window.

    These are relative, short-window results from one 5 Hz product. They do not calibrate the interrogator noise floor, prove that lower variability means better dynamic sensitivity, independently validate the inventory placement, or separate every source of phase noise. A controlled coupling test would need independent registration, repeated quiet windows, and an independent excitation.
    """)
    return


@app.cell(hide_code=True)
def _(hole_summary, loss_segment_count, mo):
    _hole_count = len(hole_summary)
    _median_quieter = 1 / hole_summary["inside/flank"].median()
    mo.md(f"""
    ## Key points

    - All {_hole_count} inventory coupling intervals align with deep troughs in quiet LF-DAS variability because that phenomenon helped place the DAS path; this is not independent validation.
    - The notebook quantifies the registration feature: the intervals documented as cemented are {_median_quieter:.1f} times quieter than their own nearby unlabelled controls across those {_hole_count} holes.
    - {loss_segment_count} uncoupled inter-event segments show no detected monotonic association with cumulative listed discrete-event loss; this does not rule out other optical effects.
    - A causal coupling claim needs independently positioned intervals or an independent dataset and registration procedure.
    """)
    return


if __name__ == "__main__":
    app.run()
