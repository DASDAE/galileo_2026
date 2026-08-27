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
    # Did the blast leave permanent strain?

    The high-rate DAS record ends only 3.2 seconds after the first arrival. At the bottom of N180, its phase does not return to the pre-blast level before the file ends. That could be permanent rock deformation, but it could also be a slowly recovering response of the fiber or interrogator.

    This notebook follows the same channels into the 5 Hz record for another eighteen minutes, then asks an independent Brillouin strain sensor whether any step remains ten minutes to several hours later. The U-shaped cable in every borehole gives us a valuable internal check: its down- and upgoing legs occupy the same ground but are separate fiber channels.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import optimize, stats

    from galileo_2026 import get_data_path, get_inventory_path

    plt.rcParams["figure.figsize"] = (11, 5)
    return dc, get_data_path, get_inventory_path, np, optimize, pd, plt, stats


@app.cell
def _(dc, get_data_path, get_inventory_path, np):
    _inventory = dc.inventory(get_inventory_path())
    enriched = (
        dc.spool(get_data_path())
        .update()
        .attach_inventory(_inventory)
        .enrich()
    )

    # A rounded marker for the first arrival at N180 in the 2 kHz patch.
    event_time = np.datetime64("2026-08-06T10:13:26.800", "ns")
    return enriched, event_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Following the offset in the 5 Hz stream

    The low-frequency product is split into 22 one-minute files with 0.4 to 1.3 second gaps. We deliberately do **not** merge them with a relaxed tolerance: that would regularize their timestamps. Instead, each file is converted from phase to strain and reduced to one value per leg before its original time coordinate is concatenated.

    We select the deepest five metres of N180 (`hole_depth >= 25 m`). Each trace is referenced to its own mean from 180 to 10 seconds before the first arrival, so agreement between the two legs cannot be imposed by a shared baseline.
    """)
    return


@app.cell
def _(enriched, event_time, np, pd):
    lf_parts = list(enriched.select(tag="DAS_LF"))
    assert len(lf_parts) == 22
    assert all(part.dims == ("time", "distance") for part in lf_parts)

    _region_specs = {
        "N180 bottom down": ("N180", "down", 25, 100),
        "N180 bottom up": ("N180", "up", 25, 100),
        "N140 middle down": ("N140", "down", 10, 20),
        "N140 bottom down": ("N140", "down", 25, 100),
        "N160 middle down": ("N160", "down", 10, 20),
        "N160 middle up": ("N160", "up", 10, 20),
        "N160 bottom down": ("N160", "down", 25, 100),
    }
    _time_parts = []
    _trace_parts = {name: [] for name in _region_specs}
    _selection_rows = []

    for _part_number, _part in enumerate(lf_parts):
        _strain = _part.radians_to_strain()
        _borehole = np.asarray(_strain.get_array("borehole")).astype(str)
        _leg = np.asarray(_strain.get_array("leg")).astype(str)
        _depth = np.asarray(_strain.get_array("hole_depth"), dtype=float)
        _distance = np.asarray(_strain.get_array("distance"), dtype=float)
        _data = np.asarray(_strain.data) * 1e6
        _time_parts.append(_strain.get_array("time"))

        for _name, (_hole, _side, _low, _high) in _region_specs.items():
            _mask = (
                (_borehole == _hole)
                & (_leg == _side)
                & (_depth >= _low)
                & (_depth < _high)
            )
            assert _mask.any()
            _trace_parts[_name].append(_data[:, _mask].mean(axis=1))
            if _part_number == 0:
                _selection_rows.append(
                    {
                        "region": _name,
                        "channels": int(_mask.sum()),
                        "distance_min_m": _distance[_mask].min(),
                        "distance_max_m": _distance[_mask].max(),
                    }
                )

    _lf_time = np.concatenate(_time_parts)
    _order = np.argsort(_lf_time)
    _lf_time = _lf_time[_order]
    lf_seconds = (_lf_time - event_time) / np.timedelta64(1, "s")
    _baseline = (lf_seconds >= -180) & (lf_seconds <= -10)
    assert _baseline.any(), "The LF record does not cover the baseline window."

    lf_region_traces = {}
    for _name, _parts in _trace_parts.items():
        _trace = np.concatenate(_parts)[_order]
        lf_region_traces[_name] = _trace - _trace[_baseline].mean()

    lf_down = lf_region_traces["N180 bottom down"]
    lf_up = lf_region_traces["N180 bottom up"]
    lf_pair = (lf_down + lf_up) / 2
    _selection_frame = pd.DataFrame(_selection_rows)
    lf_selection = _selection_frame[
        _selection_frame["region"].str.startswith("N180")
    ]

    _bin_rows = []
    for _label, _low, _high in (
        ("2-5", 2, 5),
        ("5-10", 5, 10),
        ("10-20", 10, 20),
        ("20-40", 20, 40),
        ("40-60", 40, 60),
        ("60-120", 60, 120),
        ("120-300", 120, 300),
    ):
        _inside = (lf_seconds >= _low) & (lf_seconds < _high)
        _bin_rows.append(
            {
                "seconds after arrival": _label,
                "samples": int(_inside.sum()),
                "down [µε]": np.median(lf_down[_inside]),
                "up [µε]": np.median(lf_up[_inside]),
                "paired mean [µε]": np.median(lf_pair[_inside]),
            }
        )
    lf_bin_summary = pd.DataFrame(_bin_rows)

    lf_selection.round(3)
    return (
        lf_bin_summary,
        lf_down,
        lf_pair,
        lf_region_traces,
        lf_seconds,
        lf_selection,
        lf_up,
    )


@app.cell
def _(lf_bin_summary):
    lf_bin_summary.round(3)
    return


@app.cell
def _(
    lf_down,
    lf_pair,
    lf_region_traces,
    lf_seconds,
    lf_up,
    np,
    optimize,
    pd,
    stats,
):
    def _decay(_seconds, _amplitude, _tau, _offset):
        return _amplitude * np.exp(-_seconds / _tau) + _offset

    _fit_window = (lf_seconds >= 2) & (lf_seconds <= 60)
    _bounds = ([-100, 0.2, -10], [100, 200, 10])
    _fit_rows = []
    _fit_parameters = {}
    for _name, _trace in (
        ("down", lf_down),
        ("up", lf_up),
        ("paired mean", lf_pair),
    ):
        _parameters, _ = optimize.curve_fit(
            _decay,
            lf_seconds[_fit_window],
            _trace[_fit_window],
            p0=(-8, 15, 0),
            bounds=_bounds,
        )
        _prediction = _decay(lf_seconds[_fit_window], *_parameters)
        _rmse = np.sqrt(np.mean((_trace[_fit_window] - _prediction) ** 2))
        _fit_parameters[_name] = _parameters
        _fit_rows.append(
            {
                "trace": _name,
                "A [µε]": _parameters[0],
                "tau [s]": _parameters[1],
                "C [µε]": _parameters[2],
                "RMSE [µε]": _rmse,
            }
        )
    lf_fit_summary = pd.DataFrame(_fit_rows)
    lf_fit_parameters = _fit_parameters["paired mean"]

    lf_leg_correlation = stats.pearsonr(
        lf_down[_fit_window], lf_up[_fit_window]
    ).statistic
    lf_leg_difference_rmse = np.sqrt(
        np.mean((lf_down[_fit_window] - lf_up[_fit_window]) ** 2)
    )

    _stability_rows = []
    for _stop in (30, 40, 60, 80, 120):
        _window = (lf_seconds >= 2) & (lf_seconds <= _stop)
        _parameters, _ = optimize.curve_fit(
            _decay,
            lf_seconds[_window],
            lf_pair[_window],
            p0=(-8, 15, 0),
            bounds=_bounds,
        )
        _stability_rows.append(
            {
                "fit interval [s]": f"2-{_stop}",
                "A [µε]": _parameters[0],
                "tau [s]": _parameters[1],
                "C [µε]": _parameters[2],
            }
        )
    lf_tau_stability = pd.DataFrame(_stability_rows)

    _regional_rows = []
    for _name in (
        "N140 middle down",
        "N140 bottom down",
        "N160 middle down",
        "N160 middle up",
        "N160 bottom down",
    ):
        _trace = lf_region_traces[_name]
        _initial = np.median(_trace[(lf_seconds >= 2) & (lf_seconds < 5)])
        _parameters, _ = optimize.curve_fit(
            _decay,
            lf_seconds[_fit_window],
            _trace[_fit_window],
            p0=(_initial * np.exp(3.5 / 15), 15, 0),
            bounds=_bounds,
        )
        _regional_rows.append(
            {
                "region": _name,
                "A [µε]": _parameters[0],
                "tau [s]": _parameters[1],
                "C [µε]": _parameters[2],
            }
        )
    lf_regional_fits = pd.DataFrame(_regional_rows)

    lf_fit_summary.round(4)
    return (
        lf_fit_parameters,
        lf_fit_summary,
        lf_leg_correlation,
        lf_leg_difference_rmse,
        lf_regional_fits,
        lf_tau_stability,
    )


@app.cell
def _(
    lf_bin_summary,
    lf_down,
    lf_fit_parameters,
    lf_leg_difference_rmse,
    lf_pair,
    lf_seconds,
    lf_up,
    np,
    plt,
):
    _fig, _axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, height_ratios=(3, 1)
    )
    _shown = (lf_seconds >= -30) & (lf_seconds <= 300)
    _axes[0].plot(
        lf_seconds[_shown],
        lf_down[_shown],
        color="C0",
        lw=1,
        label="down leg",
    )
    _axes[0].plot(
        lf_seconds[_shown],
        lf_up[_shown],
        color="C1",
        lw=1,
        label="up leg",
    )
    _fit_seconds = np.linspace(2, 120, 500)
    _amplitude, _tau, _offset = lf_fit_parameters
    _axes[0].plot(
        _fit_seconds,
        _amplitude * np.exp(-_fit_seconds / _tau) + _offset,
        color="0.1",
        ls="--",
        lw=2,
        label="paired exponential fit",
    )
    _bin_midpoints = np.asarray([3.5, 7.5, 15, 30, 50, 90, 210])
    _axes[0].scatter(
        _bin_midpoints,
        lf_bin_summary["paired mean [µε]"],
        color="0.1",
        marker="o",
        zorder=4,
        label="bin medians",
    )
    _axes[0].axhline(0, color="0.5", lw=1)
    _axes[0].axvline(0, color="C3", lw=1)
    _axes[0].set(
        ylabel="strain relative to pre-blast mean [µε]",
        title="Deep N180 strain returns to baseline in about one minute",
    )
    _axes[0].legend(ncols=2)

    _difference_window = (lf_seconds >= 2) & (lf_seconds <= 60)
    _axes[1].plot(
        lf_seconds[_difference_window],
        (lf_down - lf_up)[_difference_window],
        color="C2",
        lw=1,
    )
    _axes[1].axhline(0, color="0.5", lw=1)
    _axes[1].set(
        xlabel="seconds from first N180 arrival",
        ylabel="down - up [µε]",
        title=(
            f"Co-located leg difference: {lf_leg_difference_rmse:.3f} µε RMS"
        ),
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(lf_regional_fits, lf_tau_stability, mo):
    mo.vstack(
        [
            mo.md("**Sensitivity to the end of the fit interval**"),
            lf_tau_stability.round(3),
            mo.md("**The same decay fitted to comparison regions**"),
            lf_regional_fits.round(3),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    lf_bin_summary,
    lf_fit_parameters,
    lf_fit_summary,
    lf_leg_correlation,
    lf_leg_difference_rmse,
    lf_regional_fits,
    lf_tau_stability,
    mo,
):
    _amplitude, _tau, _offset = lf_fit_parameters
    _fit_rmse = lf_fit_summary.loc[
        lf_fit_summary["trace"] == "paired mean", "RMSE [µε]"
    ].iloc[0]
    _early = lf_bin_summary.iloc[0]
    _minute = lf_bin_summary.loc[
        lf_bin_summary["seconds after arrival"] == "60-120"
    ].iloc[0]
    _late = lf_bin_summary.loc[
        lf_bin_summary["seconds after arrival"] == "120-300"
    ].iloc[0]
    _tau_min = lf_tau_stability["tau [s]"].min()
    _tau_max = lf_tau_stability["tau [s]"].max()
    _regional_tau_median = lf_regional_fits["tau [s]"].median()
    mo.md(rf"""
    ### A transient, not a permanent offset

    The paired fit is $y=Ae^{{-t/\tau}}+C$, with $A={_amplitude:.3f}$ µε, $\tau={_tau:.3f}$ s, and $C={_offset:.3f}$ µε. Its RMSE is {_fit_rmse:.4f} µε. From 2 to 60 seconds the two legs correlate at $r={lf_leg_correlation:.5f}$ and differ by only {lf_leg_difference_rmse:.3f} µε RMS. At 2-5 seconds their medians are {_early["down [µε]"]:+.3f} and {_early["up [µε]"]:+.3f} µε; by 60-120 seconds they are {_minute["down [µε]"]:+.3f} and {_minute["up [µε]"]:+.3f} µε, and over 120-300 seconds they are {_late["down [µε]"]:+.3f} and {_late["up [µε]"]:+.3f} µε.

    The fit is not fragile: changing its end from 30 to 120 seconds keeps $\tau$ between {_tau_min:.2f} and {_tau_max:.2f} seconds. Several smaller responses at N140 and N160, including responses of opposite sign, recover with a median fitted time constant of {_regional_tau_median:.2f} seconds.

    That repeatability establishes that the offset is a real, spatially localized feature of the recorded phase, but it does **not** by itself establish a rock-mechanical relaxation. Both legs and both DAS products share one interrogator. A common phase-tracking or processing response to the blast could reproduce the same decay, and the near-uniform time constant across positive and negative channel groups makes that possibility important. The safe conclusion is narrower: the high-rate step does not persist in the continuous DAS record.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An independent long-timescale test

    The Brillouin instrument samples the other fiber about once every ten minutes. It cannot resolve a 15-second recovery, but it can test whether strain remains after the DAS transient is gone.

    We merge the DSS files at their native tolerance and discard channels past the fiber end, which falls at 2598.1 m on this instrument's own axis. For each N180 leg we average all channels deeper than 2 m. The top two metres are excluded because the borehole collar is a transition from uncemented to cemented cable and the instrument resolves one metre while sampling every 0.1 m. The paired-leg check below shows that this boundary is not trustworthy for a permanent-strain estimate.

    A local regression uses the sweeps within three hours of the blast:

    $$
    y(t)=\beta_0+\beta_1t+\beta_{\mathrm{step}}I(t\geq0)+\epsilon.
    $$

    The linear term absorbs the slow drift visible through the day. A Brillouin sweep integrates for about ten minutes, so the sweep whose `time` plus `sample_span` crosses the blast is excluded rather than assigned wholesale to one side. Every other sweep is represented at the midpoint of its integration span. The interval reported for $\beta_{\mathrm{step}}$ is a two-sided 95% Student-$t$ interval from the ordinary least-squares residual variance with $n-3$ degrees of freedom. The spatial channels have already been averaged; the important unmodelled dependence is temporal autocorrelation between sweeps. The interval may therefore be miscalibrated and is a model-based detection interval, not a full uncertainty budget or a hard upper bound.
    """)
    return


@app.cell
def _(enriched, event_time, np):
    _dss_spool = enriched.select(tag="DSS")
    dss_file_count = len(_dss_spool)
    dss_day = _dss_spool.chunk(time=None, conflict="drop")[0].select(
        distance=(..., 2598.1)
    )
    assert dss_day.dims == ("time", "distance")

    dss_time = dss_day.get_array("time")
    _sample_span = dss_day.get_array("sample_span")
    dss_straddles_event = (dss_time < event_time) & (
        dss_time + _sample_span > event_time
    )
    _dss_midpoint_time = dss_time + _sample_span / 2
    dss_hours = (_dss_midpoint_time - event_time) / np.timedelta64(1, "h")
    dss_straddle_count = int(dss_straddles_event.sum())
    dss_data = np.asarray(dss_day.data, dtype=float)
    _borehole = np.asarray(dss_day.get_array("borehole")).astype(str)
    _leg = np.asarray(dss_day.get_array("leg")).astype(str)
    _depth = np.asarray(dss_day.get_array("hole_depth"), dtype=float)
    _distance = np.asarray(dss_day.get_array("distance"), dtype=float)

    _down = (_borehole == "N180") & (_leg == "down") & (_depth >= 2)
    _up = (_borehole == "N180") & (_leg == "up") & (_depth >= 2)
    assert _down.any() and _up.any(), "N180 legs were not resolved in DSS."

    dss_down = dss_data[:, _down].mean(axis=1)
    dss_up = dss_data[:, _up].mean(axis=1)
    dss_pair = (dss_down + dss_up) / 2
    dss_global = np.median(dss_data, axis=1)

    dss_coords = {
        "borehole": _borehole,
        "leg": _leg,
        "hole_depth": _depth,
    }
    return (
        dss_coords,
        dss_data,
        dss_day,
        dss_down,
        dss_file_count,
        dss_global,
        dss_hours,
        dss_pair,
        dss_straddle_count,
        dss_straddles_event,
        dss_time,
        dss_up,
    )


@app.cell
def _(
    dss_coords,
    dss_data,
    dss_down,
    dss_global,
    dss_hours,
    dss_pair,
    dss_straddles_event,
    dss_up,
    np,
    pd,
    stats,
):
    def _step_fit(_trace, _half_window):
        _inside = (np.abs(dss_hours) <= _half_window) & ~dss_straddles_event
        _design = np.column_stack(
            [
                np.ones(_inside.sum()),
                dss_hours[_inside],
                (dss_hours[_inside] >= 0).astype(float),
            ]
        )
        _response = _trace[_inside]
        _coefficients = np.linalg.lstsq(_design, _response, rcond=None)[0]
        _residual = _response - _design @ _coefficients
        _degrees_freedom = len(_response) - _design.shape[1]
        _variance = _residual @ _residual / _degrees_freedom
        _covariance = _variance * np.linalg.inv(_design.T @ _design)
        _standard_error = np.sqrt(_covariance[2, 2])
        _half_width = stats.t.ppf(0.975, _degrees_freedom) * _standard_error
        _lag1 = (
            np.corrcoef(_residual[:-1], _residual[1:])[0, 1]
            if len(_residual) > 2
            else np.nan
        )
        return {
            "samples": int(_inside.sum()),
            "step": _coefficients[2],
            "half_width": _half_width,
            "trend": _coefficients[1],
            "lag1": _lag1,
            "coefficients": _coefficients,
        }

    _local_pair = dss_pair - dss_global
    _fits = {
        "N180 down": _step_fit(dss_down, 3),
        "N180 up": _step_fit(dss_up, 3),
        "N180 paired mean": _step_fit(dss_pair, 3),
        "global spatial median": _step_fit(dss_global, 3),
        "paired after median subtraction": _step_fit(_local_pair, 3),
    }
    dss_fit_models = {
        _name: _fit["coefficients"] for _name, _fit in _fits.items()
    }
    dss_fit_summary = pd.DataFrame(
        [
            {
                "trace": _name,
                "samples": _fit["samples"],
                "step [µε]": _fit["step"],
                "95% half-width [µε]": _fit["half_width"],
                "lower [µε]": _fit["step"] - _fit["half_width"],
                "upper [µε]": _fit["step"] + _fit["half_width"],
                "lag-1 residual r": _fit["lag1"],
            }
            for _name, _fit in _fits.items()
        ]
    )

    _sensitivity_rows = []
    for _half_window in (2, 3, 5):
        _fit = _step_fit(dss_pair, _half_window)
        _sensitivity_rows.append(
            {
                "half-window [h]": _half_window,
                "samples": _fit["samples"],
                "step [µε]": _fit["step"],
                "95% half-width [µε]": _fit["half_width"],
            }
        )
    dss_sensitivity = pd.DataFrame(_sensitivity_rows)

    _borehole = dss_coords["borehole"]
    _leg = dss_coords["leg"]
    _depth = dss_coords["hole_depth"]
    _collar_rows = []
    for _hole in ("N120", "N180"):
        for _side in ("down", "up"):
            _mask = (_borehole == _hole) & (_leg == _side) & (_depth < 2)
            _fit = _step_fit(dss_data[:, _mask].mean(axis=1), 3)
            _collar_rows.append(
                {
                    "borehole": _hole,
                    "leg": _side,
                    "channels": int(_mask.sum()),
                    "step [µε]": _fit["step"],
                    "95% half-width [µε]": _fit["half_width"],
                }
            )
    dss_collar_summary = pd.DataFrame(_collar_rows)

    dss_fit_summary.round(3)
    return (
        dss_collar_summary,
        dss_fit_models,
        dss_fit_summary,
        dss_sensitivity,
    )


@app.cell
def _(dss_collar_summary, dss_sensitivity, mo):
    mo.vstack(
        [
            mo.md(
                "**Sensitivity of the paired estimate to regression window**"
            ),
            dss_sensitivity.round(3),
            mo.md("**Why the top two metres are excluded**"),
            dss_collar_summary.round(3),
        ]
    )
    return


@app.cell
def _(
    dss_down,
    dss_fit_models,
    dss_global,
    dss_hours,
    dss_pair,
    dss_straddles_event,
    dss_up,
    np,
    plt,
):
    _shown = (np.abs(dss_hours) <= 3) & ~dss_straddles_event
    _pre = (dss_hours >= -3) & (dss_hours < 0) & ~dss_straddles_event
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for _name, _trace, _color in (
        ("N180 down", dss_down, "C0"),
        ("N180 up", dss_up, "C1"),
    ):
        _center = _trace[_pre].mean()
        _axes[0].plot(
            dss_hours[_shown],
            _trace[_shown] - _center,
            "o-",
            color=_color,
            ms=3,
            lw=1,
            label=_name,
        )
        _coefficient = dss_fit_models[_name]
        _prediction = (
            _coefficient[0]
            + _coefficient[1] * dss_hours[_shown]
            + _coefficient[2] * (dss_hours[_shown] >= 0)
        )
        _axes[0].plot(
            dss_hours[_shown],
            _prediction - _center,
            color=_color,
            ls="--",
            lw=2,
        )
    _axes[0].set(
        xlabel="hours from first N180 arrival",
        ylabel="strain relative to pre-blast mean [µε]",
        title="Independent N180 legs: no resolved step",
    )
    _axes[0].legend()

    _local_pair = dss_pair - dss_global
    for _name, _trace, _color in (
        ("global spatial median", dss_global, "0.35"),
        ("N180 minus global median", _local_pair, "C3"),
    ):
        _center = _trace[_pre].mean()
        _axes[1].plot(
            dss_hours[_shown],
            _trace[_shown] - _center,
            "o-",
            color=_color,
            ms=3,
            lw=1,
            label=_name,
        )
    _axes[1].set(
        xlabel="hours from first N180 arrival",
        title="Subtraction creates an apparent step estimate",
    )
    _axes[1].legend()

    for _axis in _axes:
        _axis.axvline(0, color="C3", lw=1, alpha=0.6)
        _axis.axhline(0, color="0.7", lw=1)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    dss_collar_summary,
    dss_file_count,
    dss_fit_summary,
    dss_sensitivity,
    dss_straddle_count,
    mo,
):
    _by_name = dss_fit_summary.set_index("trace")
    _down = _by_name.loc["N180 down"]
    _up = _by_name.loc["N180 up"]
    _pair = _by_name.loc["N180 paired mean"]
    _global = _by_name.loc["global spatial median"]
    _corrected = _by_name.loc["paired after median subtraction"]
    _unresolved = all(
        abs(_row["step [µε]"]) <= _row["95% half-width [µε]"]
        for _, _row in dss_sensitivity.iterrows()
    )
    _heading = (
        "No independently resolved lasting strain"
        if _unresolved
        else "A lasting strain step is resolved"
    )
    _interval_statement = (
        "Zero lies inside every interval, and the paired result remains unresolved with two-, three-, and five-hour fitting windows."
        if _unresolved
        else "At least one fitting window excludes zero; the result requires a physical follow-up rather than a null interpretation."
    )
    _collars = dss_collar_summary.set_index(["borehole", "leg"])
    _n180_down_collar = _collars.loc[("N180", "down"), "step [µε]"]
    _n180_up_collar = _collars.loc[("N180", "up"), "step [µε]"]
    _n120_down_collar = _collars.loc[("N120", "down"), "step [µε]"]
    _n120_up_collar = _collars.loc[("N120", "up"), "step [µε]"]
    mo.md(rf"""
    ### {_heading}

    The merged day contains {dss_file_count} DSS files; {_pair["samples"]:.0f} sweeps fall within ±3 hours after excluding {dss_straddle_count} sweep that crosses the event. Away from the collars, the step estimates are **{_down["step [µε]"]:.3f} ± {_down["95% half-width [µε]"]:.3f} µε** on the down leg and **{_up["step [µε]"]:.3f} ± {_up["95% half-width [µε]"]:.3f} µε** on the up leg. Their paired mean is **{_pair["step [µε]"]:.3f} ± {_pair["95% half-width [µε]"]:.3f} µε**. {_interval_statement} The paired fit's lag-one residual correlation is {_pair["lag-1 residual r"]:.2f}; the Student-$t$ interval ignores this temporal structure, may be miscalibrated, and should not be treated as a hard upper bound.

    A tempting correction would give the opposite impression. The global spatial median has an apparent step of {_global["step [µε]"]:.3f} ± {_global["95% half-width [µε]"]:.3f} µε. Subtracting it changes the paired N180 estimate algebraically to **{_corrected["step [µε]"]:+.3f} µε**. That does not reveal a hidden local deformation; it imports an archive-wide reference-like change into every borehole. In this test the raw, replicated legs are the safer comparison.

    The collar channels fail the same replication test dramatically. At N180 the top-two-metre estimates are {_n180_down_collar:+.2f} µε down versus {_n180_up_collar:+.2f} µε up; at N120 they are {_n120_down_collar:+.2f} versus {_n120_up_collar:+.2f} µε. Co-located rock cannot plausibly produce those leg-specific steps. Spatial smearing across the loose-to-cemented boundary, cable-specific strain, and Brillouin fitting artifacts are all possible, so those channels cannot support a permanent-rock-strain claim.
    """)
    return


@app.cell(hide_code=True)
def _(dss_fit_summary, lf_fit_parameters, mo):
    _amplitude, _tau, _ = lf_fit_parameters
    _by_name = dss_fit_summary.set_index("trace")
    _pair = _by_name.loc["N180 paired mean"]
    _corrected = _by_name.loc["paired after median subtraction"]
    _dss_statement = (
        "the independent DSS fiber does not resolve a persistent N180 step away from its collar"
        if abs(_pair["step [µε]"]) <= _pair["95% half-width [µε]"]
        else "the independent DSS fiber resolves a persistent N180 step that requires follow-up"
    )
    mo.md(f"""
    ## Scientific conclusion

    The apparent high-rate offset at the bottom of N180 is **transient in the recorded DAS strain**, not permanent: two co-located fiber legs reproduce an exponential response with fitted amplitude {_amplitude:.2f} µε and time constant {_tau:.2f} seconds, and both return to their pre-blast level within about a minute. Ten minutes later and over the following hours, {_dss_statement}.

    This does not prove that the rock experienced no permanent deformation. The model-based DSS interval does not account for temporal dependence, the two cables need not transfer strain identically, and neither fiber measures the full strain tensor. Nor does the decay alone separate rock relaxation from a shared interrogator response. What the archive does rule out is the simple reading of the last samples in the five-second high-rate file as evidence for a lasting offset.

    ### Key points

    - Follow an apparent step beyond the event file before calling it permanent.
    - Co-located down/up legs are powerful controls for both signal replication and boundary artifacts.
    - Common-mode subtraction is a physical assumption, not a neutral cleanup step; here it changes the fitted local step to {_corrected["step [µε]"]:+.2f} µε.
    - The independent sensor constrains persistent N180 axial strain but does not identify the origin of the 15-second DAS recovery.
    """)
    return


if __name__ == "__main__":
    app.run()
