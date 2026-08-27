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
    # The first blast front across the north holes

    **Question.** How does the isolated first blast arrival move across the five north boreholes, and do the two fiber legs in each U-shaped hole record the same local wavefield?

    A planar fit in two projected coordinates gives a compact answer. One coordinate projects each channel's local-frame `x,y` onto the nearly straight chain of borehole collars; the other is depth along a hole. The paired down/up legs then provide an unusually direct internal check because they visit nearly the same rock twice on different sections of fiber.

    The result is deliberately described as **apparent** or **projected** velocity. A single fiber component, five nearly collinear collars, and one arrival do not determine a material wave speed, a three-dimensional slowness vector, or a source location.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import least_squares
    from scipy.signal import correlate, correlation_lags

    from galileo_2026 import get_data_path, get_inventory_path

    band_hz = (8, 180)
    borehole_names = ("N100", "N120", "N140", "N160", "N180")
    moveout_max_lag_s = 0.10
    null_min_depth_separation_m = 10.0
    validation_max_lag_s = 0.005
    window_s = (1.60, 2.10)

    plt.rcParams["figure.figsize"] = (10, 6)
    return (
        band_hz,
        borehole_names,
        correlate,
        correlation_lags,
        dc,
        get_data_path,
        get_inventory_path,
        least_squares,
        moveout_max_lag_s,
        null_min_depth_separation_m,
        np,
        pd,
        plt,
        validation_max_lag_s,
        window_s,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and processing

    The full-rate DAS file spans five seconds at 2 kHz. Its hive-style directory attributes identify the acquisition, so attaching the inventory and calling `enrich` adds `borehole`, `leg`, `hole_depth`, `x`, `y`, and `z` to every channel without any hard-coded optical-distance ranges.

    The workflow converts phase to strain, applies an 8-180 Hz Butterworth bandpass to the full record, and differentiates in time. Filtering before cutting the 0.50 s analysis window keeps filter edges away from the arrival. Cross-correlation uses the Hilbert envelope of strain rate because its broad pulse is less sensitive than the signed carrier to small differences in channel response.

    This notebook was developed and checked with DASCore's `dev` branch at commit `ef094ba8d047422d388d719c080dc94f90cefb5e`, where the inventory and enrichment APIs used here are still development features.
    """)
    return


@app.cell
def _(dc, get_data_path, get_inventory_path):
    inventory = dc.inventory(get_inventory_path())
    enriched_das = (
        dc.spool(get_data_path())
        .update()
        .select(tag="DAS")
        .attach_inventory(inventory)
        .enrich()
    )
    assert len(enriched_das) == 1
    blast_patch = enriched_das[0]
    assert blast_patch.dims == ("time", "distance")
    blast_patch
    return blast_patch, inventory


@app.cell
def _(band_hz, blast_patch, dc, window_s):
    strain_rate_full = (
        blast_patch.radians_to_strain()
        .pass_filter(time=band_hz)
        .differentiate("time")
    )
    envelope_full = strain_rate_full.envelope("time")
    strain_rate = strain_rate_full.select(time=window_s, relative=True)
    blast_envelope = envelope_full.select(time=window_s, relative=True)
    sample_interval_s = float(dc.to_float(strain_rate.get_coord("time").step))
    print(
        f"analysis window: {strain_rate.shape[0]} samples at "
        f"{1 / sample_interval_s:.0f} Hz"
    )
    return blast_envelope, sample_interval_s, strain_rate


@app.cell
def _(correlate, correlation_lags, np):
    def normalized_peak_lag(trace, reference, sample_interval_s, max_lag_s):
        """Return the strongest normalized correlation and its lag."""
        _trace = np.asarray(trace, dtype=float)
        _reference = np.asarray(reference, dtype=float)
        _trace = _trace - _trace.mean()
        _reference = _reference - _reference.mean()
        _normalizer = np.linalg.norm(_trace) * np.linalg.norm(_reference)
        assert _normalizer > 0
        _correlation = (
            correlate(_trace, _reference, mode="full", method="fft")
            / _normalizer
        )
        _lag_samples = correlation_lags(
            _trace.size, _reference.size, mode="full"
        )
        _allowed = np.abs(_lag_samples * sample_interval_s) <= max_lag_s
        _candidate_indices = np.flatnonzero(_allowed)
        _peak_index = _candidate_indices[np.argmax(_correlation[_allowed])]
        return (
            float(_lag_samples[_peak_index] * sample_interval_s),
            float(_correlation[_peak_index]),
        )

    return (normalized_peak_lag,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Envelope moveout

    Only the downgoing legs enter the fit, so each physical location is represented once. Every channel's envelope is demeaned and cross-correlated against the deepest sampled channel on the N180 downgoing leg. Positive lag means that a trace follows the reference; no sub-sample interpolation is used, so picks remain on the native 0.5 ms grid. The horizontal predictor uses every channel's enriched `x,y`, not a single collar value for an entire inclined hole. `hole_depth` then describes the remaining pooled moveout along five not-quite-parallel trajectories, so its coefficient remains a descriptive projection rather than a material velocity.
    """)
    return


@app.cell
def _(
    blast_envelope,
    moveout_max_lag_s,
    normalized_peak_lag,
    np,
    sample_interval_s,
):
    down_envelope = blast_envelope.select(borehole="N*", leg="down")
    n180_down_envelope = down_envelope.select(borehole="N180")
    _reference_index = np.argmax(n180_down_envelope.get_array("hole_depth"))
    reference_envelope = n180_down_envelope.data[:, _reference_index]
    _arrival_picks = np.asarray(
        [
            normalized_peak_lag(
                down_envelope.data[:, _index],
                reference_envelope,
                sample_interval_s,
                moveout_max_lag_s,
            )
            for _index in range(down_envelope.shape[1])
        ]
    )
    arrival_lag_s = _arrival_picks[:, 0]
    arrival_correlation = _arrival_picks[:, 1]
    channel_borehole = down_envelope.get_array("borehole")
    channel_hole_depth_m = down_envelope.get_array("hole_depth")
    assert set(channel_borehole) == {
        "N100",
        "N120",
        "N140",
        "N160",
        "N180",
    }
    return (
        arrival_correlation,
        arrival_lag_s,
        channel_borehole,
        channel_hole_depth_m,
        down_envelope,
        reference_envelope,
    )


@app.cell
def _(borehole_names, inventory, np):
    _fiber_array = inventory.networks[0].fiber_arrays[0]
    _das_path = next(
        _path
        for _path in _fiber_array.optical_paths
        if _path.location_code == "04"
    )
    _collars = []
    for _borehole in borehole_names:
        _geometry = next(
            _item
            for _item in _das_path.geometry
            if _item.name == f"{_borehole} down"
        )
        _hole_depth = np.asarray(_geometry.columns["hole_depth"])
        _collar_index = int(np.argmin(np.abs(_hole_depth)))
        assert np.isclose(_hole_depth[_collar_index], 0)
        _collars.append(
            [
                _geometry.columns["x"][_collar_index],
                _geometry.columns["y"][_collar_index],
            ]
        )

    collar_xy_m = np.asarray(_collars)
    _centered_collars = collar_xy_m - collar_xy_m.mean(axis=0)
    _, _, _principal_axes = np.linalg.svd(
        _centered_collars, full_matrices=False
    )
    _drift_axis = _principal_axes[0]
    _orientation = np.sign(
        np.dot(collar_xy_m[-1] - collar_xy_m[0], _drift_axis)
    )
    drift_axis_xy = _drift_axis * _orientation
    collar_chainage_m = (collar_xy_m - collar_xy_m[0]) @ drift_axis_xy
    collar_chainage_by_hole = dict(
        zip(borehole_names, collar_chainage_m, strict=True)
    )
    return (
        collar_chainage_by_hole,
        collar_chainage_m,
        collar_xy_m,
        drift_axis_xy,
    )


@app.cell
def _(least_squares, np):
    def huber_plane_fit(design_matrix, lag_ms):
        """Fit a plane with a 1 ms Huber transition."""
        _initial = np.linalg.lstsq(design_matrix, lag_ms, rcond=None)[0]
        return least_squares(
            lambda _coefficients: design_matrix @ _coefficients - lag_ms,
            _initial,
            loss="huber",
            f_scale=1.0,
        ).x

    return (huber_plane_fit,)


@app.cell
def _(
    arrival_correlation,
    arrival_lag_s,
    channel_borehole,
    channel_hole_depth_m,
    collar_xy_m,
    down_envelope,
    drift_axis_xy,
    huber_plane_fit,
    np,
):
    _channel_xy_m = np.column_stack(
        [down_envelope.get_array("x"), down_envelope.get_array("y")]
    )
    channel_chainage_m = (_channel_xy_m - collar_xy_m[0]) @ drift_axis_xy
    arrival_lag_ms = arrival_lag_s * 1_000
    design_matrix = np.column_stack(
        [
            np.ones(arrival_lag_ms.size),
            channel_chainage_m,
            channel_hole_depth_m,
        ]
    )
    fit_coefficients = huber_plane_fit(design_matrix, arrival_lag_ms)
    _predicted_lag_ms = design_matrix @ fit_coefficients
    _residual_ms = arrival_lag_ms - _predicted_lag_ms
    drift_slope_ms_m = float(fit_coefficients[1])
    hole_slope_ms_m = float(fit_coefficients[2])
    drift_apparent_velocity_km_s = 1 / abs(drift_slope_ms_m)
    hole_apparent_velocity_km_s = 1 / abs(hole_slope_ms_m)
    moveout_rms_ms = float(np.sqrt(np.mean(_residual_ms**2)))
    moveout_median_correlation = float(np.median(arrival_correlation))
    print(
        f"drift slope {drift_slope_ms_m:.5f} ms/m -> "
        f"{drift_apparent_velocity_km_s:.3f} km/s apparent"
    )
    print(
        f"hole slope  {hole_slope_ms_m:.5f} ms/m -> "
        f"{hole_apparent_velocity_km_s:.3f} km/s apparent"
    )
    print(
        f"median envelope correlation {moveout_median_correlation:.3f}; "
        f"RMS residual {moveout_rms_ms:.2f} ms"
    )
    return (
        arrival_lag_ms,
        channel_chainage_m,
        design_matrix,
        drift_apparent_velocity_km_s,
        drift_slope_ms_m,
        fit_coefficients,
        hole_apparent_velocity_km_s,
        hole_slope_ms_m,
        moveout_median_correlation,
        moveout_rms_ms,
    )


@app.cell
def _(
    arrival_lag_ms,
    borehole_names,
    channel_borehole,
    channel_chainage_m,
    channel_hole_depth_m,
    fit_coefficients,
    np,
    plt,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _colors = plt.get_cmap("viridis")([0.05, 0.27, 0.50, 0.73, 0.95])
    for _borehole, _color in zip(borehole_names, _colors, strict=True):
        _mask = channel_borehole == _borehole
        _axes[0].scatter(
            channel_chainage_m[_mask],
            arrival_lag_ms[_mask]
            - fit_coefficients[2] * channel_hole_depth_m[_mask],
            s=24,
            alpha=0.8,
            color=_color,
            label=_borehole,
        )
        _axes[1].scatter(
            channel_hole_depth_m[_mask],
            arrival_lag_ms[_mask]
            - fit_coefficients[1] * channel_chainage_m[_mask],
            s=24,
            alpha=0.8,
            color=_color,
        )

    _chain_line = np.array(
        [channel_chainage_m.min(), channel_chainage_m.max()]
    )
    _depth_line = np.array(
        [channel_hole_depth_m.min(), channel_hole_depth_m.max()]
    )
    _axes[0].plot(
        _chain_line,
        fit_coefficients[0] + fit_coefficients[1] * _chain_line,
        color="0.15",
        lw=2,
    )
    _axes[1].plot(
        _depth_line,
        fit_coefficients[0] + fit_coefficients[2] * _depth_line,
        color="0.15",
        lw=2,
    )
    _axes[0].set(
        xlabel="channel projection along north-drift collar axis [m]",
        ylabel="lag corrected for hole depth [ms]",
        title="Projected moveout along the drift",
    )
    _axes[1].set(
        xlabel="hole depth [m]",
        ylabel="lag corrected for projected channel position [ms]",
        title="Projected moveout down the holes",
    )
    _axes[0].legend(frameon=False, ncols=2)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    drift_apparent_velocity_km_s,
    drift_slope_ms_m,
    hole_apparent_velocity_km_s,
    hole_slope_ms_m,
    mo,
    moveout_median_correlation,
    moveout_rms_ms,
):
    mo.md(f"""
    The Huber plane has slopes **{drift_slope_ms_m:.5f} ms/m** along the channel projection onto the collar axis and **{hole_slope_ms_m:.5f} ms/m** per metre of hole depth. Their reciprocals are projected apparent velocities of **{drift_apparent_velocity_km_s:.3f} km/s** and **{hole_apparent_velocity_km_s:.3f} km/s**, respectively. Envelope similarity is high (median normalized correlation **{moveout_median_correlation:.3f}**) and the plane leaves an RMS timing residual of **{moveout_rms_ms:.2f} ms**.

    Both slopes are negative because larger north-drift projection and greater hole depth reach the N180 deep reference sooner. The horizontal coordinate follows each channel through the inclined holes, so its contribution is not folded into the depth term. The depth coefficient still pools five distinct hole trajectories. The unequal projected speeds describe this arrival in two chosen coordinates; they are not evidence by themselves for anisotropic rock velocity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What the envelopes look like

    Each panel below is independently normalized so the weak N100 arrival remains visible. Cyan points place the correlation lag relative to the reference envelope's largest peak. The coherent tilt within and between holes is the structure summarized by the plane fit.
    """)
    return


@app.cell
def _(
    arrival_lag_s,
    blast_envelope,
    borehole_names,
    channel_borehole,
    channel_hole_depth_m,
    dc,
    down_envelope,
    np,
    plt,
    reference_envelope,
):
    _time_s = dc.to_float(
        blast_envelope.get_array("time") - blast_envelope.get_array("time")[0]
    )
    _reference_peak_s = _time_s[np.argmax(reference_envelope)]
    _fig, _axes = plt.subplots(
        1, len(borehole_names), figsize=(14, 5), sharex=True, sharey=True
    )
    for _ax, _borehole in zip(_axes, borehole_names, strict=True):
        _hole_patch = down_envelope.select(borehole=_borehole)
        _depth = _hole_patch.get_array("hole_depth")
        _scale = np.percentile(_hole_patch.data, 99.5)
        _ax.pcolormesh(
            _time_s,
            _depth,
            (_hole_patch.data / _scale).T,
            shading="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        _mask = channel_borehole == _borehole
        _ax.plot(
            _reference_peak_s + arrival_lag_s[_mask],
            channel_hole_depth_m[_mask],
            ".",
            color="cyan",
            ms=5,
        )
        _ax.set_title(_borehole)
        _ax.set_xlabel("time in window [s]")
    _axes[0].set_ylabel("hole depth [m]")
    _axes[0].invert_yaxis()
    _fig.suptitle("First-arrival strain-rate envelopes, normalized per hole")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Amplitude is not simple attenuation

    For a reproducible amplitude comparison, take each downgoing channel's peak envelope in the analysis window and then the median across the hole. This resists a single hot channel while preserving the strong spatial trend.
    """)
    return


@app.cell
def _(
    arrival_correlation,
    borehole_names,
    channel_borehole,
    collar_chainage_by_hole,
    down_envelope,
    np,
    pd,
):
    _time_axis = down_envelope.get_axis("time")
    _channel_peak = down_envelope.data.max(axis=_time_axis) * 1e6
    _rows = []
    for _borehole in borehole_names:
        _mask = channel_borehole == _borehole
        _rows.append(
            {
                "borehole": _borehole,
                "collar chainage [m]": collar_chainage_by_hole[_borehole],
                "down channels": int(_mask.sum()),
                "median peak envelope [µε/s]": float(
                    np.median(_channel_peak[_mask])
                ),
                "median reference correlation": float(
                    np.median(arrival_correlation[_mask])
                ),
            }
        )
    hole_summary = pd.DataFrame(_rows)
    _n100_amplitude = hole_summary.loc[
        hole_summary["borehole"] == "N100",
        "median peak envelope [µε/s]",
    ].iloc[0]
    hole_summary["amplitude / N100"] = (
        hole_summary["median peak envelope [µε/s]"] / _n100_amplitude
    )
    hole_summary.round(
        {
            "collar chainage [m]": 2,
            "median peak envelope [µε/s]": 1,
            "median reference correlation": 3,
            "amplitude / N100": 2,
        }
    )
    return (hole_summary,)


@app.cell
def _(hole_summary, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(
        hole_summary["borehole"],
        hole_summary["amplitude / N100"],
        color="C0",
    )
    _ax.bar_label(_bars, fmt="%.1fx", padding=3)
    _ax.set(
        xlabel="borehole",
        ylabel="median peak envelope relative to N100",
        title="The first-arrival amplitude grows strongly towards N160",
    )
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(hole_summary, mo):
    _n160 = hole_summary.loc[
        hole_summary["borehole"] == "N160", "amplitude / N100"
    ].iloc[0]
    _n180 = hole_summary.loc[
        hole_summary["borehole"] == "N180", "amplitude / N100"
    ].iloc[0]
    mo.md(f"""
    N160's median peak is **{_n160:.1f} times N100's**, while N180 is slightly below N160 at **{_n180:.1f} times N100's**. This is not a geometric-spreading curve: DAS records axial strain, so incidence angle, cable coupling, local rock response, and the source radiation pattern all modulate amplitude. The non-monotonic N160-N180 pair is a warning against turning this five-hole pattern directly into attenuation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Leave one borehole out

    A five-hole, multi-channel fit can be sensitive to any one hole. Repeating the identical 1 ms-transition Huber fit after removing each borehole gives a compact stability test. These ranges are more informative than formal standard errors for such a short, structured array.
    """)
    return


@app.cell
def _(
    arrival_lag_ms,
    borehole_names,
    channel_borehole,
    design_matrix,
    huber_plane_fit,
    np,
    pd,
):
    _rows = []
    for _omitted in borehole_names:
        _keep = channel_borehole != _omitted
        _coefficients = huber_plane_fit(
            design_matrix[_keep], arrival_lag_ms[_keep]
        )
        _rows.append(
            {
                "omitted borehole": _omitted,
                "drift apparent velocity [km/s]": 1 / abs(_coefficients[1]),
                "hole apparent velocity [km/s]": 1 / abs(_coefficients[2]),
            }
        )
    leave_one_out = pd.DataFrame(_rows)
    loo_drift_range = (
        float(leave_one_out["drift apparent velocity [km/s]"].min()),
        float(leave_one_out["drift apparent velocity [km/s]"].max()),
    )
    loo_hole_range = (
        float(leave_one_out["hole apparent velocity [km/s]"].min()),
        float(leave_one_out["hole apparent velocity [km/s]"].max()),
    )
    print(
        f"leave-one-hole-out: drift {loo_drift_range[0]:.2f}-"
        f"{loo_drift_range[1]:.2f} km/s; hole {loo_hole_range[0]:.2f}-"
        f"{loo_hole_range[1]:.2f} km/s"
    )
    leave_one_out.round(3)
    return leave_one_out, loo_drift_range, loo_hole_range


@app.cell(hide_code=True)
def _(loo_drift_range, loo_hole_range, mo):
    mo.md(f"""
    The leave-one-hole-out apparent velocity spans **{loo_drift_range[0]:.2f} to {loo_drift_range[1]:.2f} km/s** along the drift and **{loo_hole_range[0]:.2f} to {loo_hole_range[1]:.2f} km/s** down the holes. The sign and order of magnitude survive every omission, but the spread—especially for the depth term—is large enough that extra source geometry or more arrivals would be needed for a stronger velocity interpretation.
    """)
    return


@app.cell(hide_code=True)
def _(mo, null_min_depth_separation_m):
    mo.md(f"""
    ## The paired legs are an internal validation

    Each borehole is a U: one fiber leg runs down and the other returns through nearly the same rock. For every down-leg channel, select the up-leg channel with nearest `hole_depth` and maximize signed strain-rate correlation only within ±5 ms. The signed carrier is deliberately used here rather than its envelope; it is a stricter test of waveform agreement.

    As a null comparison, reflect each target depth about the deepest sampled point and pair it with the resulting, deliberately wrong, up-leg depth. Near mid-depth that reflection can land back on the co-located channel, so the null keeps only pairs separated by at least {null_min_depth_separation_m:g} m in depth. This retains the same hole, instrument, time window, and processing while ensuring that the null breaks physical co-location.
    """)
    return


@app.cell
def _(
    borehole_names,
    normalized_peak_lag,
    null_min_depth_separation_m,
    np,
    pd,
    sample_interval_s,
    strain_rate,
    validation_max_lag_s,
):
    _pair_rows = []
    for _borehole in borehole_names:
        _hole_patch = strain_rate.select(borehole=_borehole)
        _down = _hole_patch.select(leg="down")
        _up = _hole_patch.select(leg="up")
        _down_depth = _down.get_array("hole_depth")
        _up_depth = _up.get_array("hole_depth")
        _deepest = max(_down_depth.max(), _up_depth.max())
        for _index, _depth in enumerate(_down_depth):
            _same_index = int(np.argmin(np.abs(_up_depth - _depth)))
            _opposite_depth = _deepest - _depth
            _opposite_index = int(
                np.argmin(np.abs(_up_depth - _opposite_depth))
            )
            _down_trace = _down.data[:, _index]
            _same_trace = _up.data[:, _same_index]
            _opposite_trace = _up.data[:, _opposite_index]
            _same_lag, _same_correlation = normalized_peak_lag(
                _down_trace,
                _same_trace,
                sample_interval_s,
                validation_max_lag_s,
            )
            _null_separation = abs(
                _up_depth[_opposite_index] - _up_depth[_same_index]
            )
            if _null_separation >= null_min_depth_separation_m:
                _, _opposite_correlation = normalized_peak_lag(
                    _down_trace,
                    _opposite_trace,
                    sample_interval_s,
                    validation_max_lag_s,
                )
            else:
                _opposite_correlation = np.nan
            _pair_rows.append(
                {
                    "borehole": _borehole,
                    "down depth [m]": float(_depth),
                    "up depth [m]": float(_up_depth[_same_index]),
                    "same-depth correlation": _same_correlation,
                    "opposite-depth correlation": _opposite_correlation,
                    "null depth separation [m]": _null_separation,
                    "same-depth lag [ms]": _same_lag * 1_000,
                    "down/up RMS amplitude": float(
                        np.linalg.norm(_down_trace)
                        / np.linalg.norm(_same_trace)
                    ),
                }
            )

    leg_pairs = pd.DataFrame(_pair_rows)
    validation_median_correlation = float(
        leg_pairs["same-depth correlation"].median()
    )
    null_median_correlation = float(
        leg_pairs["opposite-depth correlation"].median()
    )
    _null_valid = leg_pairs["opposite-depth correlation"].notna()
    validation_matched_median_correlation = float(
        leg_pairs.loc[_null_valid, "same-depth correlation"].median()
    )
    validation_median_abs_lag_ms = float(
        leg_pairs["same-depth lag [ms]"].abs().median()
    )
    validation_median_amplitude_ratio = float(
        leg_pairs["down/up RMS amplitude"].median()
    )
    validation_pair_count = len(leg_pairs)
    null_pair_count = int(
        leg_pairs["opposite-depth correlation"].notna().sum()
    )
    assert validation_pair_count > 0, "No co-located leg pairs were resolved."
    assert null_pair_count > 0, "No depth-separated null pairs were resolved."
    print(
        f"{validation_pair_count} leg pairs: overall same-depth correlation "
        f"{validation_median_correlation:.3f}; matched same-depth "
        f"{validation_matched_median_correlation:.3f} vs opposite-depth null "
        f"{null_median_correlation:.3f} ({null_pair_count} null pairs), "
        f"median |lag| "
        f"{validation_median_abs_lag_ms:.1f} ms, median amplitude ratio "
        f"{validation_median_amplitude_ratio:.2f}"
    )
    return (
        leg_pairs,
        null_median_correlation,
        null_pair_count,
        validation_median_abs_lag_ms,
        validation_median_amplitude_ratio,
        validation_median_correlation,
        validation_matched_median_correlation,
        validation_pair_count,
    )


@app.cell
def _(borehole_names, leg_pairs, np, pd):
    _rows = []
    for _borehole in borehole_names:
        _hole_pairs = leg_pairs[leg_pairs["borehole"] == _borehole]
        _null_valid = _hole_pairs["opposite-depth correlation"].notna()
        _rows.append(
            {
                "borehole": _borehole,
                "pairs": len(_hole_pairs),
                "same-depth correlation": _hole_pairs[
                    "same-depth correlation"
                ].median(),
                "matched same-depth correlation": _hole_pairs.loc[
                    _null_valid, "same-depth correlation"
                ].median(),
                "opposite-depth correlation": _hole_pairs[
                    "opposite-depth correlation"
                ].median(),
                "null pairs": int(
                    _hole_pairs["opposite-depth correlation"].notna().sum()
                ),
                "median |lag| [ms]": np.median(
                    np.abs(_hole_pairs["same-depth lag [ms]"])
                ),
                "median down/up RMS": _hole_pairs[
                    "down/up RMS amplitude"
                ].median(),
            }
        )
    validation_summary = pd.DataFrame(_rows)
    validation_summary.round(3)
    return (validation_summary,)


@app.cell
def _(borehole_names, np, plt, validation_summary):
    _positions = np.arange(len(borehole_names))
    _width = 0.36
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _ax.bar(
        _positions - _width / 2,
        validation_summary["matched same-depth correlation"],
        width=_width,
        label="nearest depth",
        color="C0",
    )
    _ax.bar(
        _positions + _width / 2,
        validation_summary["opposite-depth correlation"],
        width=_width,
        label="opposite-depth null",
        color="0.65",
    )
    _ax.set(
        xticks=_positions,
        xticklabels=borehole_names,
        ylim=(0, 1.05),
        ylabel="median signed-waveform correlation",
        title="Co-located fiber legs usually reproduce the waveform",
    )
    _ax.legend(frameon=False)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    drift_apparent_velocity_km_s,
    hole_apparent_velocity_km_s,
    mo,
    moveout_rms_ms,
    null_median_correlation,
    null_min_depth_separation_m,
    null_pair_count,
    validation_median_abs_lag_ms,
    validation_median_amplitude_ratio,
    validation_median_correlation,
    validation_matched_median_correlation,
    validation_pair_count,
    validation_summary,
):
    _n140_correlation = validation_summary.loc[
        validation_summary["borehole"] == "N140",
        "same-depth correlation",
    ].iloc[0]
    mo.md(f"""
    Across all **{validation_pair_count}** down/up pairs, the median same-depth correlation is **{validation_median_correlation:.3f}**. On the identical **{null_pair_count}**-row subset used by the depth-separated control, the same-depth median is **{validation_matched_median_correlation:.3f}**, compared with **{null_median_correlation:.3f}** after deliberately breaking co-location by at least {null_min_depth_separation_m:g} m. The median absolute lag across all same-depth pairs is only **{validation_median_abs_lag_ms:.1f} ms**, and the median down/up RMS amplitude ratio is **{validation_median_amplitude_ratio:.2f}**. Those independent fiber sections therefore agree in timing, waveform, and amplitude closely enough to support the moveout interpretation.

    N140 is the important exception: its median same-depth correlation is only **{_n140_correlation:.3f}**. The arrival remains pickable in its envelope, but the signed waveforms on its two legs are less alike. Local coupling, orientation, or sub-wavelength heterogeneity are more plausible explanations than a clock error because both legs were recorded by the same interrogator at the same time.

    ## Scientific result

    The isolated blast front is coherent across all five north holes and advances both towards N180 and down the holes. A robust two-coordinate description yields apparent velocities of **{drift_apparent_velocity_km_s:.2f} km/s along the north-drift projection** and **{hole_apparent_velocity_km_s:.2f} km/s along the pooled hole-depth coordinate**, with **{moveout_rms_ms:.2f} ms** RMS scatter. Leave-one-hole-out fits retain the pattern but expose meaningful uncertainty. The paired legs strongly validate the local timing and amplitude response, while N140 and the non-monotonic N160-N180 amplitude pattern show why the result should remain a projected wavefront measurement rather than a claim of material velocity, attenuation, or source location.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reproducibility choices

    - The analysis window is fixed at 1.60-2.10 s relative to the file start and isolates the first charge from the later firing train.
    - Filtering and differentiation operate on the full five-second patch before selection; the moveout pick uses the Hilbert envelope and a fixed deepest-N180 reference.
    - The horizontal axis is the first principal component of the five inventory collar coordinates, oriented from N100 to N180; every channel's enriched `x,y` is projected onto that axis, and hole depth comes directly from the enriched coordinate.
    - The plane uses an unweighted Huber loss with a 1 ms transition. The leave-one-hole-out table reruns the same estimator without tuning it to each omission.
    - Leg validation uses signed strain rate, nearest physical depth, a ±5 ms search, and an opposite-depth pairing as the null.
    """)
    return


if __name__ == "__main__":
    app.run()
