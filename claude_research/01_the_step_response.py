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
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What the interrogator does to a step

    The blast leaves the near-field DAS channels sitting at a new phase, and notebook 01 offered two readings of that: permanent deformation, or an instrument pushed past what it can follow. There is a third, and this notebook makes the case for it: the offset is real, the interrogator removes it, and the way it removes it is visible in the data — a first-order high-pass filter with a corner near 10 mHz, applied to every channel alike.

    That is a claim about the instrument, so it is tested the way an instrument is tested. A filter has a step response, and the blast is a step. Five checks, one per section:

    1. every channel relaxes along the *same curve*, whatever its size;
    2. the curve has one *time constant* everywhere on a 1.7 km fiber, independent of amplitude and position;
    3. the curve is the step response of a *first-order* high-pass and not a second-order one — DASCore's own `pass_filter` draws it;
    4. the quiet-time noise has the same *corner*;
    5. the filter can be *undone*, and every decay becomes a flat step.

    None of this needs the inventory, but it makes the selections read better, so the spool is enriched as in notebook 03.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    from galileo_2026 import get_data_path, get_inventory_path

    blast = np.datetime64("2026-08-06T10:13:27")
    plt.rcParams["figure.figsize"] = (10, 6)
    return (
        blast,
        curve_fit,
        dc,
        get_data_path,
        get_inventory_path,
        np,
        pd,
        plt,
    )


@app.cell
def _(blast, dc, get_data_path, get_inventory_path, np):
    inv = dc.inventory(get_inventory_path())
    spool = dc.spool(get_data_path()).update().attach_inventory(inv).enrich()

    # The whole 22-minute low-frequency record, in strain. The files merge
    # with a tolerance; 02_the_clock_in_the_files.py shows the samples are
    # continuous and the stretched axis is the true cadence.
    lf_patch = (
        spool.select(tag="DAS_LF")
        .chunk(time=..., conflict="drop", tolerance=10)[0]
        .radians_to_strain()
    )
    seconds = dc.to_float(lf_patch.get_array("time") - blast)

    # Strain relative to each channel's mean before the blast, in microstrain.
    _before = lf_patch.select(time=(..., blast - np.timedelta64(5, "s"))).mean(
        "time"
    )
    relative = (lf_patch - _before).convert_units("µϵ")
    relative
    return inv, lf_patch, relative, seconds, spool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The record

    Two boreholes a kilometre of fiber apart, one in each drift, every channel of each. Both are drawn at the same scale.
    """)
    return


@app.cell
def _(plt, relative, seconds):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for _ax, _hole in zip(_axes, ("S180", "N180")):
        _p = relative.select(borehole=_hole)
        _ax.plot(seconds, _p.data, lw=0.7)
        _ax.axvline(0, color="k", lw=0.8)
        _ax.set_xlim(-20, 150)
        _ax.set_xlabel("seconds from the blast")
        _ax.set_title(f"{_hole}, {_p.shape[1]} channels")
    _axes[0].set_ylabel("strain relative to before [µε]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One curve

    Divide each channel by its own value three seconds after the blast, where the shaking has stopped and the decay has begun, and draw them all on top of each other. If the ground were relaxing, the near-field channels and the far ones, the grouted and the hanging, would each have their own shape.
    """)
    return


@app.cell
def _(np, plt, relative, seconds):
    _at_3s = np.argmin(np.abs(seconds - 3))
    step_size = relative.data[_at_3s]
    moved = np.abs(step_size) > 2.0
    normalised = relative.data[:, moved] / step_size[moved]
    decay_median = np.median(normalised, axis=1)

    _window = (seconds > -10) & (seconds < 120)
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.plot(
        seconds[_window], normalised[_window], color="C0", alpha=0.06, lw=0.8
    )
    _ax.plot(
        seconds[_window],
        decay_median[_window],
        "k",
        lw=1.5,
        label="median of all channels",
    )
    _ax.axhline(0, color="0.5", lw=0.5)
    _ax.set_ylim(-0.4, 1.4)
    _ax.set_xlabel("seconds from the blast")
    _ax.set_ylabel("strain / strain at +3 s")
    _ax.set_title(
        f"{moved.sum()} channels that moved more than 2 µε, each divided by its own step"
    )
    _ax.legend()
    _ax
    return decay_median, moved, normalised, step_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One curve. The spread around it is noise on the small steps; the median is the shape.

    ## 2. One number

    Fit $a\,e^{-t/\tau} + c$ to each channel separately and look at the time constants: their distribution, their dependence on the size of the step, and where they sit on the fiber.
    """)
    return


@app.cell
def _(curve_fit, moved, np, pd, relative, seconds, step_size):
    def _exponential(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    _window = (seconds > 3) & (seconds < 150)
    _t = seconds[_window]
    tau = np.full(relative.shape[1], np.nan)
    for _i in np.where(moved)[0]:
        _trace = relative.data[_window, _i]
        try:
            (_a, _tau, _c), _ = curve_fit(
                _exponential,
                _t,
                _trace,
                p0=[_trace[0], 15, 0],
                bounds=([-np.inf, 0.2, -np.inf], [np.inf, 3600, np.inf]),
            )
        except RuntimeError:
            continue
        # Keep the fits that describe the channel: residual under a tenth
        # of the step.
        if np.std(_trace - _exponential(_t, _a, _tau, _c)) < 0.1 * abs(
            _trace[0]
        ):
            tau[_i] = _tau

    fits = pd.DataFrame(
        {
            "borehole": relative.get_array("borehole"),
            "distance": relative.get_array("distance"),
            "step_uE": step_size,
            "tau_s": tau,
        }
    ).dropna()
    print(
        f"{len(fits)} channels fit; τ = {fits.tau_s.median():.2f} s, IQR {fits.tau_s.quantile(0.25):.2f} to {fits.tau_s.quantile(0.75):.2f} s"
    )
    return fits, tau


@app.cell
def _(fits, np, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _axes[0].hist(fits.tau_s, bins=np.arange(10, 22, 0.25), color="C0")
    _axes[0].set_xlabel("τ [s]")
    _axes[0].set_ylabel("channels")
    _axes[1].scatter(fits.step_uE.abs(), fits.tau_s, s=6, alpha=0.6)
    _axes[1].set_xscale("log")
    _axes[1].set_xlabel("|step| [µε]")
    _axes[1].set_ylabel("τ [s]")
    _axes[2].scatter(
        fits.distance,
        fits.tau_s,
        s=6,
        alpha=0.6,
        c=np.where(fits.borehole == "", "0.6", "C3"),
    )
    _axes[2].set_xlabel("distance along the fiber [m]  (red: in a borehole)")
    _axes[2].set_ylabel("τ [s]")
    for _ax in _axes[1:]:
        _ax.set_ylim(10, 22)
        _ax.axhline(fits.tau_s.median(), color="k", lw=0.8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A peak a fraction of a second wide at 16.2 s. Flat against amplitude over two decades of step size; flat along 1.7 km of fiber, in the boreholes and out of them. On the ground, the same number would appear in the map, and it does: `map_fiber` paints each channel's τ where the channel is.
    """)
    return


@app.cell
def _(np, relative, tau):
    _tau_patch = relative.select(time=0, samples=True).squeeze().new(data=tau)
    _ax = _tau_patch.viz.map_fiber(
        x="x", y="y", color=np.clip(tau, 14, 18), cmap="viridis"
    )
    _ax.set_title("time constant of the relaxation, per channel [s]")
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Which filter

    An exponential relaxation is what a first-order high-pass filter does to a step: the output jumps with the input and then sinks back to zero as $e^{-t/\tau}$, with a corner frequency of $f_c = 1/(2\pi\tau)$. A second-order filter does something visibly different — its step response overshoots through zero before settling. DASCore's `pass_filter` builds both. Feed each a unit step and lay the results on the median curve from section 1.
    """)
    return


@app.cell
def _(dc, decay_median, fits, np, plt, seconds):
    tau_fit = fits.tau_s.median()
    corner_hz = 1 / (2 * np.pi * tau_fit)

    # A unit step at t = 0 on the record's own sampling, as a one-channel patch.
    _t = np.arange(-20, 150, 0.2)
    step_patch = dc.Patch(
        data=np.where(_t >= 0, 1.0, 0.0)[:, None],
        coords={
            "time": np.datetime64("2026-08-06T10:13:27")
            + dc.to_timedelta64(_t),
            "distance": np.array([0.0]),
        },
        dims=("time", "distance"),
    )
    # One-sided corners make a high-pass; zerophase=False is what a live
    # instrument does, since it cannot see the future.
    _at_3s = np.argmin(np.abs(_t - 3))
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _window = (seconds > -10) & (seconds < 120)
    _ax.plot(
        seconds[_window],
        decay_median[_window],
        "k",
        lw=2.5,
        alpha=0.5,
        label="the data: median normalised decay",
    )
    for _corners, _color in ((1, "C3"), (2, "C2")):
        _response = step_patch.pass_filter(
            time=(corner_hz, None), corners=_corners, zerophase=False
        ).data[:, 0]
        _response = _response / _response[_at_3s]
        _misfit = np.sqrt(
            np.mean(
                (
                    np.interp(
                        seconds[(seconds > 3) & (seconds < 120)], _t, _response
                    )
                    - decay_median[(seconds > 3) & (seconds < 120)]
                )
                ** 2
            )
        )
        _ax.plot(
            _t,
            _response,
            color=_color,
            lw=1.2,
            label=f"pass_filter, corners={_corners}, {corner_hz * 1e3:.1f} mHz high-pass: rms misfit {_misfit:.3f}",
        )
    _ax.axhline(0, color="0.5", lw=0.5)
    _ax.set_ylim(-0.4, 1.4)
    _ax.set_xlabel("seconds from the step")
    _ax.set_ylabel("response / response at +3 s")
    _ax.legend()
    _ax
    return corner_hz, step_patch, tau_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first-order response lies on the data to a few thousandths; the second-order one is nowhere near. The interrogator's output is high-passed once, with a single pole near 10 mHz, before it is written. That is a sensible thing for a phase-measuring instrument to do — its zero is arbitrary and it drifts — and it is exactly what makes the five-second file misleading: a step of any size looks permanent on a window a hundred times shorter than τ.

    ## 4. The same corner in the noise

    A filter shapes everything that passes through it, not only the blast. The fiber hanging in the drifts drifts in phase, more at low frequency than high — a red spectrum. A high-pass at 10 mHz flattens that spectrum below its corner. Fourteen quiet minutes after the decay is over resolve frequencies down to a millihertz.
    """)
    return


@app.cell
def _(blast, corner_hz, lf_patch, np, plt):
    _quiet = lf_patch.select(time=(blast + np.timedelta64(240, "s"), ...))
    _spectrum = _quiet.detrend("time").dft("time", real=True).abs()
    _frequency = _spectrum.get_array("ft_time")
    _in_hole = _spectrum.get_array("borehole") != ""

    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _mask, _label in (
        (~_in_hole, "hanging fiber"),
        (_in_hole, "grouted fiber"),
    ):
        _ax.loglog(
            _frequency[1:],
            _spectrum.data[1:, _mask].mean(axis=1),
            lw=1,
            label=f"{_label}, mean of {_mask.sum()} channels",
        )
    _ax.axvline(
        corner_hz,
        color="k",
        ls="--",
        label=f"{corner_hz * 1e3:.1f} mHz from the step response",
    )
    _ax.set_xlabel("frequency [Hz]")
    _ax.set_ylabel("amplitude spectrum [µε·s]")
    _ax.set_title("14 quiet minutes after the blast")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The hanging fiber's spectrum climbs towards low frequency and then stops climbing, at the corner the step response predicted, from data that contain no step at all. The grouted fiber is a hundred times quieter and does the same. Two independent measurements of one instrument constant.

    ## 5. Undo it

    A first-order high-pass is invertible. If $y$ is what the instrument wrote and $x$ is what the fiber did, then $x = y + \frac{1}{\tau}\int y\,dt$: add back, at each moment, what the filter has bled away so far. Apply that to the record — `integrate` along time does the running sum — and the decays should become steps.
    """)
    return


@app.cell
def _(dc, plt, relative, seconds, tau_fit):
    # τ is a time, and the integral is strain times time; leaving the seconds off
    # is a mistake the units refuse to let through.
    restored = relative + relative.integrate("time", definite=False) / (
        tau_fit * dc.get_unit("s")
    )

    _fig, _axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    for _ax, _patch, _title in (
        (_axes[0], relative, "N180 as recorded"),
        (
            _axes[1],
            restored,
            f"N180 with the {tau_fit:.1f} s high-pass undone",
        ),
    ):
        _ax.plot(seconds, _patch.select(borehole="N180").data, lw=0.7)
        _ax.axvline(0, color="k", lw=0.8)
        _ax.set_title(_title, loc="left")
        _ax.set_ylabel("µε")
    _axes[1].set_xlim(-60, 600)
    _axes[1].set_xlabel("seconds from the blast")
    _fig
    return (restored,)


@app.cell
def _(np, restored, seconds):
    _n180 = restored.select(borehole="N180").data
    _late = (seconds > 60) & (seconds < 600)
    _drift = np.polyfit(seconds[_late], _n180[_late], 1)[0] * 100
    print(
        f"restored N180 offsets at +5 min: {np.abs(_n180[np.argmin(np.abs(seconds - 300))]).max():.1f} µε at most, "
        f"drifting {np.abs(_drift).max():.2f} µε per 100 s at worst"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every decay is a step again, and the steps hold for the ten minutes the record allows, which is what a permanent offset does. This is the instrument's output with the instrument's filter removed; it is not, on its own, ground deformation, and the restored offsets themselves say so. Look at their signs along a hole.
    """)
    return


@app.cell
def _(np, pd, restored, seconds):
    _at_5min = np.argmin(np.abs(seconds - 300))
    _offset = restored.data[_at_5min]
    _borehole = restored.get_array("borehole")
    _rows = []
    for _hole in sorted(set(_borehole) - {""}):
        _v = _offset[_borehole == _hole]
        _big = np.abs(_v) > 5
        _pairs = _big[:-1] & _big[1:]
        _rows.append(
            {
                "borehole": _hole,
                "largest |offset| [µε]": np.abs(_v).max(),
                "mean offset [µε]": _v.mean(),
                "adjacent pairs > 5 µε": int(_pairs.sum()),
                "of which opposite sign": int(
                    (np.sign(_v[:-1]) != np.sign(_v[1:]))[_pairs].sum()
                ),
            }
        )
    pd.DataFrame(_rows).set_index("borehole").round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adjacent channels, 1.6 m apart, can carry offsets of twenty microstrain with opposite signs, and every hole's offsets average to about zero while single channels in it sit at ten or twenty. Rock does not deform that way over a metre and a half; a phase measurement driven past its linear range for a tenth of a second, channel by channel, does. What the fiber saw at the collar of a borehole forty metres from a blast is a separate question from what the ground did, and only an instrument that measures strain absolutely can answer the second one. The BOTDR does: comparing the median of the three hours after the blast against the three hours before, channel by channel, no borehole moves above its 12 to 19 µε noise floor. Whatever the DAS was doing, the grouted rock ended the day where it started, to the resolution available.

    Two cautions on the inversion. The integral accumulates noise, so the restored record wanders on a scale of minutes and cannot be trusted for hours. And it is only right for a first-order filter with this τ: section 3 is what licenses it.

    ### Exercise

    1. `pass_filter` has a `zerophase` argument. Rerun section 3 with `zerophase=True` and explain what happens to the response before $t = 0$, and why no instrument can do that.
    2. Section 2 used channels that moved more than 2 µε. Lower the threshold to 0.5 µε and watch the τ histogram. What broadens it, and is the peak still at 16.2 s?
    3. Section 4 used the quiet minutes *after* the blast. Use the three and a half minutes before it instead. What is the lowest frequency you can resolve, and is the corner still visible?
    4. Build the BOTDR test yourself: merge `tag="DSS"`, subtract the median along distance as notebook 02 does, and difference the median of three hours after the blast against three hours before, per borehole. Use the median absolute deviation of the pre-blast sweeps as the noise. Does anything clear it? Then compare what you find at N160 against `restored` at the same `hole_depth`, and say what it would take to decide between them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - The post-blast offsets in the DAS phase decay as one exponential with τ = 16.2 s on every channel, independent of amplitude and position: an instrument high-pass with a single pole near 10 mHz, not the ground.
    - The test for an instrument effect is uniformity: the same shape, the same number, everywhere, whatever the signal did.
    - `pass_filter(time=(f_c, None), corners=1, zerophase=False)` is that filter; its step response matches the data to a few thousandths, and a second-order one does not.
    - The corner appears in the quiet-time noise spectrum as well as in the step response.
    - A first-order high-pass can be undone with `integrate`: $x = y + \int y\,dt / \tau$. The restored offsets are what the fiber saw; whether the ground moved is a question for an absolute-strain instrument.
    """)
    return


if __name__ == "__main__":
    app.run()
