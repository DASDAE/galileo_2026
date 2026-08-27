# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "galileo-2026 @ git+https://github.com/DASDAE/galileo_2026@main",
#     "dascore @ git+https://github.com/DASDAE/dascore@dev",
#     "marimo>=0.24",
#     "matplotlib>=3.10",
#     "numba",
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
    # The clock in the files

    Notebook 02 found that the low-frequency DAS files do not quite abut: `get_gaps` reports six breaks of 0.4 to 1.3 s in a 22-minute archive, and merging them took `tolerance=10`, with DASCore warning that the sampling of the result had been altered to make it fit. That is an uncomfortable thing to do to data. Were those seconds really missing, in which case the merge stretched twenty-two minutes of record over the holes, or were they never there, in which case the merge quietly did the right thing?

    The data can answer. This notebook is a small piece of detective work on timestamps, and it ends with a number every user of this archive should know: how far to trust the time axis of the 5 Hz product.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from galileo_2026 import get_data_path

    plt.rcParams["figure.figsize"] = (10, 6)
    return dc, get_data_path, np, pd, plt


@app.cell
def _(dc, get_data_path):
    lf_spool = (
        dc.spool(get_data_path())
        .update()
        .select(tag="DAS_LF")
        .sort("time_min")
    )
    lf_files = list(lf_spool)
    print(
        f"{len(lf_files)} files of {lf_files[0].shape[0]} samples at {lf_files[0].get_coord('time').step}"
    )
    return lf_files, lf_spool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The gaps as the index sees them

    Each file is 300 samples at a nominal 0.2 s, so one file should end 0.2 s before the next begins. Where the index sees more than that, `get_gaps` reports it.
    """)
    return


@app.cell
def _(lf_spool):
    gaps = lf_spool.get_gaps()[["time_min", "time_max"]]
    gaps["length_s"] = (
        (gaps.time_max - gaps.time_min).dt.total_seconds().round(3)
    )
    gaps
    return (gaps,)


@app.cell
def _(lf_spool):
    lf_spool.viz.coverage()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two stories

    Two things could produce that table.

    1. **The timestamps are right and samples are missing.** The interrogator's decimator dropped a second of output now and then, and the files honestly say so. Merging with a tolerance then stretches the samples on either side of a hole over it, and every time in the merged patch after the first gap is wrong by up to four seconds.
    2. **The samples are continuous and the timestamps are wrong.** Each file is stamped from a clock, the samples within it are counted at the nominal rate, and every so often the clock and the count disagree by a second. Nothing is missing, and a merge that stretches the axis recovers the true cadence.

    They predict different things at a file boundary. The fiber hanging in the drift between the holes carries slowly drifting phase, tens of radians a second on the worst channels, smooth over seconds. Fit a curve to the last few seconds of one file and there are two places the first sample of the next file can land: 0.2 s later, or as far later as its own timestamp claims. Only one will sit on the curve.
    """)
    return


@app.cell
def _(lf_files, np):
    def fit_boundary(before, after, n_fit=30, n_show=10):
        """Quadratic fit to the end of `before`, and where `after` begins."""
        seconds = np.arange(-n_fit + 1, 1) * 0.2
        end = before.data[-n_fit:].astype(float)
        coef = np.polyfit(seconds, end, 2)  # one column per channel
        fitted = (
            coef[0] * seconds[:, None] ** 2
            + coef[1] * seconds[:, None]
            + coef[2]
        )
        residual = np.std(end - fitted, axis=0)
        # Channels that drift smoothly: a small residual, and a slope that
        # stands well above it, so the two placements are told apart.
        smooth = (
            (residual < 0.15)
            & (np.abs(coef[1]) > 3 * residual)
            & (np.abs(coef[1]) > 0.5)
        )
        gap = (
            after.get_array("time")[0] - before.get_array("time")[-1]
        ) / np.timedelta64(1, "s")

        def predict(dt):
            return coef[0] * dt**2 + coef[1] * dt + coef[2]

        return {
            "seconds": seconds,
            "end": end,
            "start": after.data[:n_show].astype(float),
            "coef": coef,
            "smooth": smooth,
            "gap": gap,
            "predict": predict,
        }

    return (fit_boundary,)


@app.cell
def _(fit_boundary, lf_files, np, plt):
    # The boundary after the blast, where the index reports 1.018 s.
    boundary = fit_boundary(lf_files[3], lf_files[4])
    _slope = boundary["coef"][1]
    _channel = np.where(boundary["smooth"])[0][
        np.argmax(np.abs(_slope[boundary["smooth"]]))
    ]
    _ahead = np.arange(boundary["start"].shape[0]) * 0.2

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.plot(
        boundary["seconds"],
        boundary["end"][:, _channel],
        "o-",
        ms=3,
        label="last six seconds of file 4",
    )
    _curve = np.linspace(-6, 3, 100)
    _ax.plot(
        _curve,
        np.polyval(boundary["coef"][:, _channel], _curve),
        "k--",
        lw=1,
        label="quadratic fit, extrapolated",
    )
    _ax.plot(
        0.2 + _ahead,
        boundary["start"][:, _channel],
        "s-",
        ms=3,
        label="file 5 placed 0.2 s after file 4",
    )
    _ax.plot(
        boundary["gap"] + _ahead,
        boundary["start"][:, _channel],
        "^-",
        ms=3,
        label=f"file 5 at its own timestamps, {boundary['gap']:.3f} s after",
    )
    _ax.set_xlabel("seconds from the last sample of file 4")
    _ax.set_ylabel("phase [rad]")
    _ax.set_title(
        f"channel at {lf_files[3].get_array('distance')[_channel]:.0f} m, drifting {_slope[_channel]:.1f} rad/s"
    )
    _ax.legend()
    _ax
    return (boundary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The squares continue the curve. The triangles, the same samples at the times the file gives them, do not: they sit a second to the right of where the phase was heading. On this channel, story 2 is what happened.

    One channel could be a coincidence, so do it for every drifting channel at every boundary, and report the median miss of each placement.
    """)
    return


@app.cell
def _(fit_boundary, lf_files, np, pd):
    _rows = []
    for _k in range(len(lf_files) - 1):
        _b = fit_boundary(lf_files[_k], lf_files[_k + 1])
        _first = _b["start"][0][_b["smooth"]]
        _rows.append(
            {
                "boundary": f"{_k + 1} → {_k + 2}",
                "index gap [s]": round(_b["gap"], 3),
                "channels used": int(_b["smooth"].sum()),
                "miss if contiguous [rad]": np.median(
                    np.abs(_first - _b["predict"](0.2)[_b["smooth"]])
                ),
                "miss at own timestamp [rad]": np.median(
                    np.abs(_first - _b["predict"](_b["gap"])[_b["smooth"]])
                ),
            }
        )
    boundaries = pd.DataFrame(_rows).set_index("boundary").round(3)
    boundaries[boundaries["index gap [s]"] > 0.3]
    return (boundaries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At all six boundaries the contiguous placement misses by about what it misses at the boundaries with no gap at all (run the cell without the filter to see those), and the placement at the file's own timestamp misses by several times more. No samples are missing. The seconds in `get_gaps` are seconds the clock jumped, not seconds the fiber went unrecorded.

    ## The timestamps themselves say so

    If a file's start time were a count of samples from the previous file, its sub-second part would be the previous one plus a multiple of 0.2 s. If it is a fresh reading of a clock, it can be anything. Look at the fractions.
    """)
    return


@app.cell
def _(lf_spool, np, pd):
    _contents = lf_spool.get_contents()
    _starts = _contents.time_min.values
    _fraction = (_starts - _starts.astype("datetime64[s]")) / np.timedelta64(
        1, "s"
    )
    _spacing = np.diff(_starts) / np.timedelta64(1, "s")
    starts = pd.DataFrame(
        {
            "file": _contents.source_path.str.split("/")
            .str[-1]
            .str.split("_")
            .str[3],
            "start": _starts,
            "fraction [s]": _fraction.round(6),
            "since previous [s]": np.r_[np.nan, _spacing].round(3),
            "fraction mod 0.2": (_fraction % 0.2).round(4),
        }
    )
    starts
    return (starts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fraction holds steady for three or four files, then changes, and never by a multiple of 0.2 s: `fraction mod 0.2` takes a new value at every jump. Between jumps the files are stamped by counting; at a jump the interrogator reads its clock again and finds that the count has fallen behind. The name of each file, which carries the wall-clock second, agrees.

    ## Which clock is right?

    The 2 kHz record is the same interrogator writing the same minute, timestamped independently, and the blast is the sharpest event in both. Notebook 01 put its onset at N180 at 10:13:26.8 in the 2 kHz file. Find the same onset in the 5 Hz file that contains it, the fourth one, whose end is where the 1.018 s jump happens.
    """)
    return


@app.cell
def _(dc, get_data_path, lf_files, np):
    hf_patch = dc.spool(get_data_path()).update().select(tag="DAS")[0]
    _channels = (1577.9, 1634.7)  # N180 on the DAS fiber, from notebook 01

    def first_jump(patch, threshold):
        """The time of the first sample-to-sample phase jump above threshold."""
        data = patch.select(distance=_channels).data
        jump = np.abs(np.diff(data, axis=0)).max(axis=1)
        return patch.get_array("time")[1:][np.argmax(jump > threshold)]

    hf_onset = first_jump(hf_patch, 0.5)
    lf_onset = first_jump(lf_files[3], 2.0)
    print(f"2 kHz file: {hf_onset}\n5 Hz file 4: {lf_onset}")
    print(
        f"the 5 Hz file is {(hf_onset - lf_onset) / np.timedelta64(1, 's'):.2f} s early"
    )
    return hf_onset, hf_patch, lf_onset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A second early, at the end of the file whose successor is stamped 1.018 s later than a count would put it. The count had fallen a second behind the clock by then, and the next file caught up. Within a file the 5 Hz timestamps are not the interrogator's clock; they are that clock at the file's start plus 0.2 s per sample, and 0.2 s is not quite what the decimator delivers.

    ## What this means for the archive

    Three practical things.

    **Every time in the 5 Hz files is good to about a second**, and worse late in a run between resyncs. Anything needing better timing than that comes from the 2 kHz record.

    **The true cadence is not 5 Hz.** Twenty-two files of 300 samples, continuous, span the 1324 s between the first and last timestamp, which is what the merge with a tolerance recovers:
    """)
    return


@app.cell
def _(dc, lf_spool):
    lf_merged = lf_spool.chunk(time=..., conflict="drop", tolerance=10)[0]
    _step = dc.to_float(lf_merged.get_coord("time").step)
    _span = dc.to_float(
        lf_merged.get_coord("time").max() - lf_merged.get_coord("time").min()
    )
    print(
        f"{lf_merged.shape[0]} samples over {_span:.1f} s: "
        f"step {_step:.6f} s, {1 / _step:.4f} Hz"
    )
    return (lf_merged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The warning DASCore raised in notebook 02, that the sampling of the result was altered, was the truth: the sampling in the files was the thing that was wrong, and the stretched axis is the honest one, to the extent that the file starts were themselves honest.

    **`get_gaps` reports what the index sees, not what happened.** That is the right thing for it to do; the index cannot know. Whether a gap is a hole or a jump is a question for the data, and the ten minutes above are how to ask it.

    ## The other instrument's clock

    The Brillouin sweeps carry a `sample_span` coordinate, one value per sweep: the time the interrogator took to acquire it. Compare that with the spacing between sweeps.
    """)
    return


@app.cell
def _(dc, get_data_path):
    dss_patch = (
        dc.spool(get_data_path())
        .update()
        .select(tag="DSS")
        .chunk(time=None, conflict="drop")[0]
    )
    _span = dss_patch.get_coord("sample_span")
    _step = dc.to_float(dss_patch.get_coord("time").step)
    print(
        f"{dss_patch.shape[0]} sweeps every {_step:.1f} s, each acquired over "
        f"{dc.to_float(_span.min()):.2f} to {dc.to_float(_span.max()):.2f} s"
    )
    return (dss_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten minutes of averaging and seven and a half seconds of housekeeping, sweep after sweep, all day; a sweep's timestamp is its start, and the strain it reports is the mean over the ten minutes that follow. A step that fell inside a sweep, the blast at 10:13:27 for one, is diluted into the sweep that started at 10:11:44 in proportion to how much of the sweep it covered. Two instruments, two different things a timestamp means.

    ### Exercise

    1. Rerun the boundary table without the `> 0.3` filter. At boundaries the index calls contiguous, how large is the miss, and what does that say about the precision of the test?
    2. Suppose you needed the 5 Hz record on the true clock. The jumps say how far behind the count had fallen at each resync. Sketch how you would use them to correct the times inside each file, and what you would assume about the drift between resyncs.
    3. The `fraction mod 0.2` column is constant between jumps. What does that imply about how the interrogator decides to start a new file?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - `get_gaps` reports gaps in timestamps. Whether the data are missing or the clock jumped is a separate question, and the data can answer it: continuous phase across a boundary means nothing was lost.
    - In this archive the 5 Hz files are stamped by counting samples from a clock read at each file start; the count runs slow and is resynced by a second or so every few files. Times inside a file are good to about a second.
    - `chunk(time=..., tolerance=...)` stretches a merged axis to fit the timestamps it was given, and warns. Here that stretch recovers the true cadence, 4.984 Hz, and is the right treatment; had the samples really been missing, it would have been wrong by up to four seconds.
    - The 2 kHz file is an independent timestamp from the same interrogator, and the blast onset ties the two records together to a sample.
    - A timestamp is a convention: the BOTDR's is the start of a ten-minute average, and `sample_span` says how long the average ran.
    """)
    return


if __name__ == "__main__":
    app.run()
