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
    # The Patch

    *This is the solutions copy: each exercise is followed by a worked answer.*

    This notebook addresses learning objective 1. We will use DASCore's `Patch` to explore a blast recorded on the fiber network.
    """)
    return


@app.cell
def _():

    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from galileo_2026 import get_data_path

    # Define optical distances for boreholes of interest on DAS recording.
    borehole_distances = {
        "N180": (1577.9, 1634.7),
        "N160": (1486.0, 1544.4),
        "N140": (1399.0, 1459.4),
        "N120": (1306.0, 1365.0),
        "N100": (1216.3, 1274.2),
    }

    # Zoomed in times around the blast
    blast_time_zoom_0 = ("2026-08-06T10:13:26.5", "2026-08-06T10:13:28.7")
    blast_time_zoom_1 = ("2026-08-06T10:13:26.6", "2026-08-06T10:13:27")
    blast_time_zoom_2 = ("2026-08-06T10:13:26.79", "2026-08-06T10:13:26.86")

    # Set default mpl figuresize to better suit the notebook width

    plt.rcParams["figure.figsize"] = (10, 6)  # width, height in inches
    return (
        blast_time_zoom_0,
        blast_time_zoom_1,
        blast_time_zoom_2,
        borehole_distances,
        dc,
        get_data_path,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    To read data, we use the `spool` function. We will learn more about this in the next notebook, for now we focus on the `Patch` that comes out of it.
    """)
    return


@app.cell
def _(dc, get_data_path):
    # First get a path to our data directory.
    _data_path = get_data_path()

    # Next create the spool and make sure it is up-to-date, and select our DAS file.
    spool = dc.spool(_data_path).update().select(tag="DAS")
    return (spool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Patch and its parts

    Patches are extracted (loaded) only when you ask for them. This is done primarily through item access or iteration as in the following examples:
    """)
    return


@app.cell
def _(spool):
    patch = spool[0]
    return (patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can look at a patch's summary by printing it.
    """)
    return


@app.cell
def _(patch):
    patch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Accessing the parts of the patch are simply done as follows:
    """)
    return


@app.cell
def _(patch):
    print(patch.dims, patch.data.shape)

    # coordinates which label the axes
    print(patch.coords)

    # attrs is a model, so it prints as a block but reads like an object
    print(patch.attrs.data_type, patch.attrs.data_units)

    # the raw array is always available, unlabelled
    patch.data[:, :5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting and Selecting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As with all other DASCore objects, the Patch has a namespace dedicated to visualizations called `viz`.  Using this we can see the blast clearly
    """)
    return


@app.cell
def _(patch):
    patch.viz.waterfall()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `select` is used to trim patches in either dimension.
    """)
    return


@app.cell
def _(blast_time_zoom_2, patch):
    # Select part of time
    zoomed_in_patch = patch.select(time=blast_time_zoom_2)

    zoomed_in_patch.viz.waterfall()
    return (zoomed_in_patch,)


@app.cell
def _(borehole_distances, zoomed_in_patch):
    # Select part of space
    n180_patch = zoomed_in_patch.select(distance=borehole_distances["N180"])

    n180_patch.viz.waterfall()
    return (n180_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The wiggle plot is also helpful, in this case it accentuates the post-blast offsets.
    """)
    return


@app.cell
def _(n180_patch):
    # The wiggle plot is also helpful
    n180_patch.viz.wiggle(scale=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observation**

    Notice post-blast static offset. The blast might have caused permanent deformation to the ground, but the dynamic response of the instrument has very likely been exceeded as well with these near-field recordings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtering

    A filter is the quickest way to see past the offsets. `pass_filter` keeps a band of frequencies along one dimension, with the corners in the inverse of that dimension's units, so `time=(5, 200)` is 5 to 200 Hz. The offset is at zero frequency, so it goes, and the arrivals stay.
    """)
    return


@app.cell
def _(blast_time_zoom_0, borehole_distances, patch, plt):
    _n180_blast = patch.select(
        distance=borehole_distances["N180"], time=blast_time_zoom_0
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))
    _n180_blast.viz.wiggle(ax=_axes[0], scale=0.5)
    _n180_blast.pass_filter(time=(5, 200)).viz.wiggle(ax=_axes[1], scale=0.5)
    _axes[0].set_title("raw phase")
    _axes[1].set_title("5 to 200 Hz")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise**:
    1) Assuming the borehole is symmetric, select the downgoing leg of N180 and zoom into the blast front more in the same select call.
    2) Plot the result with a different colormap.
    3) `decimate(time=4)` the filtered patch and plot it again. What changed, and what did not?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Solution**
    """)
    return


@app.cell
def _(borehole_distances, patch):
    # 1) Both trims in one select: the downgoing leg is the first half of the
    # hole's span, and the front is within 40 ms of the first charge.
    _d1, _d2 = borehole_distances["N180"]
    front_patch = patch.select(
        distance=(_d1, (_d1 + _d2) / 2),
        time=("2026-08-06T10:13:26.80", "2026-08-06T10:13:26.84"),
    )
    # 2) Any matplotlib colormap name works.
    front_patch.viz.waterfall(cmap="viridis")
    return (front_patch,)


@app.cell
def _(blast_time_zoom_0, borehole_distances, patch):
    # 3) decimate low-passes and then keeps every fourth sample, so 2 kHz
    # becomes 500 Hz. The picture is the same: the 5 to 200 Hz band is well
    # inside the new 250 Hz Nyquist. Only the sample count changed.
    _filtered = patch.select(
        distance=borehole_distances["N180"], time=blast_time_zoom_0
    ).pass_filter(time=(5, 200))
    decimated = _filtered.decimate(time=4)
    print(_filtered.shape, "->", decimated.shape)
    decimated.viz.wiggle(scale=0.5)
    return (decimated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Processing and Transformations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DASCore provides various transformation and processing routines as patch methods. These are typically composed through method chaining.

    Let's see if we can

    1) determine the apparent velocity from N180, which sits approximately above the blast
    2) count how many charges were in the blast

    It will be simpler to take a derivative, which removes the large low-frequency offsets and turns strain into strain rate.
    """)
    return


@app.cell
def _(blast_time_zoom_2, borehole_distances, patch):
    from dascore.units import ms

    _d1, _d2 = borehole_distances["N180"]

    # The borehole is a U, so the fiber runs down one leg and back up the other.
    # Only a single leg is a straight line in distance; a window spanning the
    # bend would see each arrival as a V rather than a dipping front.
    _downgoing = (_d1, (_d1 + _d2) / 2)

    processed = (
        patch.select(distance=_downgoing)
        .radians_to_strain()  # Convert from phase to strain
        .differentiate(
            "time"
        )  # differentiate along time axis to get strain rate
    )

    # The envelope collapses each wiggle to its amplitude, which makes the blast
    # front easy to see.
    processed.envelope("time").select(time=blast_time_zoom_2).viz.wiggle()
    return ms, processed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Counting the charges

    Averaging the envelope over the leg gives one trace for the whole hole, in which each charge should be a peak. DASCore hands its arrays to NumPy and SciPy whenever it has nothing of its own to offer: `find_peaks` counts them.
    """)
    return


@app.cell
def _(blast_time_zoom_0, dc, plt, processed):
    from scipy.signal import find_peaks

    _envelope = (
        processed.envelope("time")
        .select(time=blast_time_zoom_0)
        .mean("distance")
        .squeeze()
    )
    _time = _envelope.get_array("time")
    _seconds = dc.to_float(_time - _time[0])
    _step = dc.to_float(_envelope.get_coord("time").step)
    # A charge is a peak at least a quarter the height of the largest one and
    # at least 30 ms after the previous.
    _peaks, _ = find_peaks(
        _envelope.data,
        height=0.25 * _envelope.data.max(),
        distance=int(0.03 / _step),
    )
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_seconds, _envelope.data)
    _ax.plot(_seconds[_peaks], _envelope.data[_peaks], "rv")
    _ax.set_xlabel("seconds from window start")
    _ax.set_ylabel("mean envelope of strain rate")
    _ax.set_title(f"{len(_peaks)} peaks")
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One peak on its own, then a train of them 0.6 s later, spaced 50 and 100 ms apart, which is consistent with delay detonators. They are candidate charges rather than a firing record: a peak can also be a reverberation, and whether the count is eleven or twelve depends on the threshold. Try changing `height`.

    ### The spectrum

    `dft` transforms along a dimension and replaces it with a frequency coordinate, `ft_time`. The amplitude spectrum averaged over the leg shows where the blast's energy is, and a roll-off towards the 1 kHz Nyquist frequency that is consistent with the instrument's response, though one blast cannot separate the instrument from the source and the path.
    """)
    return


@app.cell
def _(blast_time_zoom_0, plt, processed):
    _spectrum = (
        processed.select(time=blast_time_zoom_0)
        .dft("time", real=True)
        .abs()
        .mean("distance")
        .squeeze()
    )
    _fig, _ax = plt.subplots(figsize=(10, 4))
    # Skip the zero-frequency bin, which a log axis cannot show.
    _ax.loglog(_spectrum.get_array("ft_time")[1:], _spectrum.data[1:])
    _ax.set_xlabel("frequency [Hz]")
    _ax.set_ylabel("amplitude")
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apparent velocity

    The blast front sweeps along the leg, so each channel records it at a slightly different time. Cross-correlating every channel against the first turns that shift into a lag we can measure, and the slope of lag against distance is one over the apparent velocity.
    """)
    return


@app.cell
def _(blast_time_zoom_1, ms, processed):
    cc = (
        processed.select(
            time=blast_time_zoom_1
        )  # Zoom in around the first charge
        .taper(time=10 * ms)  # taper the window we are about to transform
        .correlate(
            distance=0, samples=True
        )  # correlate all channels against the first channel
        .squeeze()  # drop the length-1 source dimension
        .select(
            lag_time=(-0.02, 0.02)
        )  # Select lag times around reasonable velocities
    )

    cc.viz.waterfall()
    return (cc,)


@app.cell
def _(cc, dc, np):
    # Get the values for the distance axis.
    distance = cc.get_array("distance")

    # Get the values for the lag times.
    _lag = dc.to_float(cc.get_array("lag_time"))

    # Find the lagtime values
    _peak_indices = cc.data.argmax(axis=cc.get_axis("lag_time"))
    _picks = _lag[_peak_indices]
    slope, intercept = np.polyfit(distance, _picks, 1)

    # The slope is negative: the front reaches the deep end of the leg first and
    # travels back up it, so the absolute value keeps speed and drops direction.
    print(f"apparent velocity: {abs(1 / slope):.0f} m/s")
    return distance, intercept, slope


@app.cell
def _(cc, distance, intercept, slope):
    # Notice here we can return the matplotlib axis and is it for additional plotting.
    _ax = cc.viz.waterfall(show=False, cmap="seismic", scale=0.3)
    _ = _ax.plot(
        distance, slope * distance + intercept, "--", color="0.5", lw=4
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise**:
    1) `processed` is the downgoing leg only. Build the same strain-rate patch for the upgoing leg, from the midpoint of `borehole_distances["N180"]` to its end, repeat the correlation, and check that the two velocities agree. Which way does the front travel in that leg?
    2) `viz.spectrogram()` draws the spectrum against time for a single channel. Pick one from `processed` with `select(distance=(10, 11), samples=True).squeeze()` and try it, or hand it the whole leg and it averages the spectra.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Solution**
    """)
    return


@app.cell
def _(blast_time_zoom_1, borehole_distances, dc, ms, np, patch):
    # 1) The same chain as above, for the second half of the hole.
    _d1, _d2 = borehole_distances["N180"]
    upgoing = (
        patch.select(distance=((_d1 + _d2) / 2, _d2))
        .radians_to_strain()
        .differentiate("time")
    )
    _cc = (
        upgoing.select(time=blast_time_zoom_1)
        .taper(time=10 * ms)
        .correlate(distance=0, samples=True)
        .squeeze()
        .select(lag_time=(-0.02, 0.02))
    )
    _distance = _cc.get_array("distance")
    _lag = dc.to_float(_cc.get_array("lag_time"))
    _picks = _lag[_cc.data.argmax(axis=_cc.get_axis("lag_time"))]
    _slope, _ = np.polyfit(_distance, _picks, 1)
    # The slope is positive this time. In the upgoing leg, distance along the
    # fiber increases towards the surface, and so does arrival time: the front
    # still comes from below, only the fiber's direction has changed. The two
    # velocities agree to within a couple of percent.
    print(f"upgoing leg apparent velocity: {abs(1 / _slope):.0f} m/s")
    return (upgoing,)


@app.cell
def _(blast_time_zoom_0, processed):
    # 2) One channel, squeezed to a single dimension.
    processed.select(distance=(10, 11), samples=True).squeeze().select(
        time=blast_time_zoom_0
    ).viz.spectrogram()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - `Patch`s are a contiguous chunk of data and metadata, specifically composed of data, coords, and attrs.
    - `Patch` has many processing/transformation methods that can be chained together, as well as visualizations that return matplotlib objects
    - `select`, `taper`, `differentiate`, `envelope` and `dft` take the dimension they work along by name; `select` and `taper` read values in that dimension's units (or in samples, with `samples=True`), and `pass_filter` reads its corners in the inverse of them
    - `correlate` names the dimension and the master channel along it, and adds a `lag_time` dimension to the result
    - `get_array`, `get_coord` and `.data` hand the result to NumPy and SciPy where DASCore stops

    Next, `02_the_spool.py` picks up the archive these patches came out of.
    """)
    return


if __name__ == "__main__":
    app.run()
