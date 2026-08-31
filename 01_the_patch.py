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
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Patch

    This notebook addresses learning objective 1. We will use DASCore's `Patch` to explore a blast recorded on the fiber network.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np

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
    To read data, we use the `spool` function. We will learn more about this in the next notebook; for now we focus on the `Patch` that comes out of it.
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
    # Get the first patch in the spool and convert from phase angle to strain. 
    patch = spool[0].radians_to_strain()
    return (patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The patch is based on [xarray's](https://docs.xarray.dev/en/stable/) [data array](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html). It has several parts:

    - dims : a tuple[str] of dimension names
    - data : the array
    - coords : a dict-like container for coordinates (which label each axis)
    - attrs : a dict-like container for scalar metadata

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
    The parts of the patch are accessed as follows:
    """)
    return


@app.cell
def _(patch):
    _dims = patch.dims
    _coords = patch.coords
    _attrs = patch.attrs
    _data = patch.data
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
    As with all other DASCore objects, the Patch has a namespace dedicated to visualizations called `viz`. Using this we can see the blast clearly.
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
    return


@app.cell
def _(blast_time_zoom_2, borehole_distances, patch):
    # Select part of space
    n180_patch = patch.select(distance=borehole_distances["N180"])
    n180_patch.select(time=blast_time_zoom_2).viz.waterfall()
    return (n180_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The wiggle plot is also helpful; in this case it accentuates the post-blast offsets.
    """)
    return


@app.cell
def _(blast_time_zoom_2, n180_patch):
    # The wiggle plot is also helpful
    n180_patch.select(time=blast_time_zoom_2).viz.wiggle(scale=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Zooming out and selecting the channel at the bottom of the hole, we see all the charges clearly. Notice how each charge adds its own step to the strain record, ratcheting it up charge by charge (keep that in mind for Q1 below).
    """)
    return


@app.cell
def _(n180_patch):
    _middle_ind = len(n180_patch.get_coord("distance")) // 2
    _middle = n180_patch.select(distance=_middle_ind, samples=True)
    _middle.viz.wiggle()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Observations**

    **1) There is significant offset caused by the blast.**

    The blast might have caused permanent deformation to the ground, that is the point of blasting in a mine, but the dynamic response of the instrument could have been exceeded in these near-field recordings.

    Q1: *Is the blast offset physical or an artifact of phase-unwrapping errors?*


    **2) The strain dissipates post-blast**

    From second 28.25 or so, the background strain level begins to return to normal. This could indicate strain dissipation and [afterslip](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=afterslip+earthquake&btnG=), or perhaps it's something else?

    Q2: *Is the apparent strain dissipation physical or an artifact of the interrogator's internal processing?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtering

    Let's set aside the strain offset questions for now. We will use the other datasets to explore these. Instead, let's focus on the charge waveforms.

    A simple filter is a quick way to remove the offsets. `Patch.pass_filter` is a standard Butterworth filter. This alleviates the offset issue, but does cause some noticeable filter effects in the center of the borehole.
    """)
    return


@app.cell
def _(blast_time_zoom_0, borehole_distances, patch, plt):
    # Select the borehole and the window around the blast.
    _n180_blast = patch.select(
        distance=borehole_distances["N180"], time=blast_time_zoom_0
    )

    # Apply the pass filter.
    n180_blast_filtered = _n180_blast.pass_filter(time=(5, 200))

    # Setup and display both wiggle plots side by side.
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))
    _n180_blast.viz.wiggle(ax=_axes[0], scale=0.5)
    n180_blast_filtered.viz.wiggle(ax=_axes[1], scale=0.5)
    _axes[0].set_title("unfiltered")
    _axes[1].set_title("5 to 200 Hz")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (1.1)**:

    1) Use select to make a plot of N180 zoomed into the first charge. `blast_time_zoom_2` and the `borehole_distances` map can help here.

    2) Plot the result with a different colormap.
    """)
    return


@app.cell
def _():
    return


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

    Let's:

    1) determine the apparent velocity from N180, which sits approximately above the blast

    2) count how many charges were in the blast

    Taking the first time derivative may better remove the low-frequency offsets than our `pass_filter`.
    """)
    return


@app.cell
def _(blast_time_zoom_0, borehole_distances, patch):
    from dascore.units import ms

    _d1, _d2 = borehole_distances["N180"]

    # The borehole is a U, so the fiber runs down one leg and back up the other.
    # Only a single leg is a straight line in distance; a window spanning the
    # bend would see each arrival as a V rather than a dipping front.
    _downgoing = (_d1, (_d1 + _d2) / 2)

    processed = (
        patch.select(distance=_downgoing).differentiate(
            "time"
        )  # differentiate along time axis
    )

    # The envelope collapses each wiggle to its amplitude, which makes the blast
    # front easy to see.
    processed.envelope("time").select(time=blast_time_zoom_0).viz.wiggle()
    return ms, processed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Counting the charges

    Assuming the moveout over the array isn't large enough to smear the peaks from different charges together when stacking, averaging the envelope over the leg gives a single trace where each charge should have a peak. We can use `find_peaks` to count them.
    """)
    return


@app.cell
def _(blast_time_zoom_0, dc, plt, processed):
    from scipy.signal import find_peaks

    # Get a patch with the envelope
    _envelope = (
        processed.envelope("time")
        .select(time=blast_time_zoom_0)
        .mean("distance")
        .squeeze()
    )

    # Get the time array in seconds from the start of the window.
    _time_coord = _envelope.get_array("time")
    _time = dc.to_float(_time_coord - _time_coord[0])
    _step = dc.to_float(_envelope.get_coord("time").step)

    # We assume a peak is a charge if:
    # 1) it is at least 10% of the amplitude of the highest peak.
    # 2) it occurs at least 30 ms after the previous charge
    _peaks, _ = find_peaks(
        _envelope.data,
        height=0.1 * _envelope.data.max(),
        distance=int(0.03 / _step),
    )

    peak_times = _time[_peaks]
    _peak_values = _envelope.data[_peaks]

    # plot the results
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_time, _envelope.data)
    _ax.plot(peak_times, _peak_values, "rv")
    _ax.set_xlabel("seconds from window start")
    _ax.set_ylabel("mean envelope of strain rate")
    _ax.set_title(f"{len(_peaks)} peaks")
    _ax
    return (peak_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One peak on its own, then a train of them 0.6 s later, spaced 50 and 100 ms apart, is consistent with delay detonators. They are candidate charges rather than a firing record.

    ### The spectrum

    `dft` transforms along a dimension and replaces it with a frequency coordinate, labeled `ft_{dim}` (e.g. `ft_time`). The amplitude spectrum averaged over the leg shows where the energy is, and the roll-off towards the 1 kHz Nyquist frequency is the anti-alias low-pass filter.
    """)
    return


@app.cell
def _(blast_time_zoom_0, processed):
    # Get the average amplitude spectrum
    n180_spectrum = (
        processed.select(time=blast_time_zoom_0)
        .dft("time", real=True)
        .abs()
        .mean("distance")
        .squeeze()
    )

    # A log axis cannot show the zero-frequency bin, so skip it explicitly.
    _nonzero = n180_spectrum.select(ft_time=(1, ...), samples=True)
    _ax = _nonzero.viz.wiggle()
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax
    return (n180_spectrum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There is also a distinct frequency comb, which is more apparent when making the same plot on a linear scale.
    """)
    return


@app.cell
def _(n180_spectrum):
    # Select before plotting so the vertical scale fits this band too.
    n180_spectrum.select(ft_time=(100, 300)).viz.wiggle()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (1.2)**:

    Given that a comb (evenly spaced impulses) in the time domain is also a comb in the frequency domain:

    $$
    \sum_{n=-\infty}^{\infty} \delta(t-nT)
    \;\xleftrightarrow{\mathcal{F}}\;
    \frac{1}{T}
    \sum_{k=-\infty}^{\infty}
    \delta\left(f-\frac{k}{T}\right)
    $$

    Meaning, the frequency domain spacing ($\Delta f$) is related to the time domain spacing ($T$) as follows:

    $$
    \Delta f = \frac{1}{T}
    $$

    A plausible explanation is that the comb in the spectrum is an expression of the charge spacing. Test this hypothesis by:

    1) Use `peak_times` to determine the anticipated frequency domain peaks from the charge spacing.

    2) Use `find_peaks` on the amplitude spectrum to see if these are consistent.
    """)
    return


@app.cell
def _(peak_times):
    1/ (peak_times[1:] - peak_times[:-1])
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
        # Zoom in around the first charge
        processed.select(time=blast_time_zoom_1)
        
        # taper the window we are about to transform
        .taper(time=10 * ms)  
        
        # correlate all channels against the first channel
        .correlate(distance=0, samples=True)
        
         # drop the length-1 source dimension
        .squeeze() 
        
        # Select lag times around reasonable velocities
        .select(lag_time=(-0.02,0.02))
    )

    cc.viz.waterfall()
    return (cc,)


@app.cell
def _(cc, dc, np):
    # Get the values for the distance axis.
    distance = cc.get_array("distance")

    # Get the values for the lag times.
    _lag = dc.to_float(cc.get_array("lag_time"))

    # Find the lag time of the peak correlation for each channel.
    _peak_indices = cc.data.argmax(axis=cc.get_axis("lag_time"))
    _picks = _lag[_peak_indices]
    slope, intercept = np.polyfit(distance, _picks, 1)

    # The slope is negative: the front reaches the deep end of the leg first and
    # travels back up it, so the absolute value keeps speed and drops direction.
    print(f"apparent velocity: {abs(1 / slope):.0f} m/s")
    return distance, intercept, slope


@app.cell
def _(cc, distance, intercept, slope):
    # Plot, return the MPL axis
    _ax = cc.viz.waterfall(show=False, cmap="seismic", scale=0.3)

    # Reuse MPL axis to plot intercept
    _ = _ax.plot(
        distance, slope * distance + intercept, "--", color="0.5", lw=4
    )
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (1.3)**:

    1) `processed` is the downgoing leg only. Build the same strain-rate patch for the upgoing leg, from the midpoint of `borehole_distances["N180"]` to its end, repeat the correlation, and check that the two velocities agree. Which way does the front travel in that leg?

    2) `viz.spectrogram()` draws the spectrum against time for a single channel. Pick one from `processed` with `select(distance=(10, 11), samples=True).squeeze()` and try it, or hand it the whole leg and it averages the spectra.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - A `Patch` is a contiguous chunk of data and metadata, composed of data, coords, dims and attrs.
    - `Patch` has many processing/transformation methods. Often the dimension is used as the first argument or keyword.
    - `get_array`, `get_coord` and `.data` hand the result to NumPy and SciPy where DASCore stops.

    Next, `02_the_spool.py` picks up the archive these patches came out of.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
