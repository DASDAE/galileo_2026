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

    *This is the solutions copy: each exercise is followed by a worked answer.*

    This notebook addresses learning objective 1. We will use DASCore's `Patch` to explore a blast recorded on the fiber network.
    """)
    return


@app.cell(hide_code=True)
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
def _(mo, patch):
    # Stack the parts so each renders in the cell output rather than the
    # console. The data slice stays small so marimo does not try to display
    # the entire recording.
    mo.vstack(
        [
            mo.md("**`patch.dims`** -- the dimension names:"),
            patch.dims,
            mo.md("**`patch.coords`** -- the labels along each axis:"),
            patch.coords,
            mo.md("**`patch.attrs`** -- the scalar metadata:"),
            patch.attrs,
            mo.md(
                "**`patch.data`** -- the array itself (a small corner of it):"
            ),
            patch.data[:, :5],
        ]
    )
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
    Zooming out and selecting the channel at the bottom of the hole, we see all the charges clearly. Notice how each charge adds its own step to the strain record, ratcheting it up charge by charge -- keep that in mind for Q1 below, which asks whether those steps are real deformation or an instrument artifact.
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

    The blast might have caused permanent deformation to the ground — that is the point of blasting in a mine, after all — but the dynamic response of the instrument could have been exceeded in these near-field recordings.

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

    A simple filter is a quick way to remove the offsets. `Patch.pass_filter` is a standard Butterworth filter with corners in the inverse of the dimension's units, e.g., `time=(5, 200)` is 5 to 200 Hz. This alleviates the offset issue, but does cause some noticeable filter effects in the center of the borehole.
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
    return (n180_blast_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (1.1)**:

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
def _(blast_time_zoom_2, borehole_distances, patch):
    # 1) The fiber runs down the hole and back up, so if the hole is
    # symmetric its bottom is at the midpoint of the span and the downgoing
    # leg is the first half. select takes both trims at once, and
    # blast_time_zoom_2 is the 70 ms window on the front used above.
    _d1, _d2 = borehole_distances["N180"]
    _downgoing = (_d1, (_d1 + _d2) / 2)
    _front = patch.select(distance=_downgoing, time=blast_time_zoom_2)

    # 2) Any matplotlib colormap name works.
    _front.viz.waterfall(cmap="viridis")
    return


@app.cell
def _(dc, n180_blast_filtered):
    # 3) decimate low-pass filters, then keeps every fourth sample, so the
    # 2 kHz record becomes 500 Hz.
    _decimated = n180_blast_filtered.decimate(time=4)

    # The coordinate's step is the sample spacing; to_float makes seconds.
    _step = dc.to_float(n180_blast_filtered.get_coord("time").step)
    _decimated_step = dc.to_float(_decimated.get_coord("time").step)
    print(f"shape {n180_blast_filtered.shape} -> {_decimated.shape}")
    print(f"time step {_step * 1e3:g} ms -> {_decimated_step * 1e3:g} ms")

    _decimated.viz.wiggle(scale=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changed: the time step (0.5 to 2 ms), the sample count (4400 to 1100 per channel), the highest frequency the data can hold (the Nyquist frequency, half the sample rate: 1 kHz to 250 Hz), and the values a little, from the anti-alias filter `decimate` applies first. Unchanged: the 2.2 s window, the 35 channels, and the picture, because the 5 to 200 Hz band still sits below the new 250 Hz Nyquist.
    """)
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

    Assuming the moveout over the array isn't large enough to smear peaks from different charges together, averaging the envelope over the leg gives a single trace where each charge should have a peak. We can use `find_peaks` to count them.
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
    return find_peaks, peak_times


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Solution**
    """)
    return


@app.cell
def _(dc, find_peaks, n180_spectrum, np, peak_times):
    # 1) peak_times[1:] drops the lone first charge, 0.6 s ahead of the
    # train. The gaps between the rest are close to 50 ms, or 100 ms where a
    # slot is empty or too weak to pick, so the median finds the 50 ms grid. A
    # comb every 50 ms in time is a comb every 1 / 0.05 s = 20 Hz in
    # frequency.
    _period = np.median(np.diff(peak_times[1:]))
    _delta_f = 1 / _period
    print(
        f"charge spacing {_period * 1e3:.2f} ms -> comb every {_delta_f:.2f} Hz"
    )

    # 2) Trim the spectrum to where the comb was clearest, 100 to 300 Hz in
    # the linear plot above, with a 5 Hz margin so a tooth right at 100 or
    # 300 Hz is not on the edge of the array, where find_peaks cannot see it.
    _comb = n180_spectrum.select(ft_time=(95, 305))
    _freq = _comb.get_array("ft_time")
    _amp = _comb.data

    # find_peaks measures distance in samples, so 5 Hz is divided by the
    # frequency step; height keeps only teeth at least a third as tall as
    # the tallest.
    _freq_step = dc.to_float(_comb.get_coord("ft_time").step)
    _peaks, _ = find_peaks(
        _amp, height=_amp.max() / 3, distance=int(5 / _freq_step)
    )
    _peak_freqs = _freq[_peaks]
    _peak_amplitudes = _amp[_peaks]

    # Peaks also appear halfway between the predicted 20 Hz teeth. Use a third
    # of the observed peak spacing as the matching tolerance; unlike one fixed
    # frequency bin, this allows small period-picking errors to accumulate at
    # the higher harmonics without reaching the interleaved family.
    _observed_spacing = np.median(np.diff(_peak_freqs))
    _nearest_harmonic = np.round(_peak_freqs / _delta_f) * _delta_f
    _predicted = abs(_peak_freqs - _nearest_harmonic) <= _observed_spacing / 3
    assert _predicted.any() and (~_predicted).any()
    _amplitude_ratio = np.median(_peak_amplitudes[_predicted]) / np.median(
        _peak_amplitudes[~_predicted]
    )
    print("spectral peaks [Hz]:", np.round(_peak_freqs, 1))
    print(f"all detected peaks are {_observed_spacing:.2f} Hz apart")
    print(
        f"the predicted {_delta_f:.2f} Hz teeth have "
        f"{_amplitude_ratio:.2f}x the median amplitude"
    )

    # Draw the band, mark the picks, and put a grey line at every multiple
    # of the predicted spacing to compare against.
    _ax = _comb.viz.wiggle()
    _ax.plot(
        _peak_freqs[_predicted],
        _peak_amplitudes[_predicted],
        "rv",
        label=f"near the {_delta_f:.2f} Hz grid",
    )
    _ax.plot(
        _peak_freqs[~_predicted],
        _peak_amplitudes[~_predicted],
        "^",
        color="orange",
        label="interleaved teeth",
    )
    for _harmonic in np.arange(_delta_f, 305, _delta_f):
        _ax.axvline(_harmonic, color="0.8", zorder=0)
    _ax.set_title(f"spectral peaks against multiples of {_delta_f:.2f} Hz")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at every detected peak gives a 10 Hz spacing, but the plot separates two families: the taller red teeth fall near multiples of the predicted 20 Hz spacing, while the smaller orange teeth sit between them. The 50 ms charge spacing therefore explains the dominant teeth. The interleaved teeth imply a pattern that repeats every 100 ms, but this record cannot distinguish a missing charge from one that fired too weakly to pick in the envelope. Nor can one blast rule out another source of a 10 Hz comb, so treat this as a hypothesis that fits rather than a reconstructed firing record.

    ### Seeing the 100 ms repeat

    Both domains show it, side by side. In time, the picked charges sit on a 50 ms grid, but the pick heights tend to alternate strong/weak -- consistent with a repeating unit, a strong charge plus a weak one, spanning 100 ms (with the caveat above: envelope height is a proxy, not a firing record, and the alternation is not perfect). A comb's tooth spacing reads the repeat period of the *whole pattern*, so a 100 ms unit puts teeth every 1 / 100 ms = 10 Hz, while the 50 ms grid inside it survives as the taller every-other tooth on the 20 Hz grid.
    """)
    return


@app.cell
def _(blast_time_zoom_0, dc, find_peaks, n180_spectrum, np, plt, processed):
    # Rebuild the mean envelope and its picks from the counting cell above.
    _envelope = (
        processed.envelope("time")
        .select(time=blast_time_zoom_0)
        .mean("distance")
        .squeeze()
    )
    _time_coord = _envelope.get_array("time")
    _time = dc.to_float(_time_coord - _time_coord[0])
    _step = dc.to_float(_envelope.get_coord("time").step)
    _picks, _ = find_peaks(
        _envelope.data,
        height=0.1 * _envelope.data.max(),
        distance=int(0.03 / _step),
    )
    # Drop the lone opener and anchor a 50 ms grid on the first train charge.
    _train = _time[_picks][1:]
    _t0 = _train[0]
    _slots = int(round((_train[-1] - _t0) / 0.05)) + 1
    _grid = _t0 + 0.05 * np.arange(_slots)
    _occupied = np.array([np.abs(_train - _g).min() < 0.015 for _g in _grid])

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Time domain: the 50 ms grid, its empty slots, and the strong/weak
    # alternation that makes the true repeat unit 100 ms.
    _ax1.plot(_time, _envelope.data, color="tab:blue", lw=1)
    for _g, _occ in zip(_grid, _occupied):
        _ax1.axvline(
            _g,
            color="0.75" if _occ else "crimson",
            ls=":" if _occ else "--",
            lw=1 if _occ else 1.5,
            zorder=0,
        )
    _ax1.plot(_train, _envelope.data[_picks][1:], "rv", label="picked charges")
    _ymax = _envelope.data.max()
    _ax1.annotate(
        "",
        xy=(_t0 + 0.05, _ymax * 0.9),
        xytext=(_t0, _ymax * 0.9),
        arrowprops=dict(arrowstyle="<->", color="k"),
    )
    _ax1.text(_t0 + 0.025, _ymax * 0.94, "50 ms", ha="center")
    _ax1.annotate(
        "",
        xy=(_t0 + 0.3, _ymax * 0.72),
        xytext=(_t0 + 0.2, _ymax * 0.72),
        arrowprops=dict(arrowstyle="<->", color="darkgreen"),
    )
    _ax1.text(
        _t0 + 0.25,
        _ymax * 0.76,
        "100 ms between strong charges",
        color="darkgreen",
        ha="center",
        bbox=dict(fc="white", ec="none", pad=1),
    )
    for _g in _grid[~_occupied]:
        _ax1.text(
            _g,
            _ymax * 0.55,
            "gap",
            color="crimson",
            ha="center",
            bbox=dict(fc="white", ec="none", pad=1),
        )
    _ax1.set_xlim(_t0 - 0.08, _train[-1] + 0.08)
    _ax1.set_xlabel("seconds from window start")
    _ax1.set_ylabel("mean envelope")
    _ax1.set_title(
        "time domain: charges on a 50 ms grid, but not every slot fires alike"
    )
    _ax1.legend(loc="upper right")

    # Frequency domain: the same spectrum as above, with both families and
    # both spacings labeled.
    _comb = n180_spectrum.select(ft_time=(95, 305))
    _freq = _comb.get_array("ft_time")
    _amp = _comb.data
    _freq_step = dc.to_float(_comb.get_coord("ft_time").step)
    _teeth, _ = find_peaks(
        _amp, height=_amp.max() / 3, distance=int(5 / _freq_step)
    )
    _pf, _pa = _freq[_teeth], _amp[_teeth]
    _on_grid = np.abs(_pf - np.round(_pf / 20) * 20) <= 3

    _ax2.plot(_freq, _amp, color="0.4", lw=0.8)
    _ax2.plot(
        _pf[_on_grid],
        _pa[_on_grid],
        "rv",
        ms=9,
        label="on the 20 Hz grid = 1 / 50 ms",
    )
    _ax2.plot(
        _pf[~_on_grid],
        _pa[~_on_grid],
        "v",
        color="orange",
        ms=7,
        label="interleaved teeth",
    )
    # Anchor the annotation arrows on the teeth nearest fixed frequencies so
    # a changed pick count cannot shift them onto the wrong pair.
    _a0 = _pf[np.abs(_pf - 150).argmin()]
    _a1 = _pf[np.abs(_pf - 160).argmin()]
    _arrow_height = _amp.max() * 0.85
    _ax2.annotate(
        "",
        xy=(_a1, _arrow_height),
        xytext=(_a0, _arrow_height),
        arrowprops=dict(arrowstyle="<->", color="k"),
    )
    _ax2.text(
        (_a0 + _a1) / 2,
        _arrow_height * 1.04,
        "10 Hz = 1 / 100 ms",
        ha="center",
    )
    _s0 = _pf[np.abs(_pf - 200).argmin()]
    _s1 = _pf[np.abs(_pf - 220).argmin()]
    _ax2.annotate(
        "",
        xy=(_s1, _arrow_height * 0.6),
        xytext=(_s0, _arrow_height * 0.6),
        arrowprops=dict(arrowstyle="<->", color="k"),
    )
    _ax2.text(
        (_s0 + _s1) / 2,
        _arrow_height * 0.64,
        "20 Hz = 1 / 50 ms",
        ha="center",
    )
    _ax2.set_xlabel("frequency [Hz]")
    _ax2.set_ylabel("mean amplitude")
    _ax2.set_title(
        "frequency domain: teeth every 10 Hz -- the spacing of a 100 ms repeat"
    )
    _ax2.legend(loc="upper right")
    _fig.tight_layout()
    _fig
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
            time=blast_time_zoom_1  # Zoom in around the first charge
        )
        .taper(time=10 * ms)  # taper the window we are about to transform
        .correlate(
            distance=0,
            samples=True,  # correlate all channels against the first channel
        )
        .squeeze()  # drop the length-1 source dimension
        .select(
            lag_time=(
                -0.02,
                0.02,
            )  # Select lag times around reasonable velocities
        )
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
    # Plot
    _ax = cc.viz.waterfall(show=False, cmap="seismic", scale=0.3)
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
    **Solution**
    """)
    return


@app.cell
def _(blast_time_zoom_1, borehole_distances, dc, ms, np, patch, slope):
    # 1) The same chain as above, for the second half of the hole.
    _d1, _d2 = borehole_distances["N180"]
    _midpoint = (_d1 + _d2) / 2
    _upgoing = patch.select(distance=(_midpoint, _d2)).differentiate("time")
    _cc = (
        _upgoing.select(time=blast_time_zoom_1)
        .taper(time=10 * ms)
        .correlate(distance=0, samples=True)
        .squeeze()
        .select(lag_time=(-0.02, 0.02))
    )
    _distance = _cc.get_array("distance")
    _lag = dc.to_float(_cc.get_array("lag_time"))
    _picks = _lag[_cc.data.argmax(axis=_cc.get_axis("lag_time"))]
    _slope, _ = np.polyfit(_distance, _picks, 1)

    # The sign flips: in the upgoing leg, distance along the fiber increases
    # towards the surface, and so does arrival time. The front still comes
    # from below and travels up the leg; only the direction the fiber counts
    # distance in has changed. The speeds agree to within a few percent.
    print(f"downgoing leg: {1 / slope:+.0f} m/s")
    print(f"upgoing leg:   {1 / _slope:+.0f} m/s")
    return


@app.cell
def _(blast_time_zoom_0, processed):
    # 2) select(..., samples=True) picks channel 10 by index but keeps
    # distance as a length-1 dimension; squeeze drops it, leaving the single
    # time series spectrogram wants. Trimming to the blast window keeps the
    # quiet record from setting the colour scale.
    _channel = processed.select(distance=(10, 11), samples=True).squeeze()
    _channel.select(time=blast_time_zoom_0).viz.spectrogram()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The lone first charge is the short burst at 26.8 s and the train the block from 27.4 to 28.3 s. The energy sits below about 600 Hz, the roll-off the spectrum showed.
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
