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
    # The Spool

    This notebook introduces the `Spool` (learning objective 2). We will use DASCore's `Spool` to explore the data archive.

    This repository includes four types of fiber data:

    1. A day of Distributed Strain Sensing (DSS) data collected by a Febus G1 BOTDR.
    2. Low-frequency Distributed Acoustic Sensing (DAS) data collected by a Sintela Onyx Peta.
    3. High-frequency (2 kHz) DAS data collected by the same Onyx Peta (which we saw in notebook 01).
    4. Optical Time Domain Reflectometer (OTDR) traces taken with a Tempo Communications OFL100.

    It also includes an inventory, which will be discussed in notebook 03.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note
    The `Spool` manages a source of patches. In this case, a directory of fiber data files.

    Sometimes, for organization and metadata association, it is helpful to add metadata to files/patches. However, it is not practical (and bad practice) to change the raw data files. DASCore provides a simple way to add metadata based on the folder name.

    For example, this dataset is organized like this:

    ```text
    fiber/
    ├── tag=DSS__acquisition_key=XM.MINE1.03.WSF/
    │   ├── conf2-strainmonitoring2_C1_2026-08-06T01.02.56+0200.bsl.h5
    │   └── ...
    ├── tag=DAS_LF__acquisition_key=XM.MINE1.04.MSF/
    │   ├── LowFrequency_Decimator_2_2026-08-06_10.09.54_UTC_010214.raw
    │   └── ...
    ├── tag=DAS__acquisition_key=XM.MINE1.04.FSF/
    │   └── blast.h5
    └── tag=OTDR/
        ├── channel_3_otdr.sor
        └── channel_4_otdr.sor
    ```

    Every patch read out of the first directory carries `tag == "DSS"` and `acquisition_key == "XM.MINE1.03.WSF"`, although neither string appears anywhere inside the file. By design, this does not apply to the top-level directory or to file names. A directory may carry several key=value pairs joined by `__`.

    This is the same convention used by [Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-PartitionedTables), whose partitioned tables gave the layout its name, and which [PyArrow](https://arrow.apache.org/docs/python/dataset.html#partitioning), [DuckDB](https://duckdb.org/docs/stable/data/partitioning/hive_partitioning), Spark and Delta Lake all read the same way.
    ///
    """)
    return


@app.cell
def _():

    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np

    from galileo_2026 import get_data_path

    plt.rcParams["figure.figsize"] = (10, 6)  # width, height in inches

    # The optical distance range of borehole N180 on each fiber.
    dss_n180_dist = (2525.7, 2583.3)
    lfdas_n180_dist = (1577.9, 1634.7)

    # The time of the blast, from notebook 01.
    blast_time = np.datetime64("2026-08-06T10:13:26.8")
    return blast_time, dc, dss_n180_dist, get_data_path, lfdas_n180_dist, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Make a spool

    `dc.spool` is the go-to function for reading fiber data. It accepts a variety of data sources including a directory, a file, in-memory data, a URL to a remote file, etc. It returns a `Spool` instance.

    Calling `.update()` ensures the spool is in sync with the data source.

    Like the patch, the spool has a rich representation that provides insight into its contents.
    """)
    return


@app.cell
def _(dc, get_data_path):
    # First get a path to our data directory.
    _data_path = get_data_path()

    # Next create the spool and make sure it is up-to-date.
    spool = dc.spool(_data_path).update()
    spool
    return (spool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The spool content tables

    Internally, the spool uses an SQLite database with several tables to manage the patch sources, but this is internal to DASCore. When the information shown in the spool repr is not enough, users can export a simplified table view of a `Spool`'s managed contents as a dataframe, where each row represents one `Patch`.

    We use `get_contents` for this.
    """)
    return


@app.cell
def _(spool):
    df = spool.get_contents()
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the dataframe, we can answer some basic questions about the data.

    For example, as we saw above, the data directory is structured such that each type of data has its own tag. We could use the dataframe to count how many files of each kind there are:
    """)
    return


@app.cell
def _(df):
    df.value_counts("tag")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataframe can also be used to select the `Spool`'s contents. A boolean series built from it indexes the spool, so a pandas query becomes a smaller spool.
    """)
    return


@app.cell
def _(df, spool):
    lf_spool = spool[df["tag"] == "DAS_LF"]
    print(f"{len(lf_spool)} low-frequency DAS patches")
    return (lf_spool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (2.1)**
    Determine how many files have a valid interrogator serial number set from the file.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting availability and gaps

    `Spool.get_gaps` and `Spool.get_coverage` both return dataframes that answer questions about the archive's continuity. The `viz` namespace plots what they report: `coverage` draws the archive as lanes along a dimension, `calendar` bins it by day.
    """)
    return


@app.cell
def _(spool):
    # The archive spans about four weeks, because the two OTDR traces sit weeks
    # before everything else, so, drawn whole, the minutes-long DAS lanes are
    # thinner than a pixel. Give the dimension a window to draw the part worth
    # reading.
    spool.select(time=("2026-08-06T10", "2026-08-06T11")).viz.coverage()
    return


@app.cell
def _(spool):
    # The DSS data crosses a day boundary, which is enough to demonstrate the calendar view.
    spool.select(tag="DSS").viz.calendar()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Patches

    Patches are extracted (loaded) only when you ask for them. This is done primarily through item access or iteration as in the following examples:
    """)
    return


@app.cell
def _(spool):
    first_patch = spool[0]
    return (first_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first patch is an OTDR trace of channel 4.
    """)
    return


@app.cell
def _(first_patch):
    first_patch.viz.wiggle()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also access patches through iteration. For example, let's find the first patch with a DAS_LF tag (we will see a much better way to do this later on):
    """)
    return


@app.cell
def _(spool):
    for das_lf_patch in spool:
        if das_lf_patch.attrs.tag == "DAS_LF":
            break
    else:
        raise RuntimeError("No patch found")
    return (das_lf_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have our low-frequency DAS patch, let's convert to strain and plot.
    """)
    return


@app.cell
def _(das_lf_patch):

    das_lf_patch.radians_to_strain().viz.waterfall(
        scale=(-3e-7, 3e-7), scale_type="absolute"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unlike the 2 kHz DAS data from the last notebook, here we have all 14 boreholes, shown as vertical stripes near 0 strain. The fiber hanging on the mine walls is not tightly coupled, hence its higher strain levels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting and Reshaping Spool Contents

    There are better ways to select output from the spool than iterating over all of its contents. `Spool.select` works on both attributes and coordinates, and trims patches to match the query. For example, let's make a spool that only contains the DSS data within ±10 hours of the blast.
    """)
    return


@app.cell
def _(spool):
    spool_dss_blast = spool.select(
        time=("2026-08-06T00:15", "2026-08-06T20:15"), tag="DSS"
    )
    return (spool_dss_blast,)


@app.cell
def _(spool_dss_blast):
    print(spool_dss_blast)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The problem here is that there are multiple patches, each one representing a single file. We need to merge them together. We use [`Spool.chunk`](https://dascore.org/api/dascore/core/spool/BaseSpool/chunk.html) for this.

    `conflict="drop"` tells the spool to drop attributes whose values differ between files (here `freq_offset`) rather than refuse to merge.
    """)
    return


@app.cell
def _(spool_dss_blast):
    merged_spool_dss_blast = spool_dss_blast.chunk(time=None, conflict="drop")
    return (merged_spool_dss_blast,)


@app.cell
def _(merged_spool_dss_blast):
    assert len(merged_spool_dss_blast) == 1
    dss_patch = merged_spool_dss_blast[0]
    return (dss_patch,)


@app.cell
def _(dss_patch, plt):
    _fig, _ax = plt.subplots(1, 1)
    dss_patch.viz.waterfall(ax=_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since this is BOTDR data, we are seeing strain in relation to an unknown reference. We will remedy this in the next sections by setting a baseline. For now, note that the splices stand out as thin vertical lines that contrast with their surroundings, which lets us delineate the last hole, N180, as **distance=(2525.7, 2583.3)**.

    Next, we get a baseline using the average of the first hour of data, and look at strain in N180, the hole closest to the blast.
    """)
    return


@app.cell
def _(dss_n180_dist, dss_patch):
    # We trim the patch with relative=True (relative to start/end)
    _first_hour = dss_patch.select(time=(..., 60 * 60), relative=True)
    _averaged_first_hour = _first_hour.mean("time").squeeze()

    _corrected = dss_patch - _averaged_first_hour
    dss_n180 = _corrected.select(distance=dss_n180_dist)
    return (dss_n180,)


@app.cell
def _(dss_n180):
    dss_n180.viz.waterfall()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is a bit messy; perhaps removing the median offsets and a bit of smoothing will help.
    """)
    return


@app.cell
def _(dss_n180):
    dss_n180_processed = (
        # demedian removes each time sample's offset (the horizontal stripes);
        # rolling(distance=1) averages a 1 m window, ten channels at this
        # 0.1 m spacing, to tame the channel-to-channel speckle.
        dss_n180.demedian("distance").rolling(distance=1).mean()
    )
    dss_n180_processed.viz.waterfall()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Still noisy -- BOTDR near its resolution limit is -- but the horizontal striping is gone, and what to look for now stands out: the red column near 2550 m, strain building up over the hours around the blast, against fiber that stays quiet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (2.2)**
    Rechunk `spool_dss_blast` to 1 hour segments with 15 minutes of overlap, and examine the result. Look at each patch's start and end times: do they behave the way you expected? (Hint: a DSS sweep arrives about every ten minutes, and a chunk can only hold whole sweeps.)
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gaps and tolerance

    Next, we get a similar patch for the LF DAS data. The DSS data merged easily because it had no gaps. Normally, `chunk(time=None)` only joins data that is contiguous, and `get_gaps` reports where it is not. Some of the LF DAS files leave 0.4 to 1.3 s spaces between them, jitter in the file timing, and those show up as gaps:
    """)
    return


@app.cell
def _(lf_spool):
    lf_spool.get_gaps()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Whether the gaps break a merge is controlled by the `tolerance` parameter in `Spool.chunk`. `tolerance` is an integer number of samples or a quantity with units.

    At 5 Hz the default of 1.5 samples is 0.3 s, every gap here is wider, and `chunk(time=None)` keeps those files apart; ten samples forgives two seconds and merges all the files, but DASCore warns that the coordinate may have been altered. In this case, squashing the gaps is fine.

    To compare to the DSS record, we will simply look at the N180 section, which for this optical path is **distance=(1577.9, 1634.7)**:
    """)
    return


@app.cell
def _(lf_spool, lfdas_n180_dist):
    _merged = lf_spool.select(distance=lfdas_n180_dist).chunk(
        time=None, conflict="drop", tolerance=10
    )
    assert len(_merged) == 1
    lf_patch = _merged[0]
    lf_patch
    return (lf_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now to plot the two patches together with a line indicating the time of the blast.
    """)
    return


@app.cell
def _(blast_time, dss_n180, lf_patch, plt):
    # Each panel keeps its own time and color scale: the DSS record spans the
    # day, the LF DAS record the 22 minutes around the blast. Only the units
    # are shared, so the colorbars read alike.
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot dss
    dss_n180.viz.waterfall(ax=_axes[0])

    # Convert lfdas to comparable units and plot
    _lf_patch_ue = lf_patch.radians_to_strain().convert_units("microstrain")
    _lf_patch_ue.viz.waterfall(ax=_axes[1])

    # Plot the blast time
    for _ax in _axes:
        _ax.axhline(blast_time, color="lime", linestyle="--", linewidth=2.5)
    _axes[0].set_title("DSS, N180")
    _axes[1].set_title("LF DAS, N180")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (2.3)**
    Notebook 01 saw an offset at N180 after the blast. Let's revisit questions 1 and 2.

    1) In the N180 DSS and LF DAS records, do you see such an offset?

    2) If so, is the step visible above the noise? That is, estimate each channel's pre-blast noise level with `std("time")` on a window before the blast, and compare the size of the step to it.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Processing whole archives

    `Spool.map` applies a function to every patch, loading each only as the function needs it. Here we use it to build a patch of hourly means from the DSS data.
    """)
    return


@app.cell
def _(dc, dss_n180_dist, spool):
    def _hourly_mean(patch):
        return patch.mean("time", dim_reduce="mean")

    # Get a list of hourly means
    _hourly_means = (
        spool.select(tag="DSS", distance=dss_n180_dist)
        .chunk(time=3600, conflict="drop")
        .map(_hourly_mean)
    )

    # Put the hourly means back into one patch.
    _hourly_spool = dc.spool(_hourly_means).concatenate(
        time=None, conflict="drop"
    )
    _hourly = _hourly_spool[0]

    # Correct, taking the first hour as the baseline.
    corrected = _hourly - _hourly.select(time=0, samples=True).squeeze()
    corrected.viz.waterfall()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, we created a list of patches and concatenated them. In practice, when the results are too big to keep in memory, save each patch to disk inside the mapped function and return something small instead:
    """)
    return


@app.cell
def _(dc, dss_n180_dist, spool):
    from pathlib import Path
    from tempfile import mkdtemp

    _out_dir = Path(mkdtemp(prefix="hourly_means_"))

    def _save_hourly_mean(patch):
        reduced = patch.mean("time", dim_reduce="mean")
        # Name the file by its start time; ":" is not portable in file names.
        start = str(reduced.get_coord("time").min()).replace(":", "-")
        path = _out_dir / f"{start}.h5"
        reduced.io.write(path, "dasdae")
        return path  # a path, not a patch: nothing big stays in memory

    _paths = (
        spool.select(tag="DSS", distance=dss_n180_dist)
        .chunk(time=3600, conflict="drop")
        .map(_save_hourly_mean)
    )
    print(f"wrote {len(_paths)} files to {_out_dir}")

    # The saved files are themselves a spool, ready for the next step.
    dc.spool(_out_dir).update().get_contents().head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also note, `Spool.map` supports several types of parallelism using the `client` argument.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - `dc.spool(directory).update()` indexes an archive once; it only needs to run again if the contents of the archive change.
    - `get_contents()` is the archive as a dataframe; a boolean series built from it indexes the spool.
    - `select` narrows by attribute (with wildcards) or by coordinate range, trimming patches as well as choosing them.
    - `chunk` re-cuts the spool along a dimension, by length, overlap, or memory budget, and `chunk(time=None)` merges contiguous data.
    - `get_gaps` lists the breaks in an archive, and `tolerance`, in samples or a unit quantity, decides which of them a merge forgives.
    - `map` runs a function over every patch, and `dc.spool(results).concatenate` puts what comes back into one patch.

    Next, `03_the_inventory.py` attaches the observing system to these patches.
    """)
    return


if __name__ == "__main__":
    app.run()
