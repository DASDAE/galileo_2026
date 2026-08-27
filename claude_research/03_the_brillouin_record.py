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
    # What the Brillouin record knows

    `01_the_step_response.py` takes the DAS phase apart and shows that the offset left behind by the blast is the interrogator's own high-pass response, with a time constant of sixteen seconds. That settles what the offset *is*, and it settles that the DAS cannot answer the question underneath it: did the blast deform the rock permanently? A phase-measuring instrument with a corner near 10 mHz has no opinion about a change that lasts forever.

    An instrument that measures strain absolutely does. The Brillouin sweeps come every ten minutes all day, so this notebook asks them two questions:

    1. did any borehole move, comparing the three hours after the blast against the three hours before?
    2. and — since answering the first one forces a robust statistic — what are the wild values that made the robust statistic necessary?

    The second question turns out to be the more interesting of the two.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from galileo_2026 import get_data_path, get_inventory_path

    # The blast, from notebook 01.
    blast = np.datetime64("2026-08-06T10:13:27")

    plt.rcParams["figure.figsize"] = (10, 6)
    return (
        blast,
        dc,
        get_data_path,
        get_inventory_path,
        np,
        pd,
        plt,
    )


@app.cell
def _(dc, get_data_path, get_inventory_path):
    inv = dc.inventory(get_inventory_path())
    spool = dc.spool(get_data_path()).update().attach_inventory(inv).enrich()
    return inv, spool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting the sweeps on the path

    A fiber end is a fiber end in both distance systems, and the inventory maps between them: the path ends at 2607.7 m, and the acquisition's `distance_map` says where that falls on the instrument's own axis. Everything past that is the fiber lying in the shack.
    """)
    return


@app.cell
def _(inv, np, spool):
    # The Brillouin acquisition and the path it reads.
    _array = inv.networks[0].fiber_arrays[0]
    dss_acquisition = next(a for a in _array.acquisitions if a.code == "WSF")
    dss_path = next(p for p in _array.optical_paths if p.location_code == "03")
    _map = dss_acquisition.distance_map
    fiber_end = np.interp(
        dss_path.distance_max, _map.distance, _map.instrument_distance
    )
    print(
        f"the fiber ends at {dss_path.distance_max} m of path, {fiber_end:.1f} m on the instrument"
    )

    dss_patch = (
        spool.select(tag="DSS")
        .chunk(time=None, conflict="drop")[0]
        .select(distance=(..., fiber_end))
    )
    # What the whole fiber did in a sweep is the instrument's reference, not
    # the rock; the median along distance takes it out (notebook 02).
    dss_local = dss_patch - dss_patch.median("distance")
    dss_local
    return dss_acquisition, dss_local, dss_patch, dss_path, fiber_end


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Did anything move?

    Notebook 02 compared means. Here the profiles are medians over the sweeps, and the noise is the median absolute deviation: a Brillouin sweep occasionally returns nonsense at a channel, and a mean of eighteen sweeps with one wild value in them is the wild value.
    """)
    return


@app.cell
def _(blast, dss_local, np):
    _three_hours = np.timedelta64(3, "h")
    _ten_minutes = np.timedelta64(10, "m")
    dss_before = dss_local.select(
        time=(blast - _three_hours, blast - _ten_minutes)
    )
    dss_after = dss_local.select(
        time=(blast + _ten_minutes, blast + _three_hours)
    )

    dss_step = (dss_after.median("time") - dss_before.median("time")).squeeze()
    # Median absolute deviation of the sweeps before the blast, scaled to a
    # standard deviation. Sweeps are pulled apart with numpy: the residual is
    # the same shape as the data, so there is nothing for a patch to add.
    _residual = np.abs(dss_before.data - np.median(dss_before.data, axis=0))
    dss_noise = dss_step.new(data=1.4826 * np.median(_residual, axis=0))
    return dss_after, dss_before, dss_noise, dss_step


@app.cell
def _(dss_noise, dss_step, np, pd):
    _rows = []
    for _hole in sorted(set(dss_step.get_array("borehole")) - {""}):
        _step = dss_step.select(borehole=_hole)
        _noise = dss_noise.select(borehole=_hole)
        _largest = np.abs(_step.data).argmax()
        _rows.append(
            {
                "borehole": _hole,
                "channels": _step.shape[0],
                "noise [µε]": np.median(_noise.data),
                "largest |step| [µε]": np.abs(_step.data).max(),
                "step / noise": np.abs(_step.data).max()
                / np.median(_noise.data),
                "at hole depth [m]": _step.get_array("hole_depth")[_largest],
                "leg": _step.get_array("leg")[_largest],
            }
        )
    pd.DataFrame(_rows).set_index("borehole").round(1)
    return


@app.cell
def _(dss_noise, dss_step, np, plt):
    _holes = sorted(set(dss_step.get_array("borehole")) - {""})
    _fig, _axes = plt.subplots(
        7, 2, figsize=(12, 16), sharex=True, sharey=True
    )
    for _ax, _hole in zip(_axes.ravel(), _holes):
        _step = dss_step.select(borehole=_hole)
        _noise = dss_noise.select(borehole=_hole)
        _depth = _step.get_array("hole_depth")
        # Down leg on the left of the hole bottom, up leg on the right.
        _x = np.where(_step.get_array("leg") == "down", -_depth, _depth)
        _order = np.argsort(_x)
        _ax.fill_between(
            _x[_order],
            -_noise.data[_order],
            _noise.data[_order],
            color="0.85",
            label="±1 noise",
        )
        _ax.plot(
            _x[_order], _step.data[_order], lw=0.8, label="after - before"
        )
        _ax.axhline(0, color="0.5", lw=0.5)
        _ax.set_title(_hole, fontsize=9, loc="left")
    _axes[0, 0].legend(fontsize=7)
    _axes[0, 0].set_ylim(-80, 80)
    for _ax in _axes[-1]:
        _ax.set_xlabel("← down leg   hole depth [m]   up leg →")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("µε")
    _fig.suptitle(
        "Brillouin strain: three hours after the blast minus three hours before"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fourteen holes, and nothing that stands clear of the grey band. The noise is 12 to 19 µε; the only channels that reach twice it are single channels at the collar of N120 and S150, and the next section shows those are exactly the channels where the Brillouin fit fails. A permanent change of 40 µε at one channel would show; a change over a whole leg of 10 µε would show as a level shift, and there is none. The blast left the grouted rock, within the BOTDR's resolution, where it was.

    That is the answer, and it is a negative one, but notice what it took to reach it: the instrument that can see the transient cannot see the permanent, and the one that can see the permanent cannot see the transient. The archive needed both, and the inventory made the same hole selectable on both.

    ## What the dropouts know

    The wild sweep values that forced the medians above are not random. Flag every sample more than eight scaled deviations from its channel's median, count them per channel, and put the counts where they belong on the path: the acquisition's `distance_map` converts the instrument's axis into the path's, which is where the inventory keeps its splices and its boreholes.
    """)
    return


@app.cell
def _(dss_acquisition, dss_local, dss_path, np, pd):
    _residual = dss_local.data - np.median(dss_local.data, axis=0)
    _scale = 1.4826 * np.median(np.abs(_residual), axis=0)
    dropouts = (np.abs(_residual) > 8 * _scale).sum(axis=0)
    print(
        f"{dropouts.sum()} samples flagged in {dss_local.shape[0]} sweeps, "
        f"{(dropouts > 0).sum()} channels touched, "
        f"{(dropouts >= 5).sum()} hit five times or more"
    )

    path_distance = dss_acquisition.distance_map.map_to_distance(
        dss_local.get_array("distance")
    )
    splices = pd.Series(
        {
            c.name: c.distance_min
            for c in dss_path.optical_components
            if c.object_type == "Splice"
        }
    )
    holes = pd.DataFrame(
        [
            {
                "borehole": lab.value,
                "collar_down": lab.distance_min,
                "collar_up": lab.distance_max,
            }
            for lab in dss_path.labels
            if lab.group == "borehole"
        ]
    ).set_index("borehole")
    return dropouts, holes, path_distance, splices


@app.cell
def _(dropouts, holes, path_distance, plt, splices):
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _ax.plot(path_distance, dropouts, lw=0.7)
    for _d in splices:
        _ax.axvline(_d, color="C3", lw=0.6)
    for _, _hole in holes.iterrows():
        _ax.axvspan(_hole.collar_down, _hole.collar_up, color="0.88")
    _ax.set_xlabel("path distance [m]")
    _ax.set_ylabel("dropouts per channel")
    _ax.set_title(
        "where the Brillouin fit fails: grey are boreholes, red are the OTDR's splices"
    )
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sharp spikes, one or two channels wide, and not in the middle of anything. Each hot channel sits at a distinct place: a splice the OTDR reported, a borehole collar, or somewhere else, and the table says which.
    """)
    return


@app.cell
def _(dropouts, holes, np, path_distance, pd, splices):
    _rows = []
    for _i in np.where(dropouts >= 5)[0]:
        _d = path_distance[_i]
        _splice = (splices - _d).abs().idxmin()
        _collars = pd.concat(
            [
                holes.collar_down.rename(lambda h: f"{h} down"),
                holes.collar_up.rename(lambda h: f"{h} up"),
            ]
        )
        _nearest_collar = (_collars - _d).abs().idxmin()
        _rows.append(
            {
                "path [m]": round(_d, 1),
                "dropouts": int(dropouts[_i]),
                "nearest splice": _splice,
                "past it [m]": round(_d - splices[_splice], 1),
                "nearest collar": _nearest_collar,
                "before it [m]": round(_collars[_nearest_collar] - _d, 1),
            }
        )
    hot_spots = pd.DataFrame(_rows)
    # One row per spot rather than per channel of a wide spot.
    hot_spots = hot_spots[hot_spots["path [m]"].diff().fillna(99).abs() > 1]
    hot_spots
    return (hot_spots,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the last two columns. Seven spots are *at* a collar, within twenty centimetres: the cable enters the grout there, and within one metre of spatial resolution the fiber is hanging free on one side and strained by the rock on the other. The Brillouin spectrum in that cell has two peaks, and the fit picks one or the other from sweep to sweep. The worst spot of all, 66 dropouts, is 2.7 m past splice 4 — the main splice box, where two fibers meet, for the same reason with a different cause.

    Then there are the spots **14.6 to 14.8 m before a collar** — N100, N120, N180 — where the OTDR reported nothing, and one 14 m *past* the up-collar of S180. The inventory README explains what should be there: each inclined hole's cable is 15 m of pigtail, 30 m down, 30 m up, 15 m of pigtail, spliced to the next, so a junction sits 15 m ahead of every hole, and the vertical cable ends the same way after S180. The OTDR put that last junction at 1554 m, fourteen metres off, and did not resolve the other three at all. The Brillouin dropouts find all four to within half a metre, from the strain data alone. Two spots, 25 m into the record in the telemetry cable and 6 m past S150, the inventory has nothing to say about.

    ### Cleaning them

    Once the dropouts are understood, `hampel_filter` along time replaces them: a rolling median over the sweeps, and any sample too far from it is set to that median. The window has to be long enough to make a median meaningful on a series with a few hundred points, and the threshold is in scaled deviations.
    """)
    return


@app.cell
def _(dss_local, plt):
    dss_clean = dss_local.hampel_filter(time=25, samples=True, threshold=12)
    _changed = dss_clean.data != dss_local.data
    print(
        f"hampel_filter replaced {_changed.sum()} samples ({100 * _changed.mean():.2f} %)"
    )

    # The worst channel, before and after.
    _worst = _changed.sum(axis=0).argmax()
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(
        dss_local.get_array("time"),
        dss_local.data[:, _worst],
        lw=0.8,
        label="as recorded",
    )
    _ax.plot(
        dss_clean.get_array("time"),
        dss_clean.data[:, _worst],
        lw=1.2,
        label="hampel_filter",
    )
    _ax.set_ylabel("µε")
    _ax.set_title(
        f"channel at {dss_local.get_array('distance')[_worst]:.1f} m on the instrument"
    )
    _ax.legend()
    _ax
    return (dss_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise

    1. Repeat the before-and-after comparison on `dss_clean` with plain means instead of medians. Does the answer change, and does the table of hot spots explain any channel where it does?
    2. The junctions found above sit 14.6 to 14.8 m before their collars. The README says 15 m of pigtail, and each survey is scaled onto its leg. Which of the two numbers would you now trust to place the next hole's splice, and what is your uncertainty?
    3. The dropout threshold is eight scaled deviations and the "hot spot" cut is five sweeps. Move both. Which of the spots in the table survive, and which were the threshold's doing?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - Permanent strain is the BOTDR's question, not the DAS's. Three hours after the blast against three hours before, no borehole moved above a 12 to 19 µε noise floor. `01_the_step_response.py` shows why the DAS could never have answered it.
    - Medians and the median absolute deviation survive a few wild values; means do not. `hampel_filter(time=..., samples=True)` removes the wild values once they are understood.
    - The wild values were information: Brillouin peak-hopping marks where two fiber states share one resolution cell, and it located four cable junctions, three of which the OTDR missed, to within half a metre.
    - `acquisition.distance_map.map_to_distance` puts an instrument's axis onto the path, where the inventory's components and labels are; `select(borehole=...)`, `leg` and `hole_depth` made every comparison above hole by hole, on both fibers, without an optical distance in sight.
    """)
    return


if __name__ == "__main__":
    app.run()
