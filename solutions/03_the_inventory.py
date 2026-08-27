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
    # An Introduction to the Inventory

    *This is the solutions copy: each exercise is followed by a worked answer.*

    As we saw in the last notebook, there is a lot of important context to track for this particularly complex fiber-optic network. For example, just to name a few:

    1. A mapping from optical distance to physical distance.
    2. Coupling condition of the fiber
    3. Location of splices and connectors, and their associated optical loss

    On top of that, all of these can vary in time as the network is re-configured to meet monitoring needs or recover from a break. Tracking all of this metadata is a challenge, and can easily devolve into a pile of excel files and notes that do not synchronize automatically with the analysis code.

    This is exactly the problem DASCore's `Inventory` solves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is in it

    An inventory describes the *observing system*, not the data: what fiber was measured, where it ran, and which interrogator settings produced which recordings. It contains approximately the same abstraction layers as the ubiquitous StationXML. In fact, DASCore's inventory is a superset of StationXML, which means in the future it will be possible to store both forms of metadata in a single container.

    The Inventory has the following form:
    """)
    return


@app.cell(hide_code=True)
def _(get_image_path, mo):
    mo.image(
        get_image_path("inventory_hierarchy.svg"),
        alt="The inventory's domains: point sensors on the left, fiber on the right, meeting at a shared network",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although little thought is often given to the location code in StationXML, the `OpticalPath` is DASCore Inventory's most important part. It contains four components, each supporting a wide range of detail but requiring very little.

    - `optical_components` is the fiber itself, part by part, including fiber segments, splices, connectors, and terminators.
    - `geometry` places the fiber in space, and is a *function* of distance: partial coverage is fine, gaps are simply unplaced.
    - `labels` say what a stretch of fiber **is**. In our case, which borehole, which drift.
    - `coupling` says how the cable meets the ground.
    """)
    return


@app.cell(hide_code=True)
def _(get_image_path, mo):
    mo.image(
        get_image_path("optical_path_concept.svg"),
        alt="Four tracks along one optical path: optical components, geometry, labels and coupling, each stated against distance along the fiber",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In practice, the inventory can conveniently be kept as a directory of CSV and YAML files. This deployment's Inventory is found in `galileo_2026/data/inventory`.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from galileo_2026 import get_data_path, get_image_path, get_inventory_path

    return dc, get_data_path, get_image_path, get_inventory_path, np, pd, plt


@app.cell
def _(dc, get_inventory_path):
    inv = dc.inventory(get_inventory_path())
    inv
    return (inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `timeline` method provides a quick visualization of the contents.
    """)
    return


@app.cell
def _(inv):
    inv.viz.timeline()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `map` method plots a basic map of the fiber segments which have defined geometry.
    """)
    return


@app.cell
def _(inv):
    inv.viz.map()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `path` draws one optical path as lanes against optical distance: the channels each acquisition lays on it, the components that give it its length, its coupling, and one lane per label group. This is the concept figure from the start of the notebook, drawn from the real thing.
    """)
    return


@app.cell
def _(inv):
    inv.viz.path("XM.MINE1.03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The DAS fiber carries two acquisitions, at full rate and decimated, and a geometry column can be drawn beneath the lanes, sharing the distance axis.
    """)
    return


@app.cell
def _(inv):
    inv.viz.path("XM.MINE1.04", columns="hole_depth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the numbers come from

    Every loss in `optical_components` points at the measurement it was read from, and that record says how it was measured: wavelength, pulse width, the time, and the trace file, which is in the archive.
    """)
    return


@app.cell
def _(inv):
    _path = next(
        p
        for p in inv.networks[0].fiber_arrays[0].optical_paths
        if p.location_code == "03"
    )
    splices = [
        c for c in _path.optical_components if c.object_type == "Splice"
    ]
    print(f"{len(splices)} splices, the first {splices[0].loss_db} dB")
    inv.get_resource(splices[0].loss_measurement)
    return (splices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So the check can be made: draw the trace and mark the splices on it. The OTDR was launched from about a kilometre ahead of the interrogator's zero, and the inventory README explains how the 1060 m between the two axes was pinned down. Not every splice made a step the trace resolved: the OTDR picked eight events on the sensing cable, and only those splices carry a `loss_db`. The rest were located from the Brillouin record, which we come to below.
    """)
    return


@app.cell
def _(dc, get_data_path, splices):
    otdr_patch = dc.read(get_data_path("OTDR") / "channel_3_otdr.sor")[0]
    _ax = otdr_patch.select(distance=(1000, 3700)).viz.wiggle()
    for _splice in splices:
        _ax.axvline(_splice.distance_min + 1060, color="C3", lw=1)
    _ax
    return (otdr_patch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise
    Zoom the trace in on `S170 top, up leg` (1618 m), one of the splices the OTDR did resolve, and describe what it does there: it is not a plain step down. Then use the path's `labels` and its components to work out which boreholes have a splice at each end and which have none for hundreds of metres, and why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Solution**
    """)
    return


@app.cell
def _(otdr_patch, plt, splices):
    # Three of the OTDR's events sit on a rise followed by a fall: a
    # "gainer", where the fiber after the splice scatters more light back
    # than the fiber before it, so the trace steps up before the splice's
    # loss takes it down. The loss the inventory records is the net of the
    # two. The splices are picked by name; the loss each carries is the
    # OTDR event's, carried to the splice the Brillouin record located.
    _names = ["S170 top, up leg", "S110 top, up leg", "N120 to N140 junction"]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    _by_name = {s.name: s for s in splices}
    for _ax, _splice in zip(_axes, [_by_name[n] for n in _names]):
        _x = _splice.distance_min + 1060
        _zoom = otdr_patch.select(distance=(_x - 60, _x + 60)).squeeze()
        _ax.plot(_zoom.get_array("distance"), _zoom.data)
        _ax.axvline(_x, color="C3")
        _ax.set_title(_splice.name)
        _ax.set_xlabel("distance [m]")
    _fig
    return


@app.cell
def _(inv, pd, splices):
    _path = next(
        p
        for p in inv.networks[0].fiber_arrays[0].optical_paths
        if p.location_code == "03"
    )
    _splice_at = [s.distance_min for s in splices]
    _rows = []
    for _label in _path.labels:
        if _label.group != "borehole":
            continue
        # A collar splice was picked to within a metre of the hole's end,
        # on either side of it, so allow that much overlap.
        _before = [d for d in _splice_at if d < _label.distance_min + 1]
        _after = [d for d in _splice_at if d > _label.distance_max - 1]
        _rows.append(
            {
                "borehole": _label.value,
                "start [m]": round(_label.distance_min, 1),
                "end [m]": round(_label.distance_max, 1),
                "splice before [m]": (
                    round(_label.distance_min - max(_before), 1)
                    if _before
                    else None
                ),
                "splice after [m]": (
                    round(min(_after) - _label.distance_max, 1)
                    if _after
                    else None
                ),
            }
        )
    # The five vertical holes (S100 to S180) share one unspliced cable, so
    # hundreds of metres separate them from any splice. Each inclined hole
    # is its own segment of cable, spliced at the collar on the way in and
    # out, so a splice sits within a metre of either end.
    pd.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inventory-Spool integration

    On its own, the inventory is just a description. It becomes useful when coupled to the spool, but the degree and type of coupling depends on the application. There are 3 options:

    - `attach_inventory` : Make the spool aware of the inventory; allow querying based on Inventory properties.
    - `conform_to_inventory` : Trim the spool to only contain patches with valid inventory metadata.
    - `enrich` : Copy applicable inventory information to extracted patches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Attaching
    `attach_inventory` is the first step. It makes the `Spool` aware of the inventory.
    """)
    return


@app.cell
def _(dc, get_data_path, inv):
    spool = dc.spool(get_data_path()).update().attach_inventory(inv)
    return (spool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conforming

    `conform_to_inventory` keeps exactly the patches the inventory describes and drops the rest. The OTDR traces go because no acquisition claims them.
    """)
    return


@app.cell
def _(spool):
    # "ignore" because this inventory deliberately covers only part
    # of the archive; the default would raise instead.
    conformed = spool.conform_to_inventory(on_unresolved="ignore")
    print(
        f"{len(spool)} patches in the archive, "
        f"{len(conformed)} described by the inventory"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Enriching

    `enrich` copies what the inventory knows onto each patch as it is pulled.
    Two different things arrive. **Attributes** are true of the patch as a whole, and come from the acquisition. **Coordinates** vary channel by channel, and come from the optical path's geometry and labels. This information can be convenient, but it can also slow down large workflows, so it is best to only enrich when needed for some processing.
    """)
    return


@app.cell
def _(spool):
    enriched = spool.enrich()
    _new_patch = enriched.select(tag="DSS")[0]
    print(_new_patch)
    return (enriched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note what those coordinates are. `x`, `y` and `z` place every channel in space, `hole_depth` says how far down its borehole it sits, and `borehole`, `leg` and `drift` say what it is. None of them are dimensions — they are *associated* coordinates, sharing the `distance` dimension. That is what makes the next part work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An enriched patch also knows which acquisition produced it, and through that acquisition's `distance_map` the inventory's components can be put on the patch's own axis. That is worth doing for the Brillouin data: the interrogator reports strain against a reference that differs from one piece of fiber to the next, so the median strain profile steps at every splice. This is how the splices on the sensing cable were located — three per inclined hole, at the collar going in, at the bottom and at the collar coming out, one where each pigtail meets the next segment, and three more on the runs between holes — while the five vertical holes on the unspliced cable show no step at all.
    """)
    return


@app.cell
def _(enriched, inv, np, splices):
    _dss = enriched.select(tag="DSS").chunk(time=None, conflict="drop")[0]
    _profile = _dss.median("time")
    # Put the splices on the instrument's axis, the reverse of what enrich did.
    _map = inv.resolve(_dss.attrs.acquisition_key).acquisition.distance_map
    _at = np.interp(
        [s.distance_min for s in splices],
        _map.distance,
        _map.instrument_distance,
    )
    _ax = _profile.select(distance=(1050, 2650)).viz.wiggle()
    for _x in _at:
        _ax.axvline(_x, color="C3", lw=0.8)
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting

    Once the metadata is attached to a spool, it can be used for convenient querying and spool reshaping.
    """)
    return


@app.cell
def _(enriched):
    das_spool = (
        enriched.select(tag="DAS_LF", borehole="N*", leg="down")
        .expand_by("borehole")
        # We just chunk with a large tolerance in order to force a merge; a few seconds of error in LF das is ok for us.
        .chunk(time=..., conflict="drop", tolerance=10)
    )
    return (das_spool,)


@app.cell
def _(das_spool):
    das_spool
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Selecting on a label is not the same as selecting a distance range: the boreholes are scattered along the fiber, so `borehole="N*"` picks five separate stretches at once. The channels between them — cable running along the drift — are simply not in any borehole, and drop out.

    It is also the same hole on two fibers. N180 sits at one distance on the DAS fiber and another on the Brillouin one, and neither number needs to be known:
    """)
    return


@app.cell
def _(enriched):
    for _tag in ["DAS_LF", "DSS"]:
        _distance = enriched.select(tag=_tag, borehole="N180")[0].get_coord(
            "distance"
        )
        print(
            f"{_tag:<7} N180 spans {_distance.min():.1f} to {_distance.max():.1f} m"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the ground

    With `x` and `y` on every channel, a patch can be drawn where it is. `map_fiber` plots the outline of the fiber coloured by any per-channel value; here, the standard deviation of strain over the minute of the blast, on a log scale.
    """)
    return


@app.cell
def _(enriched, np):
    _blast_minute = enriched.select(
        tag="DAS_LF", time=("2026-08-06T10:13", "2026-08-06T10:14")
    ).chunk(time=..., conflict="drop", tolerance=10)[0]
    shaking = _blast_minute.radians_to_strain().std("time").squeeze()
    shaking.viz.map_fiber(
        x="x", y="y", color=np.log10(shaking.data), cmap="magma"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise

    Take the low-frequency DAS data, keep only the downgoing leg of each borehole in the south drift, and work out how many channels that leaves. Then compare the mean strain in the top 5 m of the holes against the bottom 5 m — `hole_depth` is the coordinate you want.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Solution**
    """)
    return


@app.cell
def _(enriched, pd):
    south_down = (
        enriched.select(tag="DAS_LF", drift="south", leg="down")
        .expand_by("borehole")
        .chunk(time=..., conflict="drop", tolerance=10)
    )
    _rows = []
    for _hole_patch in south_down:
        _strain = _hole_patch.radians_to_strain()
        _top = _strain.select(hole_depth=(0, 5)).mean().data.squeeze()
        _bottom = _strain.select(hole_depth=(25, 30)).mean().data.squeeze()
        _rows.append(
            {
                "borehole": _hole_patch.get_array("borehole")[0],
                "channels": _hole_patch.shape[1],
                "top 5 m [µε]": float(_top) * 1e6,
                "bottom 5 m [µε]": float(_bottom) * 1e6,
            }
        )
    _table = pd.DataFrame(_rows)
    print(f"{len(_table)} holes, {_table['channels'].sum()} channels")
    # The top of every hole strains ten to a hundred times more than the
    # bottom over these 22 minutes: it is the end nearest the open drift.
    _table
    return (south_down,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - An `Inventory` describes the observing system, not the data: fiber, geometry, labels and coupling along each optical path, and the acquisitions that recorded it. A directory of CSV and YAML files is enough to author one.
    - A measured value such as a splice loss points at the record of how it was measured, and `viz.path` draws the whole path against distance.
    - `attach_inventory` lets a spool be queried by what the inventory knows, `conform_to_inventory` keeps only what it describes, and `enrich` copies attributes and per-channel coordinates onto each patch.
    - Once a patch carries `borehole`, `leg`, `x` and `y`, selections are made by what a channel is rather than where it falls along the fiber, and the same query works on any fiber the inventory describes.
    """)
    return


if __name__ == "__main__":
    app.run()
