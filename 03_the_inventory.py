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

    As we saw in the last notebook, there is a lot of important context to track for this particularly complex fiber-optic network. For example:

    1. A mapping from optical distance to physical distance.
    2. Coupling condition of the fiber
    3. Location of splices and connectors, and their associated optical loss

    On top of that, all of these can vary in time as the network is re-configured to meet monitoring needs or recover from a fiber break. Tracking all of this metadata is a challenge, and can easily devolve into a pile of structureless files that must be manually synchronized with the analysis code. This is a recipe for errors.

    This is exactly the problem DASCore's `Inventory` solves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is in it

    An inventory describes the *observing system*, for example:

    1) what components made up the measured optical path
    2) where the cable was located
    3) the interrogator settings
    4) important analysis labels (e.g. borehole N180)

    just to name a few.

    The inventory has approximately the same abstraction layers as the ubiquitous StationXML. In fact, DASCore's inventory is a superset of StationXML, which means in the future it will be possible to store both forms of metadata in a single container.

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
    Although little thought is often given to the location code in StationXML, the `OpticalPath` is DASCore Inventory's most important part. It contains four components, each supporting a wide range of detail but requiring very little to be useful.

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
    import numpy as np

    from galileo_2026 import get_data_path, get_image_path, get_inventory_path

    return dc, get_data_path, get_image_path, get_inventory_path, np


@app.cell
def _(dc, get_inventory_path):
    inv = dc.inventory(get_inventory_path())
    inv
    return (inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Like `Spool`, the `Inventory` has a number of visualization methods. The `timeline` method provides a quick visualization of the contents.
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
    `path` draws one optical path as lanes against optical distance: the components, coupling, and labels are layered into the figure.
    """)
    return


@app.cell
def _(inv):
    inv.viz.path("XM.MINE1.03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inventory-Spool integration

    On its own, the inventory is just a description. It becomes useful when coupled to the spool, but the degree and type of coupling depends on what you are trying to do.

    There are 3 options:

    - `attach_inventory`
    - `conform_to_inventory`
    - `enrich`
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
    This allows selecting on inventory-based information. For example, we can use `borehole=N180` to automatically restrict the patch `distance` dimension to that corresponding to `N180`.
    """)
    return


@app.cell
def _(spool):
    spool.select(borehole="N180")[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conforming

    `conform_to_inventory` keeps exactly the patches the inventory describes and drops the rest. In this case, the OTDR traces go because no acquisition claims them.
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

    `enrich` copies what the inventory knows onto each patch as it is loaded. These include both attributes and coordinates.
    """)
    return


@app.cell
def _(spool):
    enriched = spool.enrich()
    _new_patch = enriched.select(tag="DSS")[0]
    _new_patch
    return (enriched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the coordinates. `x`, `y` and `z` place every channel in space, `hole_depth` says how far down each segment sits, and `borehole`, `leg` and `drift` provide semantic meaning to the experiment. None of the inventory-related coordinates are dimensions. They are *associated* coordinates, sharing the `distance` dimension. That is what makes the next part work.
    """)
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
    Selecting on a label is not the same as selecting a distance range: the boreholes are scattered along the fiber, so `borehole="N*"` picks five separate stretches at once. The channels between them, the cable running along the drift, are not in any borehole, and thus drop out.

    It is also the same hole on two optical paths that have a different distance map. N180 sits at one distance on the DAS fiber and another on the Brillouin one, but the inventory handles this on its own.
    """)
    return


@app.cell
def _(enriched):
    # Show the distances are different for different optical paths.
    for _tag in ["DAS_LF", "DSS"]:
        _patch = enriched.select(tag=_tag, borehole="N180")[0]
        _distance = _patch.get_coord("distance")
        _dmin, _dmax = _distance.min(), _distance.max()

        print(f"{_tag:<7} N180 spans {_dmin:.1f} to {_dmax:.1f} m")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the ground

    With `x` and `y` on every channel, we can now map data-derived quantities to real space. `map_fiber` plots the outline of the fiber coloured by any per-channel value. For example, we can take the standard deviation of the LF DAS patch in the minute the blast occurs (in log scale).
    """)
    return


@app.cell
def _(enriched, np):
    _blast_time = ("2026-08-06T10:13", "2026-08-06T10:14")

    # Get the minute during the blast
    _blast_minute = enriched.select(tag="DAS_LF", time=_blast_time).chunk(
        time=..., conflict="drop", tolerance=10
    )[0]

    # Reduce time dimension by standard deviation
    shaking = _blast_minute.radians_to_strain().std("time").squeeze()

    # Plot in real space.
    shaking.viz.map_fiber(
        x="x", y="y", color=np.log10(shaking.data), cmap="magma"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise (3.1)**

    Take the low-frequency DAS data, keep only the downgoing leg of each borehole in the south drift, and work out how many channels that leaves. Then compare the mean strain in the top 5 m of the holes against the bottom 5 m — `hole_depth` is the coordinate you want.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - An `Inventory` describes the observing system: fiber, geometry, labels and coupling along each optical path, and the acquisitions that recorded it. A directory of CSV and YAML files is enough to author one.
    - `attach_inventory` lets a spool be queried by what the inventory knows, `conform_to_inventory` keeps only what it describes, and `enrich` copies attributes and per-channel coordinates onto each patch.
    - Once a patch carries `borehole`, `leg`, `x` and `y`, selections are made by what a channel is rather than where it falls along the fiber, and the same query works on any fiber the inventory describes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ///note

    There is quite a bit more to the inventory than shown here. For example, support for different epochs for both acquisitions and optical paths, custom coordinate reference systems, shared resources referenced by IDs (e.g. interrogators, cables). Read more about it in DASCore's official documentation.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
