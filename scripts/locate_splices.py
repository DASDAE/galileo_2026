"""
Locate the sensing-cable splices from the Brillouin record.

The OTDR traces the inventory's ``optical_components.csv`` files were built
from resolved only a handful of the splices on the sensing cable. Each
inclined hole is a segment of four-fiber cable with a splice at the collar
on the way in, one at the bottom where the down leg loops into the up leg,
one at the collar on the way out, and one about 15 m past the hole where its
pigtail meets the next segment. A Brillouin interrogator reports a different
reference frequency on each side of a splice, so the median strain profile
of the DSS record steps at every one of them, and the five vertical holes on
the unspliced cable step nowhere, which is the control.

This script picks those steps on the Brillouin path, transfers them to the
DAS path through the borehole boundaries the two paths share, and rewrites
the sensing-cable section of both paths' ``optical_components.csv``. The
telemetry section, up to the splice where the sensing cable begins, is the
OTDR's and is kept as it stands. The loss the OTDR reported at each of its
events is carried to the nearest located splice. The Brillouin path's end
is moved to where the record loses the fiber, which its trace's automatic
end pick sits 12 m short of; the DAS path keeps the end its own trace saw.

Run from a checkout after any rebuild of the inventory::

    uv run --no-sync python scripts/locate_splices.py
"""

from __future__ import annotations

import dascore as dc
import numpy as np
import pandas as pd

from galileo_2026 import get_data_path, get_inventory_path

# Half-width of the windows the step detector compares, in samples of the
# 0.1 m Brillouin bin: one metre each side.
HALF_WINDOW = 10
# Search windows around what the labels say, in metres of optical distance.
COLLAR_TOLERANCE = 5.0
BOTTOM_TOLERANCE = 6.0
JUNCTION_WINDOW = (5.0, 25.0)
# Steps outside the targeted windows count only when this large: the
# in-hole strain contrasts on the vertical cable stay under it.
FREE_STEP_MIN = 800.0
# An OTDR event is carried to a located splice no further away than this,
# and only when the runner-up is clearly further off.
LOSS_CARRY_MAX = 8.0
LOSS_CARRY_MARGIN = 2.0
INCLINED_CABLE = "strain-cable-inclined"
COLUMNS = [
    "object_type",
    "distance_min",
    "distance_max",
    "name",
    "container",
    "loss_db",
    "loss_measurement",
    "reflectance_db",
    "reflectance_measurement",
]


def median_profile(inventory):
    """
    Return the Brillouin path's median strain profile against its distance.

    Every DSS file is stacked and the median taken over time, then the
    instrument axis is mapped onto the path through the acquisition's
    ``distance_map``.
    """
    patches = list(dc.spool(get_data_path()).update().select(tag="DSS"))
    data = np.concatenate(
        [p.transpose("time", "distance").data for p in patches]
    )
    profile = np.median(data, axis=0)
    instrument = patches[0].get_coord("distance").values
    context = inventory.resolve(patches[0].attrs.acquisition_key)
    dmap = context.acquisition.distance_map
    distance = np.interp(instrument, dmap.instrument_distance, dmap.distance)
    return distance, profile


def step_profile(profile):
    """Median of the metre ahead minus the median of the metre behind."""
    steps = np.full_like(profile, np.nan)
    for i in range(HALF_WINDOW, len(profile) - HALF_WINDOW):
        ahead = np.median(profile[i : i + HALF_WINDOW])
        behind = np.median(profile[i - HALF_WINDOW : i])
        steps[i] = ahead - behind
    return steps


def pick(distance, steps, low, high):
    """Return the distance of the largest step in ``(low, high)``."""
    inside = (distance > low) & (distance < high)
    index = np.flatnonzero(inside)[np.nanargmax(np.abs(steps[inside]))]
    return float(distance[index])


def hole_table(path):
    """Return one row per borehole, in fiber order, with its leg split."""
    holes = [x for x in path.labels if x.group == "borehole"]
    downs = [x for x in path.labels if x.group == "leg" and x.value == "down"]
    rows = []
    for hole in sorted(holes, key=lambda x: x.distance_min):
        split = next(
            x.distance_max
            for x in downs
            if x.distance_min == hole.distance_min
        )
        rows.append(
            {
                "hole": hole.value,
                "start": hole.distance_min,
                "bottom": split,
                "end": hole.distance_max,
            }
        )
    return pd.DataFrame(rows)


def inclined_holes(path, holes):
    """Return the names of the holes on the segmented four-fiber cable."""
    segments = [
        x
        for x in path.optical_components
        if x.object_type == "FiberSegment" and x.container == INCLINED_CABLE
    ]
    on_cable = [
        row.hole
        for row in holes.itertuples()
        if any(s.distance_min <= row.start < s.distance_max for s in segments)
    ]
    return on_cable


def locate(distance, steps, holes, inclined):
    """
    Pick every splice on the sensing cable, in fiber order, and its end.

    Returns a DataFrame of ``distance`` and ``name``, and the distance at
    which the record loses the fiber. Targeted picks come from the windows
    the labels imply for each inclined hole; free picks are the large
    steps left over between holes, which are the junctions of the cable
    runs that carry no hole.
    """
    rows = []

    def add(name, low, high):
        rows.append((pick(distance, steps, low, high), name))

    for i, row in enumerate(holes.itertuples()):
        if row.hole not in inclined:
            continue
        collar, deep = COLLAR_TOLERANCE, BOTTOM_TOLERANCE
        add(
            f"{row.hole} top, down leg",
            row.start - collar,
            row.start + collar,
        )
        add(
            f"{row.hole} bottom",
            row.bottom - deep,
            row.bottom + deep,
        )
        add(
            f"{row.hole} top, up leg",
            row.end - collar,
            row.end + collar,
        )
        if i + 1 == len(holes):
            # The last pigtail runs out at the fiber end, not into a splice:
            # the largest step of all, where the record loses the fiber.
            fiber_end = pick(
                distance,
                steps,
                row.end + JUNCTION_WINDOW[0],
                row.end + JUNCTION_WINDOW[1],
            )
            continue
        add(
            f"{row.hole} to {holes.hole.iloc[i + 1]} junction",
            row.end + JUNCTION_WINDOW[0],
            row.end + JUNCTION_WINDOW[1],
        )
    picks = pd.DataFrame(rows, columns=["distance", "name"])

    # Everything else large enough, from the first hole to the fiber end,
    # outside the holes and clear of what was just picked.
    free = np.abs(steps) > FREE_STEP_MIN
    free &= distance > holes.start.min()
    free &= distance < fiber_end - COLLAR_TOLERANCE
    for row in holes.itertuples():
        free &= ~((distance >= row.start) & (distance <= row.end))
    for d in picks.distance:
        free &= np.abs(distance - d) > COLLAR_TOLERANCE
    found = []
    for index in np.flatnonzero(free):
        if found and index - found[-1][-1] <= 3 * HALF_WINDOW:
            found[-1].append(index)
        else:
            found.append([index])
    extra = [
        float(distance[c[int(np.argmax(np.abs(steps[c])))]]) for c in found
    ]
    first_inclined = holes[holes.hole.isin(inclined)].start.min()
    last_vertical = holes[~holes.hole.isin(inclined)].end.max()
    lead = [d for d in extra if last_vertical < d < first_inclined]
    between = [d for d in extra if d > first_inclined]
    assert len(lead) == 1, f"one junction ahead of the inclined cable: {lead}"
    assert extra == lead + between, f"unexpected steps: {extra}"
    names = ["vertical to inclined cable junction"] + [
        f"south to north drift splice {n}" for n in range(1, len(between) + 1)
    ]
    for d, name in zip(extra, names, strict=True):
        picks.loc[len(picks)] = [d, name]
    picks = picks.sort_values("distance").reset_index(drop=True)
    # The picture the cable description implies: four splices per inclined
    # hole less the junction the last one has no partner for, plus the run
    # steps. Anything else means the record or the labels have changed.
    expected = 4 * len(inclined) - 1 + len(extra)
    assert len(picks) == expected, f"{len(picks)} picks, not {expected}"
    assert picks.distance.is_unique and picks.distance.is_monotonic_increasing
    return picks, fiber_end


def transfer(picks, holes_from, holes_to):
    """Map picks onto another path through the borehole boundaries."""
    assert holes_from.hole.tolist() == holes_to.hole.tolist(), (
        "the two paths must carry the same holes in the same order"
    )
    bounds_from = np.r_[holes_from.start, holes_from.end]
    bounds_to = np.r_[holes_to.start, holes_to.end]
    order = np.argsort(bounds_from)
    bounds_from, bounds_to = bounds_from[order], bounds_to[order]
    assert (np.diff(bounds_to) > 0).all(), "hole order differs between paths"
    assert picks.distance.between(bounds_from[0], bounds_from[-1]).all()
    out = picks.copy()
    out["distance"] = np.interp(picks.distance, bounds_from, bounds_to)
    return out


def rewrite(csv_path, picks, junction_name, fiber_end=None):
    """
    Rebuild the sensing-cable section of one ``optical_components.csv``.

    Rows up to and including the splice where the sensing cable begins are
    kept verbatim, and everything after is rebuilt from the picks. The
    terminator keeps its reflectance and moves to ``fiber_end`` when one
    is given. A splice being dropped that carries a loss is an OTDR
    event, and its loss goes to the nearest pick; on a rerun that is the
    same splice, so running twice writes the same file. Returns the loss
    carried from each event.
    """
    raw = csv_path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    is_segment = table.object_type == "FiberSegment"
    sensing = table.index[is_segment & (table.container != "telemetry-cable")]
    head = table.iloc[: sensing[0]]
    terminator = table[table.object_type == "Terminator"].iloc[0].copy()
    if fiber_end is not None:
        terminator["distance_min"] = _fmt(round(fiber_end, 1))
    end = float(terminator.distance_min)
    dropped = table.iloc[sensing[0] :]
    dropped = dropped[
        (dropped.object_type == "Splice") & (dropped.loss_db != "")
    ]

    picks = picks.copy()
    picks["distance"] = picks.distance.round(1)
    picks["loss_db"] = ""
    picks["loss_measurement"] = ""
    carried = []
    for event in dropped.itertuples():
        gaps = np.abs(picks.distance - float(event.distance_min))
        nearest, runner_up = gaps.nsmallest(2).index
        assert gaps[nearest] <= LOSS_CARRY_MAX, f"{event.name} is orphaned"
        assert gaps[runner_up] - gaps[nearest] >= LOSS_CARRY_MARGIN, (
            f"{event.name} is ambiguous"
        )
        assert not picks.loss_db[nearest], f"{event.name} doubles up"
        picks.loc[nearest, "loss_db"] = event.loss_db
        picks.loc[nearest, "loss_measurement"] = event.loss_measurement
        carried.append(
            (
                event.name,
                float(event.distance_min),
                picks.name[nearest],
                picks.distance[nearest],
                float(gaps[nearest]),
            )
        )

    rows = []
    start = float(head.distance_min.iloc[-1])
    container = table.container[sensing[0]]
    for splice in picks.itertuples():
        rows.append(_segment(start, splice.distance, container))
        rows.append(
            {
                "object_type": "Splice",
                "distance_min": _fmt(splice.distance),
                "name": splice.name,
                "loss_db": splice.loss_db,
                "loss_measurement": splice.loss_measurement,
            }
        )
        start = splice.distance
        if splice.name == junction_name:
            container = INCLINED_CABLE
    rows.append(_segment(start, end, container))
    body = pd.DataFrame(rows, columns=COLUMNS).fillna("")
    out = pd.concat([head, body, terminator.to_frame().T], ignore_index=True)
    out[COLUMNS].to_csv(csv_path, index=False, lineterminator=newline)
    return pd.DataFrame(
        carried, columns=["otdr event", "at", "carried to", "at ", "gap"]
    )


def _segment(start, end, container):
    return {
        "object_type": "FiberSegment",
        "distance_min": _fmt(start),
        "distance_max": _fmt(end),
        "name": f"fiber to {_fmt(end)} m",
        "container": container,
    }


def _fmt(value):
    """Write whole metres as the OTDR rows do and tenths otherwise."""
    return f"{value:g}" if float(value).is_integer() else f"{value:.1f}"


def main():
    """Pick, transfer, rewrite, and report."""
    inventory = dc.inventory(get_inventory_path())
    network = inventory.networks[0]
    array = network.fiber_arrays[0]
    paths = {p.location_code: p for p in array.optical_paths}
    brillouin, das = paths["03"], paths["04"]

    distance, profile = median_profile(inventory)
    steps = step_profile(profile)
    holes = hole_table(brillouin)
    inclined = inclined_holes(brillouin, holes)
    picks, fiber_end = locate(distance, steps, holes, inclined)
    picks_das = transfer(picks, holes, hole_table(das))

    arrays = get_inventory_path() / "fiber_arrays"
    base = arrays / f"{network.code}.{array.code}"
    junction = "vertical to inclined cable junction"
    # The DAS path keeps the end its own trace saw; the Brillouin record
    # measured this one directly.
    ends = {"03": fiber_end, "04": None}
    for code, table in [("03", picks), ("04", picks_das)]:
        carried = rewrite(
            base / f"path.{code}" / "optical_components.csv",
            table,
            junction,
            ends[code],
        )
        print(
            f"\npath.{code}: {len(table)} splices located; OTDR loss carried"
        )
        print(carried.to_string(index=False, float_format="{:.1f}".format))

    both = picks.assign(das=picks_das.distance).rename(
        columns={"distance": "brillouin"}
    )
    print(f"\nBrillouin path ends at {fiber_end:.1f} m")
    print("\nLocated splices, optical distance in m on each path\n")
    print(both[["name", "brillouin", "das"]].to_string(index=False))


if __name__ == "__main__":
    main()
