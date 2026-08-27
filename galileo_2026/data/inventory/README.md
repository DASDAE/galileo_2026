# Inventory

A DASCore inventory describing the observing system that recorded the data in `../fiber`. It is written in DASCore's authoring-directory format, so it loads as it stands:

```python
import dascore as dc

from galileo_2026 import get_inventory_path

inventory = dc.inventory(get_inventory_path())
```

> `dc.inventory` is new. It needs a DASCore newer than 0.1.20, which is why `pyproject.toml` installs the `dev` branch rather than a release. `03_the_inventory.py` is built on it.

## What it says

Strain sensing cable is grouted into fourteen boreholes across two drifts, south and north, each about 30 m long. In each hole the fiber runs down and back up, so one hole is two legs meeting at the bottom.

Two fibers run the length of the network. One feeds a Brillouin interrogator in the instrument room, reached over a kilometre of telemetry cable. The other feeds a DAS interrogator, connected at the main splice box that telemetry passes through on its way to the sensing cable — so it never sees the kilometre of telemetry ahead of the box, and its distances start there rather than at the instrument room.

Two cables do the sensing. The five vertical holes, all in the south drift, were done first with a single unspliced run carrying one tight-buffered and one loose-buffered fiber. The nine inclined holes came later on a four-fiber cable supplied in segments cut to length, each with a splice enclosure at either end. Ahead of both is the telemetry cable, which carries no strain and simply gets the light from the instrument room to the gallery.

```
inventory.yaml                          the coordinate reference system
fiber_arrays/XM.MINE1/attrs.yaml        the array
fiber_arrays/XM.MINE1/path.03/          the Brillouin fiber, site channel 3
fiber_arrays/XM.MINE1/path.04/          the DAS fiber, site channel 4
acquisitions/                           one file per interrogator setup
resources/                              interrogators, cables, OTDR record
```

The Brillouin path carries four tracks along the fiber, each a CSV a field crew could keep in a spreadsheet:

| File | Contents |
| --- | --- |
| `optical_components.csv` | The fiber itself, part by part: runs of fiber, the splices along it, the loss the OTDR reported where it resolved one, and the fiber end. |
| `geometry.csv` | Where the fiber is. Each borehole contributes a `down` and an `up` segment carrying `x`, `y`, `z` and `hole_depth`. |
| `labels.csv` | What each stretch is: `borehole` names the hole, `leg` says down or up, `drift` says south or north. |
| `coupling.csv` | How the cable meets the ground — grouted in the boreholes, and nothing claimed for the fiber between them. |

## The OTDR traces

`../fiber/tag=OTDR/` holds two Bellcore SOR traces, `channel_3_otdr.sor` and `channel_4_otdr.sor`. They are traces of the two fibers — channel 3 the Brillouin one, channel 4 the DAS one. Their key events are what each path's `optical_components.csv` is built from: a non-reflective loss event becomes a `Splice` carrying the loss the trace reported, a reflection part way along becomes a `Connector` carrying both its loss and its reflectance, and the reflection each trace ends on becomes the fiber end — except on the Brillouin path, whose end is placed where its own record loses the fiber, 12 m past the trace's pick, as the caveats below explain.

Both traces were shot from about a kilometre before the interrogator's zero, so both open with fiber that belongs to neither path; the 1060 m shift below is what puts their events on the axis the data uses. After that shift both paths keep the telemetry run, because both are stated from the instrument room outwards. The DAS launches part way along that run, at the splice box, so most of it is fiber it never sees, but that is a fact about the acquisition rather than the path, and the acquisition's `distance_map` is where it is recorded.

The trace detects loss, not hardware. Its events are automatic `loss/drop/gain` picks, and on the telemetry cable those picks are all the inventory has, so a `Splice` there means "a point that loses light", which is the inventory's vocabulary for it. On the sensing cable the OTDR resolved far fewer splices than there are. Each inclined hole is a segment of the four-fiber cable with a splice at the collar on the way in, one at the bottom where the down leg loops into the up leg, one at the collar on the way out, and one about 15 m past the hole where its pigtail meets the next segment — except the last hole, whose pigtail runs out at the fiber end. With three more on the runs between holes, where the vertical cable meets the first inclined segment and twice on the way from the south drift to the north, that is thirty-eight splices on the sensing cable, of which the trace picked eight on channel 3 and seven on channel 4. The Brillouin record sees them all: the interrogator reports strain against a reference frequency that differs from one piece of fiber to the next, so the median strain profile of the DSS data steps at every splice, by thousands of microstrain at most of them and a few tens at the weakest, and shows no step at all along the five vertical holes on the unspliced cable. The splices on the sensing cable are located from those steps, on the Brillouin path directly and on the DAS path by transferring each one through the borehole boundaries the two paths share, since the DAS record does not resolve them. Each OTDR event on that cable lies within 8 m of a located splice, and its loss is carried to that splice; the rest carry none. Every loss still points at `otdr-channel-3` or `otdr-channel-4`, an `OpticalMeasurement` record holding the wavelength, pulse width and time it was obtained under, which in turn points at the trace file. That is the shape the inventory keeps throughout: a measured number never travels without the record saying how it was measured.

```python
path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
splices = [c for c in path.optical_components if c.object_type == "Splice"]
print([c.loss_db for c in splices])
```

## Using it

An acquisition is one interrogator setup, keyed `network.fiber_array.location.acquisition`. The shipped files carry no `acquisition_key`, since the recorders did not write one, so the key is what connects a patch to the inventory — and, as the next section shows, the directory names are what supply it:

| Key | Produced | Fiber |
| --- | --- | --- |
| `XM.MINE1.03.WSF` | the `tag=DSS` data | site channel 3 |
| `XM.MINE1.04.FSF` | the `tag=DAS` blast recording | site channel 4 |
| `XM.MINE1.04.MSF` | the `tag=DAS_LF` data | site channel 4 |

The third token is the location code, which names the fiber, and it follows the site's own channel numbering. The last is the acquisition: `FSF` and `MSF` are two setups on one fiber, one at full rate and one decimated to 5 Hz, both written continuously over the same week.

Each data directory states its key in its own name — `tag=DSS__acquisition_key=XM.MINE1.03.WSF` — which DASCore reads as a patch attribute, so a patch arrives already knowing which acquisition produced it and nothing needs to be set by hand:

```python
from galileo_2026 import get_data_path

patch = dc.spool(get_data_path()).select(tag="DSS")[0]
patch = patch.enrich(inventory)
```

The two OTDR traces have no acquisition key: they are characterisations of the fiber rather than recordings of the ground, and they enter the inventory as the measurement records the splice losses point back to.

`enrich` projects `x`, `y`, `z`, `hole_depth`, `borehole`, `leg` and `drift` onto the patch's distance axis, so a channel can be selected by what it is rather than by where it falls along the fiber. Both interrogators resolve the same way: the blast recording lands on the five northern holes, and the low frequency data spans all fourteen.

## Scope and caveats

- **Both paths are stated in optical distance**, the metric their OTDR traces use, at a group index of 1.46832. Neither interrogator's own axis is that metric, so each acquisition's `distance_map` carries the conversion. The Brillouin instrument bins at 0.1 m, which is a 1 GHz digitiser assuming an index of 1.49896, so its metres run 2.09% short and its map has that slope. The DAS shares the OTDR's index, so its map is a pure offset. Before this was worked out, the borehole positions on both paths were in the Brillouin instrument's metric while the components around them were in optical distance, and the two drifted apart by about 2% along the fiber.
- **Only the shipped data is described.** The real deployment ran through several build-out phases on two interrogator channels; this inventory keeps the one configuration the shipped files came from.
- **The OTDR launched about a kilometre ahead of the interrogator's zero**, so its distances are shifted by 1060 m to put them on the axis the data uses. The two instruments were never tied together by survey, but the shift is not a guess. One unspliced cable instruments all five vertical holes, so no event may fall across their span, which brackets the offset to 1024–1075 m. Within that, each inclined hole is a 90 m cable — 15 m of pigtail, 30 m down, 30 m up, 15 m of pigtail — spliced to the next, so a junction sits 15 m past every hole but the last. Fitting the trace's events onto those junctions gives 1060 m. The splices the Brillouin record then located confirm it: all eight of the trace's events on the sensing cable fall within 8 m of one of them. The end reflection is the one pick that does not fit, 12 m short of where the record runs out, so the path's end is taken from the record rather than the trace. Treat the trace's own event positions as good to some tens of metres. The loss values are as the trace reported them, automatically picked and carrying no stated uncertainty.
- **The splices on the sensing cable are located from the DSS record**, by `scripts/locate_splices.py` in the repository root, which rewrites that section of both paths' `optical_components.csv` and is run after any rebuild of this directory. It picks the collar, bottom and pigtail-junction steps of each inclined hole on the median strain profile, plus the three steps on the runs between holes and the fiber end, and carries the splices to the DAS path through the borehole boundaries, matched hole by hole. The collar and bottom positions are good to about a metre on the Brillouin path; the junctions and everything on the DAS path to a few metres.
- **One further hole was drilled but never instrumented**, so it carries no fiber and appears nowhere in this inventory.
- **The Brillouin path ends at 2607.7 m**, where its record loses the fiber, 15 m past N180's collar like every other pigtail. Its trace's automatic end pick sits 12 m short of that, so the end keeps the reflectance the trace reported but takes its position from the record, as the splices do. The DAS path's trace puts its end at 2608 m too, and keeps it.
- **Times are UTC**, written without a designator, which is the format's convention throughout.
- **Coordinates carry no georeference.** `x`, `y`, `z` are metres in a local frame whose origin is arbitrary: the surveyed positions with a fixed offset subtracted, and nothing else done to them. Distances, depths, bearings and the shape of the array are all faithful, so the geometry is fully usable and `docs/geometry.html` sits in this same frame — but the offset is not recorded here, so nothing places any of it on a map. The operator and the site are not named.
- **Each survey is scaled onto its leg.** The optical half-span and the surveyed hole length disagree by up to a metre and a half; they are independent measurements and the turnaround itself consumes fiber.
- **The DAS path is placed from the DAS data itself.** Its fiber follows the same route as the Brillouin one and carries the same surveyed `x`, `y`, `z`, but not the same distances: the boreholes are grouted, so they show as quiet bands in the DAS record, and each hole's position along `path.04` is picked from those bands rather than inherited. The interrogator connects at the main splice box through about 11.3 m of patch cable, so its channel zero sits before the box and the `distance_map` starts at the box, leaving the channels on that lead placed nowhere. The Sintela and the OTDR convert time of flight with the same group index, so one interrogator metre is one path metre and the map is a pure offset. Each hole is then checked by sliding its labelled span along the fiber and finding where the mean per-channel variance inside it is lowest, on a quiet ten-minute window of the DAS_LF record; all fourteen sit within 1.1 m rms of that minimum, 2.2 m at worst. Comparing the labels against the picks they were built from proves nothing, so it is the record they are checked against.
- **Both interrogators outrun their fiber.** The Brillouin fiber ends at 2607.7 m and the DAS one at 2608 m, but the Brillouin acquisition was configured out to 2763 m of path and the DAS reaches 2630 m, once each is mapped on. Channels past the end simply get nothing.
