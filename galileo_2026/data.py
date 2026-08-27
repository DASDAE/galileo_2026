"""
Locate the tutorial data.

The data ships inside this package, under ``galileo_2026/data``, so that
installing the repository -- ``pip install git+https://github.com/DASDAE/galileo_2026``,
which is what every notebook header asks for -- brings the recordings along
with the code. That is what makes the notebooks work on molab, where a
notebook arrives as a single file with no clone behind it.

Reading data out of ``site-packages`` is no way to run a tutorial, though, so
an installed package unpacks its data to ``galileo_2026_data`` beside the
notebook the first time it is asked for one. A repository checkout is used
where it stands: there is nothing to unpack.

The figures the notebooks draw ship alongside, under ``galileo_2026/images``,
and are read straight from wherever the package landed: marimo inlines them, so
they need neither unpacking nor a network.

The recordings live in directories whose names carry their own metadata --
``tag=DSS__acquisition_key=XM.MINE1.03.WSF`` -- which DASCore reads as patch
attributes. Spool the ``fiber`` directory and every patch arrives knowing both
its tag and which inventory acquisition produced it, so one spool covers the
whole archive and resolves against the inventory without any further help.
"""

from __future__ import annotations

import os
import shutil
import warnings
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

#: The copy that ships in the package, which is also the copy in a checkout.
PACKAGED_DATA_DIR = Path(__file__).resolve().parent / "data"

#: The figures the notebooks draw, which ship in the package too.
PACKAGED_IMAGE_DIR = PACKAGED_DATA_DIR.parent / "images"

#: A checkout has the project file one level above the package; an install
#: has whatever else happens to sit in ``site-packages``.
_PROJECT_FILE = PACKAGED_DATA_DIR.parent.parent / "pyproject.toml"

#: Where an installed package unpacks its data, relative to the notebook.
UNPACK_DIR_NAME = "galileo_2026_data"

#: Set this to put the data somewhere else, or to point at a copy you already
#: have. It is used as given, and is not unpacked into.
ENV_VAR = "GALILEO_2026_DATA"

#: Written at the end of a copy, and only then, so that a copy interrupted
#: half way is repeated rather than trusted. It names the version it came
#: from, so a newer install refreshes what an older one left behind.
_MARKER_NAME = ".unpacked"


def _stamp() -> str:
    """What a finished copy of this version's data is marked with."""
    try:
        return f"galileo_2026 {version('galileo-2026')}"
    except PackageNotFoundError:  # only if the package is not installed
        return "galileo_2026 unknown"


def _unpack() -> Path:
    """Copy the packaged data beside the notebook, once."""
    target = Path.cwd() / UNPACK_DIR_NAME
    marker = target / _MARKER_NAME
    stamp = _stamp()
    if marker.is_file() and marker.read_text() == stamp:
        return target
    try:
        shutil.copytree(PACKAGED_DATA_DIR, target, dirs_exist_ok=True)
        marker.write_text(stamp)
    except OSError as error:
        # A read-only or full working directory is no reason to stop: the
        # packaged copy is readable, it is only awkward to go looking in.
        warnings.warn(
            f"Could not unpack the tutorial data into {target} ({error}); "
            f"reading it from the package instead. Set {ENV_VAR} to put it "
            "somewhere writable.",
            stacklevel=3,
        )
        return PACKAGED_DATA_DIR
    return target


@cache
def get_data_dir() -> Path:
    """
    Return the directory holding every part of the tutorial data.

    An override wins; a checkout is used where it stands; anything else is an
    installed package, whose data is unpacked beside the notebook.

    Resolved once per session, so that a notebook which changes directory
    mid-run keeps reading the data it started with -- and does not copy it a
    second time.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    if _PROJECT_FILE.is_file():
        return PACKAGED_DATA_DIR
    return _unpack()


def get_inventory_path() -> Path:
    """
    Return the path to the tutorial's DASCore inventory.

    It is resolved exactly as the recordings are -- see ``get_data_dir`` --
    so the inventory and the patches it enriches always come from the same
    copy of the data.
    """
    return get_data_dir() / "inventory"


def get_image_path(name: str) -> Path:
    """
    Return the path to one of the figures the notebooks show.

    They ship inside the package, beside the data and for the same reason:
    a notebook on molab has no repository to read them from, and marimo
    inlines whatever ``mo.image`` is handed, so nothing has to be unpacked
    and no notebook needs a network to draw its figures.
    """
    path = PACKAGED_IMAGE_DIR / name
    if not path.is_file():
        have = sorted(x.name for x in PACKAGED_IMAGE_DIR.glob("*"))
        msg = (
            f"No figure named {name!r} in {PACKAGED_IMAGE_DIR}."
            f" Try: {', '.join(have)}"
        )
        raise FileNotFoundError(msg)
    return path


def _tag_of(directory: Path) -> str:
    """The tag a data directory states, ignoring anything it states after."""
    return directory.name.split("__")[0].partition("=")[2]


def get_data_path(tag: str | None = None) -> Path:
    """
    Return the path to the tutorial data.

    Parameters
    ----------
    tag
        The tag of one dataset, eg ``"DSS"``, ``"DAS"``, ``"DAS_LF"``, or
        ``"OTDR"``. If None, return the directory holding all of them, which
        is what you want to pass to ``dc.spool``.

    Returns
    -------
    The requested directory.
    """
    fiber_dir = get_data_dir() / "fiber"
    if tag is None:
        return fiber_dir
    for directory in sorted(fiber_dir.glob("tag=*")):
        if directory.is_dir() and _tag_of(directory) == tag:
            return directory
    options = sorted(_tag_of(x) for x in fiber_dir.glob("tag=*") if x.is_dir())
    hint = (
        f" Available tags are: {', '.join(options)}."
        if options
        else f" Nothing is there; is {ENV_VAR} set to the wrong place?"
    )
    msg = f"No tutorial data tagged {tag!r} under {fiber_dir}.{hint}"
    raise FileNotFoundError(msg)
