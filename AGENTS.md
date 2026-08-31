# Agent Instructions

## Writing

Do not hard-wrap Markdown prose. One paragraph is one line, however long; keep normal blank-line separation between blocks. This applies everywhere — the readme, notebook markdown cells, `galileo_2026/data/inventory/README.md`, scratch notes, commit messages, and anything posted to GitHub such as issues and PR bodies. Tables, lists and code fences keep their own line structure.

## This repository

The tutorial for the 14th EGU Galileo Conference, Aussois, 31 Aug – 4 Sep 2026.

Data lives under `galileo_2026/data/fiber/` in directories whose names carry their own metadata, hive style: `tag=DSS__acquisition_key=XM.MINE1.03.WSF`. DASCore reads those as patch attributes, so `dc.spool(get_data_path())` gives every patch its tag and its inventory identity without any further help. Use `get_data_path()` and `get_inventory_path()` rather than hard-coding paths; both resolve through `get_data_dir()`, so the recordings and the inventory always come from one copy.

`galileo_2026/data/inventory/` is a DASCore inventory in authoring-directory format. It is generated from private operator metadata, so do not hand-edit the CSVs; they are rebuilt by a script kept outside this repo. The one exception is the sensing-cable section of each path's `optical_components.csv`, which `scripts/locate_splices.py` rewrites from the DSS record; run it after any rebuild, and change the picking there rather than in the CSVs.

Coordinates in `galileo_2026/data/inventory/` and `galileo_2026/data/docs/geometry.html` are in a local frame with a fixed offset subtracted. That offset is not recorded in this repository and must never be committed to it. Anything derived from the operator's raw survey — plots included — belongs in `.gitignore` until it has been moved into the local frame.

## Environment

DASCore comes from the `dev` branch as a direct reference, because the inventory, the SOR reader and hive-style directory attributes are not in any release yet. It is written as a PEP 508 URL rather than `[tool.uv.sources]` so that pip and uv both resolve it. Replace it with a plain version specifier once the beta lands.

Every notebook carries the same PEP 723 header, which names the whole environment, this package included, and resolves all of it from GitHub:

```python
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
```

It reads as duplication of `pyproject.toml` and it is: the two lists have to be kept in step by hand when a dependency changes. That is the price of molab, which hands a notebook nothing but its own file. The header this replaced named the package alone and installed it from `path = "."`, which resolves against the notebook's directory -- fine from a clone, nothing at all in the browser, where there is no clone to find. `pyproject.toml` is still what a local `pip install -e .` reads, and still the only place a version bound is decided; the headers restate it. `04_dasjax.py` additionally restates the `bonus` extra: `dasjax==0.0.3`, `jax==0.11.1`, and the CUDA plugin pinned package by package under molab's Linux x86_64 marker.

Those CUDA pins name the CUDA 13 family, and they have to. molab is not an empty container: its image preloads its own JAX, and the sandbox venv it builds from the notebook header sees those packages behind its own. jaxlib pairs only with a plugin reporting its exact version, and it probes `jax_cuda13_plugin` before `jax_cuda12_plugin`. Ask for a CUDA 12 plugin and both are importable, so jaxlib loads the image's CUDA 13 one first, rejects it on the version check -- but nanobind has already registered `TritonKernelCall` and its neighbours by then, so the CUDA 12 plugin that jaxlib does accept imports stripped of them, and `import jax` dies with `AttributeError: module 'jax_cuda12_plugin._triton' has no attribute 'TritonKernelCall'`. Pinning the CUDA 13 family instead means our plugin shadows the image's copy under the same name, and nothing else is ever probed. It also survives molab bumping its own JAX, where a matching pin would not. The failure needs two plugin families present, so it reproduces nowhere else: local machines and CI install one.

Because the package installs rather than being pointed at, the data has to install with it: it lives at `galileo_2026/data` and `pyproject.toml` ships it as package data. `get_data_dir()`, which every other accessor goes through, uses a checkout where it stands, and an installed package unpacks its data to `galileo_2026_data` beside the notebook -- `site-packages` is no place for attendees to go looking. `GALILEO_2026_DATA` overrides both.

`@main` means an attendee's environment follows this branch, so a bad push during the session reaches everyone who installs after it. uv caches per commit.

To work against a DASCore checkout instead, install it over the synced environment and then keep uv from undoing that: `uv pip install -e ../dascore`, followed by `uv run --no-sync marimo edit`. The same applies to this package: a sandbox run installs it from GitHub, not from the working tree, so test local changes with `uv run marimo edit` in the project environment.

A figure a notebook draws lives in `galileo_2026/images/` and ships with the package, because molab has the notebook and nothing beside it: a relative `images/...` path resolves to nothing there, and a raw.githubusercontent URL wants a network the conference may not give us. Draw it with `mo.image(get_image_path("name.svg"))`, which inlines the file, so the notebook needs neither. Assets only the slides or the readme use stay in the top-level `images/`; they are rendered from a checkout and keep their relative paths. `images/inventory_hierarchy.svg` is used by both, so it moved with the notebooks and `intro.qmd` reaches across to it.

The project floor is python 3.12. molab runs newer, but the floor also covers attendees on older local pythons.

The notebooks are marimo notebooks: plain python files, opened with `marimo edit`, and run in the browser on [molab](https://molab.marimo.io) from the badges in the readme. There is no bootstrap cell -- molab reads the notebook's PEP 723 header, and everything the notebook needs, this package and its data included, comes from there. There is no `%matplotlib inline` either: marimo renders the matplotlib `Axes` that a `.viz` call returns, so `show=True` only adds a second path to the same figure.

A marimo notebook is a dependency graph, not a list of cells, and that constrains how tutorial code may be written:

- **A name belongs to one cell.** Two cells cannot both assign `patch`. Where the tutorial wants a second one, name it for what it is (`das_lf_patch`, `dss_patch`) rather than letting marimo generate `patch_1`.
- **Cells are ordered by their dependencies**, not by their position in the file, so moving a cell does not change what it computes.
- **A leading `_` keeps a name cell-local.** This is how the plotting cells each get their own `_fig` and `_ax`.

## Notebooks in git

A marimo notebook holds no outputs, so it commits as ordinary python: no filter, no `.gitattributes` entry, and `git diff` shows the change that matters. Ruff formats them alongside the rest of the repo; lint rules are disabled, as `pyproject.toml` explains.

They were converted from Jupyter with `marimo convert`; `marimo export ipynb` goes back the other way if a Colab fallback is ever wanted again.

## Checks

Run before handing off: `uvx ruff check .`, `uvx ruff format --check .`, `uv lock --check`, and execute every notebook -- `uv run python 01_the_patch.py` runs one end to end without a browser. `.github/workflows/notebooks.yml` does the same on every push and pull request, on the 3.12 floor, with the pinned CPU-only DASJax/JAX pair installed so notebook 04 runs without the `bonus` extra's CUDA wheels. Notebooks are stubs under active authoring — never overwrite a cell you did not write, and check the cell count before and after any programmatic edit.
