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
    # Setup

    Check that your environment and the tutorial data are ready. Run this before the session — everything else builds on it.
    """)
    return


@app.cell
def _():
    import dascore as dc

    from galileo_2026 import get_data_path

    print(f"dascore {dc.__version__}")

    for _tag in ["DSS", "DAS", "DAS_LF", "OTDR"]:
        _path = get_data_path(_tag)
        _n_files = len(list(_path.glob("*")))
        print(f"{_tag:<8} {_n_files:>3} files")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If that printed four rows without error, you are ready.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The rest of the tutorial

    molab opens one notebook per link, so these are the four to work through in order. Each one arrives with this whole repository behind it, data included.

    | Notebook | Topic |
    | --- | --- |
    | [`01_the_patch.py`](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/01_the_patch.py) | The `Patch`: filtering, transforms and plots |
    | [`02_the_spool.py`](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/02_the_spool.py) | The `Spool`: an archive, selected and re-cut |
    | [`03_the_inventory.py`](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/03_the_inventory.py) | The observing system, with `Inventory` |
    | [`04_dasjax.py`](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/04_dasjax.py) | *(bonus)* Compiled kernels with DASJax |
    """)
    return


if __name__ == "__main__":
    app.run()
