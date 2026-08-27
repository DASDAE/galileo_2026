# Galileo Conference DASDAE Tutorial

**[dasdae.github.io/galileo_2026](https://dasdae.github.io/galileo_2026/)** -- the rendered site, with the slides and this page.

Data and code for the [Galileo 2026](https://www.egu-galileo.eu/gc14-fibreoptic/home.html) [DASDAE](https://github.com/DASDAE) tutorial. 

- The [intro slides](https://dasdae.github.io/galileo_2026/intro.html) set the session up
- the [conclusion slides](https://dasdae.github.io/galileo_2026/conclusions.html) close it.

# Learning objectives

1. **`Patch`** -- filter, smooth, transform and plot DFOS data.
2. **`Spool`** -- subselect, chunk, merge, reshape and visualize an archive.
3. **`Inventory`** -- attach deployment metadata, and let it enrich selections and patches.
4. *(bonus, if time permits)* **[DASJax](https://github.com/DASDAE/dasjax)** -- compile a CPU/GPU kernel and time it against DASCore's numpy.


# Contents

| Notebook | Topic | molab |
| --- | --- | --- |
| `00_setup.py` | Check your environment and data | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/00_setup.py) |
| `01_the_patch.py` | The `Patch`: filtering, transforms and plots | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/01_the_patch.py) |
| `02_the_spool.py` | The `Spool`: an archive, selected and re-cut | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/02_the_spool.py) |
| `03_the_inventory.py` | The observing system, with `Inventory` | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/03_the_inventory.py) |
| `04_dasjax.py` | *(bonus)* Compiled kernels with DASJax | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/04_dasjax.py) |

The same three notebooks with every exercise worked are in [`solutions/`](solutions/); open them the same way.

And, just for fun, I asked Claude Code (running Fable 5) and Codex (running GPT 5.6 Sol Ultra) to do some independent research, and return something interesting. These are pure-AI generated notebooks which we will examine, time permitting, and evaluate the model's reasoning and code quality. Their contents are as follows:

## Claude Code

| Notebook | Topic | molab |
| --- | --- | --- |
| [`01_the_step_response.py`](claude_research/01_the_step_response.py) | The offset the blast leaves in the DAS phase decays as the step response of the interrogator's own first-order high-pass, established by five checks, the last of which undoes the filter. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/claude_research/01_the_step_response.py) |
| [`02_the_clock_in_the_files.py`](claude_research/02_the_clock_in_the_files.py) | The gaps `get_gaps` reports in the 5 Hz files are a drifting sample count rather than missing data, so times inside a file are good to about a second. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/claude_research/02_the_clock_in_the_files.py) |
| [`03_the_brillouin_record.py`](claude_research/03_the_brillouin_record.py) | No borehole moved above the BOTDR's noise floor, and the wild sweeps that forced a robust statistic turn out to locate four cable junctions. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/claude_research/03_the_brillouin_record.py) |

## Codex

| Notebook | Topic | molab |
| --- | --- | --- |
| [`01_blast_moveout.py`](codex_research/01_blast_moveout.py) | A robust plane fit to the first blast arrival across the five north holes gives projected velocities along the drift and down the holes, with the paired fiber legs as its check. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/codex_research/01_blast_moveout.py) |
| [`02_transient_or_permanent.py`](codex_research/02_transient_or_permanent.py) | The apparent step at the bottom of N180 decays within a minute on both co-located fiber legs, and the Brillouin record resolves no step that persists, so it is transient in the recorded DAS. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/codex_research/02_transient_or_permanent.py) |
| [`03_coupling_or_optics.py`](codex_research/03_coupling_or_optics.py) | The inventory's cemented intervals are several times quieter than nearby unlabelled cable -- a registration feature rather than an independent coupling test -- and no monotonic association is detected between the uncoupled cable's variability and the cumulative listed connector and splice loss. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/DASDAE/galileo_2026/blob/main/codex_research/03_coupling_or_optics.py) |

# Background

The data comes from an underground hardrock mine in Europe, where two types of sensing cable are grouted into 14 boreholes above the planned extraction volume, to watch how the rockmass holds up as extraction proceeds. See the [borehole geometry](galileo_2026/data/docs/geometry.html) and the [optical paths](galileo_2026/data/docs/optical_topo.pdf).

Four kinds of fiber data ship with it, along with the metadata that describes them: DSS from a Febus G1 BOTDR, low frequency and 2 kHz DAS from a Sintela Onyx Peta (the interrogator samples at 20 kHz, but everything it writes out is decimated), and OTDR traces from a Tempo Communications OFL100. Later notebooks study a mine blast that went off at **2026-08-06T10:13:27** UTC.

# Setup

The notebooks are [marimo](https://marimo.io) notebooks: plain python files that name their own dependencies, so there is no environment to build by hand. You will need **python 3.12 or newer**, and familiarity with numpy, pandas and matplotlib.

## In the browser

Click a molab badge above: it opens the notebook on a machine that is not the conference wifi, and the notebook installs this repository, data and all, from its own first lines. Signing in with Google or GitHub is the only setup. **Please open one before you arrive**, so a login problem is not the first thing you hit in the session.

The bonus notebook is the one place the machine matters: molab's *notebook specs* button will attach an NVIDIA GPU, and `04_dasjax.py` compiles for whatever it finds. It runs on the default CPU too, and says which one it got.

## On your own machine

**Please do this before you arrive.** Conference wifi is not a good place to install a scientific python stack for the first time.

To open a single marimo file using its own requirements, run:

```bash
uvx marimo edit --sandbox 00_setup.py
```

That installs this repository from GitHub, data included -- about fifty megabytes -- and unpacks the recordings into `galileo_2026_data` beside the notebook.

Cloning is still the better way to work through the session, since it brings all five notebooks and lets you read the data in place:

```bash
git clone https://github.com/DASDAE/galileo_2026
cd galileo_2026
```

Without git, take the ZIP from the green *Code* button on [the repo page](https://github.com/DASDAE/galileo_2026) instead. From a clone, [uv](https://docs.astral.sh/uv/) builds one environment for every notebook:

```bash
uv run marimo edit
```

Without uv, build that environment yourself:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e .     # use ".[bonus]" if you want notebook 04
marimo edit
```

The `bonus` extra is a heavy one on Linux: it brings JAX's CUDA wheels, a few gigabytes of them, so that notebook 04 can reach a GPU. Everywhere else it installs the CPU build and the notebook runs the same, just slower.

Either way this pulls DASCore from its `dev` branch -- the inventory, the OTDR reader and the hive-style directory attributes are not released yet -- so the install needs git on your path and a few minutes.

Run `00_setup.py`. If it prints a DASCore version and four rows of file counts, you are ready for the session.

# Building the site

The landing page and both slide decks are one [quarto](https://quarto.org) project, so one command builds all of it into `_site`:

```bash
quarto render      # or: quarto preview, for live reload
```

Nothing is executed -- the pages are markdown, the figures and the borehole geometry are committed files -- so quarto is the only thing you need installed. Pushing any of them to `main` publishes the site to [GitHub Pages](https://dasdae.github.io/galileo_2026/).

# License

Code is MIT, data is CC BY 4.0. See [LICENSING.md](LICENSING.md).

# AI usage

Claude Code and Codex were used in preparing and editing this material.
