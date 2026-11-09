# Computational Systems Biology in MATLAB and Python

An open primer on computational systems biology, with every worked example in both MATLAB and Python. It accompanies the book *An Introduction to Computational Systems Biology: Systems-Level Modelling of Cellular Networks* (Karthik Raman, CRC Press, 2021) and the course BT5240 at IIT Madras.

**Read it online:** <https://ramanlab.github.io/sysbio-primer> (a PDF version is linked from the site).

## Layout

```
chapters/            the text, one Quarto (.qmd) file per chapter
examples/<topic>/<example>/
    <name>.m         MATLAB version
    <name>.py        Python version
    example.yml      book reference, toolboxes needed, what to compare, expected values
    out/matlab/      figures + results.json written by the MATLAB script (not committed)
    out/python/      figures + results.json written by the Python script (not committed)
data/                shared models and datasets
matlab/+sbp/         small MATLAB helpers (save results and figures)
python/sbp/          the same helpers for Python
tests/compare.py     checks the results: Python vs MATLAB vs book
tools/               scripts to run all examples, and Quarto filters
```

## Running the examples

See the [Getting started](chapters/getting-started.qmd) chapter. In short:

```bash
pip install -r requirements.txt && pip install -e .
python tools/run_examples.py          # run every Python example
python tests/compare.py               # check the results
```

In MATLAB, from the repository folder: `addpath tools; run_examples` (run `initCobraToolbox` first for the constraint-based examples).

## Building the site

```bash
quarto preview          # live preview while editing
quarto render           # website and PDF, into _book/
```

The pages show the figures under `examples/**/out/`, which the scripts write when you run them; these outputs are not committed. To preview locally, run the Python examples first (`python tools/run_examples.py`), and the MATLAB ones too if you have MATLAB. Any figure that has not been generated is replaced by a short note.

## Checks

On every push, GitHub Actions ([`check.yml`](.github/workflows/check.yml)) runs all examples in both languages and runs `tests/compare.py`, which checks that

* each language reproduces the values listed under `expected:` in `example.yml` (taken from the book where available), and
* MATLAB and Python agree, within the example's tolerance, on the quantities listed under `compare:`.

MATLAB runs in CI free of charge because this repository is public. CI uses MATLAB R2024b: on the GitHub runners, `linprog`'s default algorithm fails in R2025a and R2025b with "Unrecognized field name optimstatus" (checked in October 2026), while R2024b works with every algorithm. Examples that need the COBRA Toolbox run last, after `initCobraToolbox`.

When the checks pass on `main`, [`publish.yml`](.github/workflows/publish.yml) builds the website and PDF from the outputs of that very run, in both languages, and publishes them on GitHub Pages. So the published figures always come from the code as it stands, and nobody needs MATLAB to update the site. To bring the latest CI outputs into your own working copy, use `tools/fetch_ci_outputs.sh matlab`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add an example.

## Licence

Text: [CC BY 4.0](LICENSE-TEXT.md). Code: [MIT](LICENSE). The data in `data/` come from BiGG Models, STRING and NetGenes, each under its own terms; see [data/README.md](data/README.md).
