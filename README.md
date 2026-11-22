# Computational Systems Biology in MATLAB and Python

[![check](https://github.com/RamanLab/sysbio-code-companion/actions/workflows/check.yml/badge.svg)](https://github.com/RamanLab/sysbio-code-companion/actions/workflows/check.yml)
[![publish](https://github.com/RamanLab/sysbio-code-companion/actions/workflows/publish.yml/badge.svg)](https://ramanlab.github.io/sysbio-code-companion)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Text: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey.svg)](LICENSE-TEXT.md)

An open code companion to the book *[An Introduction to Computational Systems Biology: Systems-Level Modelling of Cellular Networks](https://doi.org/10.1201/9780429486951)* (Karthik Raman, CRC Press, 2021), also used in the course BT5240 at IIT Madras. Its chapters and sections follow the book's exactly, and every worked example is in both MATLAB and Python.

**Read it online:** <https://ramanlab.github.io/sysbio-code-companion> (a PDF version is linked from the site).

## What is inside

Fifty-eight worked examples, covering every section of the book that can carry code:

* **Static modelling:** graphs and their representations, network structure (degree distributions, small worlds, assortativity, communities, motifs), and applications to protein interaction, co-expression and regulatory networks of *E. coli*.
* **Dynamic modelling:** ODE models and their numerical solution, Michaelis–Menten kinetics, S-systems, stochastic simulation, parameter estimation (least squares, global search, model selection, sensitivity and sloppiness) and Boolean networks.
* **Constraint-based modelling:** flux balance and flux variability analysis, parsimonious FBA, MoMA, ROOM, OptKnock, E-Flux, dynamic FBA, elementary flux modes, gene and reaction deletions, synthetic lethals and FSEOF, on the *E. coli* core model and on the genome-scale model iML1515.
* **Advanced topics:** microbial communities, host–pathogen dynamics, synthetic circuits (the repressilator, the toggle switch, hysteresis), robustness and evolvability.

The two versions of each example are kept parallel, line by line where the languages allow, and produce the same numbers and the same figures, so a reader can follow either one, or learn one language from the other.

## Layout

```
chapters/            the text, one Quarto (.qmd) file per book chapter, with the book's sections
examples/<topic>/<example>/
    <name>.m         MATLAB version
    <name>.py        Python version
    example.yml      book reference, toolboxes needed, what to compare, expected values
    out/matlab/      figures + results.json written by the MATLAB script (not committed)
    out/python/      figures + results.json written by the Python script (not committed)
data/                shared models and datasets, each documented in data/README.md
matlab/+sbp/         small MATLAB helpers (saving results and figures, colours, random numbers)
python/sbp/          the same helpers for Python
tests/compare.py     checks the results: Python vs MATLAB vs book
tools/               scripts to run all examples, make the data, and Quarto filters
```

## Running the examples

See the [Getting started](chapters/getting-started.qmd) chapter. In short, for Python:

```bash
pip install -r requirements.txt && pip install -e .
python tools/run_examples.py          # run every Python example
python tests/compare.py               # check the results
```

In MATLAB (R2022a or newer, tested with R2024b, with the Optimization Toolbox), from the repository folder: `setup_companion; addpath tools; run_examples`. A few constraint-based examples also need the [COBRA Toolbox](https://opencobra.github.io/cobratoolbox/); run `initCobraToolbox` first for those.

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

## Contributing

Corrections and new examples are welcome, as issues or pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add an example, and for the conventions the examples follow (including the colour-blind-friendly palette used in every figure).

## Citing

If you use the companion, please cite it together with the book; GitHub's "Cite this repository" button, under the repository's About section, gives the citation in several formats, from [`CITATION.cff`](CITATION.cff).

> Raman, K. (2021). *An Introduction to Computational Systems Biology: Systems-Level Modelling of Cellular Networks*. CRC Press. <https://doi.org/10.1201/9780429486951>

## Licence

Text: [CC BY 4.0](LICENSE-TEXT.md). Code: [MIT](LICENSE). The data in `data/` come from BiGG Models, STRING, NetGenes, PRECISE and PRECISE-1K (which include RegulonDB's regulatory network) and Zachary's karate club study, each under its own terms, or were simulated by scripts in `tools/`; see [data/README.md](data/README.md).
