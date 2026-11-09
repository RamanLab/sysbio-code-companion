# Adding an example

1. **Make a folder** `examples/<topic>/<example_name>/` (lower case, underscores).

2. **Write the two scripts**, `<example_name>.py` and `<example_name>.m`. Keep them parallel, so a reader can follow one while looking at the other:
   * same variable names where the languages allow, same order of steps, same comments;
   * a short header saying what the example does, and which toolboxes or packages it needs;
   * the same figure in both: same title, axis labels, legend and colours (blue `#1f77b4`, orange `#ff7f0e`, green `#2ca02c`);
   * lines of at most 79 characters, so they display without wrapping;
   * at the end, save the figure(s) with `sbp.save_figure` and the key numbers with `sbp.save_results`, using the **same keys** in both languages.

3. **Write `example.yml`**:
   ```yaml
   title: One-line title
   book: "Ch. 8, §8.7 (Eq. 8.12)"   # where the theory is
   needs:
     matlab: [Optimization Toolbox]  # or [] for core MATLAB
     python: [scipy]
   compare: [growth, fva100]         # keys that must agree between languages
   tolerance: {rtol: 1.0e-6, atol: 1.0e-6}
   expected:                          # values from the book, if any
     growth: 50
   ```
   Only list under `compare:` quantities that are *unique*. For example, an FBA flux vector is usually one of many optima, so different solvers may return different ones; compare the objective value instead. For random examples (random networks, stochastic simulation), compare summary statistics with a suitable tolerance, or read the same input from `data/`.

4. **Run and check**:
   ```bash
   python tools/run_examples.py examples/<topic>/<example_name>
   python tests/compare.py examples/<topic>/<example_name>
   python tests/lint_matlab.py
   ```
   The last check catches a MATLAB rule that Python does not have: a script may not use its own name for a variable or a local function (`efms.m` cannot assign to `efms`).
   Run the MATLAB script too if you have MATLAB; otherwise the CI run will (and `tools/fetch_ci_outputs.sh matlab` brings its outputs into your working copy for previewing).

5. **Add it to the chapter, under the book section it illustrates.** The chapters follow the book's numbering exactly, with every numbered section and subsection of the book present, so never add or remove a numbered heading; extra headings inside a section must be level 4 (`####`) or marked `{.unnumbered}`. Material that goes beyond the book, rather than illustrating one of its sections, goes at the end of the chapter (before Exercises) in a section labelled with a letter, written into the heading by hand: `## 5.A Stochastic simulation {.unnumbered #sec-stochastic}`, with subsections `### 5.A.1 …`. The next one in that chapter is 5.B. Replace the section's "There is no code for this section" line with a short introduction and a tab set:
   ````markdown
   ::: {.panel-tabset group="language"}
   ## MATLAB
   ```{.matlab include="/examples/<topic>/<name>/<name>.m"}
   ```
   ![](/examples/<topic>/<name>/out/matlab/fig-<name>.png){width=75%}

   ## Python
   ```{.python include="/examples/<topic>/<name>/<name>.py"}
   ```
   ![](/examples/<topic>/<name>/out/python/fig-<name>.png){width=75%}
   :::
   ````

6. **Commit the code, `example.yml` and the chapter text**, but not the `out/` folders: CI regenerates the outputs, checks them, and publishes the site from them.
