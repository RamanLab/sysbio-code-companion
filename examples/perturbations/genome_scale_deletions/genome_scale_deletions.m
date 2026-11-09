%% Single gene deletions in a genome-scale model: E. coli iML1515
% The same procedure as for the core model (single_deletions), on the
% genome-scale reconstruction iML1515 (Monk et al., 2017): 1516 genes and
% 2712 reactions, in its default medium (aerobic, glucose uptake up to
% 10 mmol/gDW/h). A gene deletion removes the reactions whose GPR rule it
% makes false, and is lethal if growth falls below 1% of the wild type.
% Requires: COBRA Toolbox (run initCobraToolbox once per session first).

model = readCbModel(fullfile(sbp.data_dir(), 'iML1515.mat'));
sol = optimizeCbModel(model, 'max');
wt = sol.f;

[grRatio, ~, ~, ~, delRxns] = singleGeneDeletion(model, 'FBA');
grRatio(isnan(grRatio)) = 0;   % infeasible: no growth
[genes, order] = sort(model.genes);
grRatio = grRatio(order);
delRxns = delRxns(order);

essential = genes(grRatio < 0.01);
n_reduced = sum(grRatio >= 0.01 & grRatio < 0.99);
no_rxn = genes(cellfun(@isempty, delRxns));

fprintf('Wild-type growth %.4f /h; %d genes deleted one by one\n', ...
    wt, numel(genes));
fprintf('%d essential genes (%.1f%%)\n', numel(essential), ...
    100 * numel(essential) / numel(genes));
fprintf('%d genes whose loss reduces growth, but not below 1%%\n', n_reduced);
fprintf('%d genes whose loss removes no reaction (isozymes)\n', numel(no_rxn));

%% Plot
counts = [numel(essential), n_reduced, numel(genes) - numel(essential) - n_reduced];
fig = figure('Position', [100 100 500 450]);
b = bar(counts, 'FaceColor', 'flat');
b.CData = [0.839 0.153 0.157; 1.000 0.498 0.055; 0.122 0.467 0.706];
text(1:3, counts, string(counts), 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');
set(gca, 'XTickLabel', {'lethal (< 1%)', 'reduced', 'no effect'});
ylabel('Genes');
title('Single gene deletions, iML1515');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-genome-scale-deletions');

results.wild_type_growth = wt;
results.growth_ratio = cell2struct(num2cell(grRatio), genes, 1);
results.essential_genes = essential';
results.n_reduced = n_reduced;
results.no_reaction_lost = no_rxn';
sbp.save_results(mfilename('fullpath'), results);
