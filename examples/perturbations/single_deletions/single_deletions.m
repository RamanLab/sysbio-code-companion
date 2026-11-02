%% Single reaction and gene deletions in the E. coli core model
% A reaction is deleted by fixing its flux to zero (lower and upper bounds
% both 0; book Eq. 10.2); the maximum growth rate is then found by FBA. A
% deletion is called lethal if growth falls below 1% of the wild type.
% Genes are deleted through the gene-protein-reaction (GPR) rules, which
% the COBRA Toolbox evaluates for us: a reaction is lost only if the
% deletion makes its rule false, so isozymes can stand in for one another.
% Requires: COBRA Toolbox (run initCobraToolbox once per session first).

model = readCbModel(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
sol = optimizeCbModel(model, 'max');
wt = sol.f;
% Delete internal reactions only: no exchange reactions, no biomass
rxns = sort(model.rxns(~startsWith(model.rxns, {'EX_', 'Biomass'})));

%% Reaction deletions
ratio = zeros(numel(rxns), 1);
for k = 1:numel(rxns)
    m = changeRxnBounds(model, rxns{k}, 0, 'b');
    sol = optimizeCbModel(m, 'max');
    if sol.stat == 1
        ratio(k) = sol.f / wt;
    end                        % infeasible: no growth, ratio stays 0
end
essential_rxns = rxns(ratio < 0.01);
n_reduced = sum(ratio >= 0.01 & ratio < 0.99);

%% Gene deletions (the GPR rules decide which reactions are lost)
grRatio = singleGeneDeletion(model, 'FBA');
grRatio(isnan(grRatio)) = 0;   % infeasible: no growth
[essential_genes, idx] = sort(model.genes(grRatio < 0.01));
names = model.geneNames(grRatio < 0.01);
names = names(idx);

fprintf('Wild-type growth %.4f /h; %d reactions tested\n', wt, numel(rxns));
fprintf('%d essential reactions: %s\n', numel(essential_rxns), ...
    strjoin(essential_rxns, ', '));
fprintf('%d reactions whose loss reduces growth, but not below 1%%\n', ...
    n_reduced);
fprintf('%d essential genes, of %d: %s\n', numel(essential_genes), ...
    numel(model.genes), strjoin(strcat(essential_genes, ' (', names, ...
    ')'), ', '));

%% Plot: growth after each reaction deletion, sorted
[values, order] = sort(ratio);
colors = repmat([0.122 0.467 0.706], numel(values), 1);       % no effect
colors(values < 0.99, :) = repmat([1.000 0.498 0.055], ...
    sum(values < 0.99), 1);                                   % reduced
colors(values < 0.01, :) = repmat([0.839 0.153 0.157], ...
    sum(values < 0.01), 1);                                   % lethal
fig = figure('Position', [100 100 1000 400]);
b = bar(values, 'FaceColor', 'flat');
b.CData = colors;
hold on
lethal = find(values < 0.01);               % zero-height bars: mark them
plot(lethal, 0.03 * ones(size(lethal)), 'v', 'Color', [0.839 0.153 0.157], ...
    'MarkerFaceColor', [0.839 0.153 0.157], 'MarkerSize', 5);
set(gca, 'XTick', 1:numel(values), 'XTickLabel', rxns(order), ...
    'XTickLabelRotation', 90, 'FontSize', 7);
ylabel('Growth relative to wild type');
title('Single reaction deletions, E. coli core model, aerobic glucose');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-single-deletions');

results.wild_type_growth = wt;
results.growth_ratio = cell2struct(num2cell(ratio), rxns, 1);
results.essential_reactions = essential_rxns';
results.essential_genes = essential_genes';
sbp.save_results(mfilename('fullpath'), results);
