%% The bow-tie structure of a genome-scale metabolic network
% Following Ma and Zeng (2003), we build a directed substrate graph from
% the E. coli model iML1515: an edge from every substrate of a reaction to
% every product, and back again if the reaction is reversible. Exchange
% and other one-metabolite reactions, the biomass reactions and ATP
% maintenance are left out, as are currency metabolites (in every
% compartment), which would otherwise connect almost everything.
% The bow-tie (book §13.1.3.1) then has four parts:
%   * the core, or "knot": the largest strongly connected component, in
%     which every metabolite can be converted into every other one;
%   * IN: metabolites from which the core can be reached, but not back;
%   * OUT: metabolites that can be reached from the core, but not back;
%   * the rest, which neither reach the core nor are reached from it.

s = load(fullfile(sbp.data_dir(), 'iML1515.mat'));
model = s.iML1515;
S = model.S; mets = model.mets; rxns = model.rxns;

currency = {'h', 'h2o', 'atp', 'adp', 'amp', 'pi', 'ppi', 'nad', 'nadh', ...
    'nadp', 'nadph', 'co2', 'coa', 'nh4', 'o2', 'q8', 'q8h2', 'mql8', ...
    'mqn8', '2dmmq8', '2dmmql8', 'fad', 'fadh2', 'gtp', 'gdp', 'ctp', ...
    'cdp', 'utp', 'udp', 'h2o2', 'na1', 'k', 'fe2', 'fe3', 'glu__L', 'akg', ...
    'accoa', 'so4', 'gln__L'};
base = regexprep(mets, '_[^_]+$', '');            % drop the compartment
keep = ~ismember(base, currency);

src = {}; dst = {};
for j = 1:numel(rxns)
    column = S(:, j);
    if nnz(column) < 2 || startsWith(rxns{j}, 'BIOMASS') || strcmp(rxns{j}, 'ATPM')
        continue
    end
    substrates = mets(column < 0 & keep);
    products = mets(column > 0 & keep);
    [a, b] = ndgrid(1:numel(substrates), 1:numel(products));
    src = [src; substrates(a(:))]; dst = [dst; products(b(:))]; %#ok<AGROW>
    if model.lb(j) < 0
        src = [src; products(b(:))]; dst = [dst; substrates(a(:))]; %#ok<AGROW>
    end
end
pairs = unique([string(src), string(dst)], 'rows');
G = digraph(pairs(:, 1), pairs(:, 2));
names = string(G.Nodes.Name);

bins = conncomp(G, 'Type', 'strong');
[~, largest] = max(accumarray(bins', 1));
core = find(bins == largest)';
seed = core(1);
out = setdiff(nearest(G, seed, Inf), core);
inn = setdiff(nearest(G, seed, Inf, 'Direction', 'incoming'), core);
rest = setdiff((1:numnodes(G))', [core; out; inn]);
parts = struct('in', inn, 'core', core, 'out', out, 'rest', rest);
keys = fieldnames(parts)';

results.metabolites = numnodes(G);
results.edges = numedges(G);
fprintf('%d metabolites, %d edges\n', numnodes(G), numedges(G));
for k = keys
    nodes = names(parts.(k{1}));
    tags = extractAfter(nodes, regexpPattern('.*_'));
    counts = [sum(tags == "c"), sum(tags == "p"), sum(tags == "e")];
    results.(k{1}) = struct('size', numel(nodes), 'by_compartment', counts);
    fprintf('  %-5s %5d metabolites (%.0f%%): %d cytosolic, %d periplasmic, %d extracellular\n', ...
        k{1}, numel(nodes), 100 * numel(nodes) / numnodes(G), counts);
end
for m = ["glc__D_e", "pyr_c", "succ_c", "ala__L_c", "cbl1_e", "btn_c"]
    where = "absent";
    for k = keys
        if any(names(parts.(k{1})) == m), where = k{1}; end
    end
    results.landmarks.(m) = where;
    fprintf('  %-10s is in %s\n', m, where);
end

%% Plot
fig = figure('Position', [100 100 650 400]);
heights = zeros(4, 3);
for k = 1:4
    heights(k, :) = results.(keys{k}).by_compartment;
end
b = bar(categorical({'IN', 'core', 'OUT', 'rest'}, {'IN', 'core', 'OUT', 'rest'}), ...
    heights, 'stacked');
colours = [0.122 0.467 0.706; 1.000 0.498 0.055; 0.173 0.627 0.173];
for c = 1:3, b(c).FaceColor = colours(c, :); end
ylabel('Metabolites'); title('The bow-tie of iML1515');
legend({'cytosol', 'periplasm', 'extracellular'}, 'Box', 'off'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-bow-tie');

sbp.save_results(mfilename('fullpath'), results);
