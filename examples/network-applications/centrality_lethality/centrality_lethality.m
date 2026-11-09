%% Are central proteins more often essential?
% The E. coli K-12 protein interaction network from STRING (v12.0, links
% with a combined score of at least 700), and the genes that NetGenes
% predicts to be essential. We compute three centralities for each
% protein, see how the fraction of essential genes changes with degree, and
% measure how well each centrality alone separates essential from other
% genes.
% Requires: core MATLAB only.

data = sbp.data_dir();
links = readtable(fullfile(data, 'ecoli_string_ppi.csv'), ...
    'TextType', 'string');
proteins = readtable(fullfile(data, 'ecoli_string_proteins.csv'), ...
    'TextType', 'string');
netgenes = readtable(fullfile(data, 'ecoli_netgenes_essential.csv'), ...
    'TextType', 'string', 'Delimiter', ',');

G = graph(links.protein1, links.protein2);
[comp, sizes] = conncomp(G);
[~, largest] = max(sizes);
G = subgraph(G, comp == largest);
[nodes, order] = sort(G.Nodes.Name);       % a fixed order, as in Python
G = reordernodes(G, order);
is_ess = ismember(nodes, netgenes.string_id);
fprintf(['Largest connected component: %d proteins, %d interactions, ' ...
    '%d essential\n'], numnodes(G), numedges(G), nnz(is_ess));

names = {'degree', 'closeness', 'eigenvector'};
for c = 1:3
    cent.(names{c}) = centrality(G, names{c});
end
for c = 1:3
    aucs.(names{c}) = auc(cent.(names{c}), is_ess);
    fprintf('AUC of %s: %.4f\n', names{c}, aucs.(names{c}));
end

% Fraction essential in ten equal groups, from least to most connected
deg = cent.degree;
[~, order] = sort(deg);
n = numel(deg);
group_size = floor(n / 10) + ((1:10) <= mod(n, 10));
edges = [0 cumsum(group_size)];
frac_ess = zeros(1, 10); mean_deg = zeros(1, 10);
for g = 1:10
    idx = order(edges(g) + 1:edges(g + 1));
    frac_ess(g) = mean(is_ess(idx));
    mean_deg(g) = mean(deg(idx));
end
fprintf('Fraction essential by degree group:%s\n', ...
    sprintf(' %.3f', frac_ess));

[~, top] = sort(deg, 'descend');
top = top(1:10);
[~, where] = ismember(nodes(top), proteins.string_id);
top_names = proteins.name(where);
disp('Ten most connected proteins:');
for k = 1:10
    tag = '';
    if is_ess(top(k)), tag = '  essential'; end
    fprintf('  %-6s degree %d%s\n', top_names(k), deg(top(k)), tag);
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
semilogx(mean_deg, frac_ess, 'o-', 'Color', [0.122 0.467 0.706], ...
    'MarkerFaceColor', [0.122 0.467 0.706]);
yline(mean(is_ess), ':', 'all proteins', 'Color', [0.6 0.6 0.6], ...
    'LabelHorizontalAlignment', 'left');
xlim([1 100]); xticks([1 2 5 10 20 50 100]);
xlabel('Mean degree of the group'); ylabel('Fraction essential');
title('Ten groups of proteins, by degree');
box off
subplot(1, 2, 2)
colours = [0.122 0.467 0.706; 1.000 0.498 0.055; 0.173 0.627 0.173];
hold on
for c = 1:3
    x = cent.(names{c});
    thresholds = sort(unique(x), 'descend');
    tpr = arrayfun(@(t) mean(x(is_ess) >= t), thresholds);
    fpr = arrayfun(@(t) mean(x(~is_ess) >= t), thresholds);
    plot([0; fpr], [0; tpr], 'Color', colours(c, :), 'DisplayName', ...
        sprintf('%s (AUC %.2f)', names{c}, aucs.(names{c})));
end
plot([0 1], [0 1], ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
xlabel('Fraction of other genes above threshold');
ylabel('Fraction of essential genes above threshold');
title('ROC curves');
legend('Location', 'southeast', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-centrality-lethality');

results.n_nodes = numnodes(G);
results.n_edges = numedges(G);
results.n_essential = nnz(is_ess);
results.auc = aucs;
results.fraction_essential_by_degree = frac_ess;
results.top_degree = cellstr(top_names)';
sbp.save_results(mfilename('fullpath'), results);

%% Local function
function a = auc(score, positive)
% Area under the ROC curve: the probability that a randomly chosen
% positive scores higher than a randomly chosen negative (ties count one
% half), from the ranks of the scores.
[s, order] = sort(score);
ranks = zeros(size(score));
i = 1;
while i <= numel(s)                       % average ranks over ties
    j = i;
    while j < numel(s) && s(j + 1) == s(i)
        j = j + 1;
    end
    ranks(order(i:j)) = (i + j) / 2;
    i = j + 1;
end
n_pos = nnz(positive); n_neg = nnz(~positive);
a = (sum(ranks(positive)) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg);
end
