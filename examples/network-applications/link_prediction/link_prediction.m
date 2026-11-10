%% Link prediction: can the structure of a network predict missing edges?
% We hide 10% of the edges of the E. coli STRING network (combined score at
% least 700), and score pairs of nodes using only the remaining network,
% by four similarity indices (book §4.7):
%   common neighbours    CN(x, y) = |N(x) ∩ N(y)|
%   Jaccard              CN(x, y) / |N(x) ∪ N(y)|
%   Adamic-Adar          sum over common neighbours z of 1 / log k_z
%   preferential attachment   k_x k_y
% An index is good if it scores the hidden edges above pairs that are not
% linked in the full network. We measure this by the AUC: the probability
% that a randomly chosen hidden edge scores higher than a randomly chosen
% non-edge (ties count one half). The hidden edges, and an equal number of
% non-edges, are chosen with sbp.uniform, so that MATLAB and Python use the
% same ones.

T = readtable(fullfile(sbp.data_dir(), 'ecoli_string_ppi.csv'), 'TextType', 'string');
names = unique([T.protein1; T.protein2]);         % sorted
[~, ia] = ismember(T.protein1, names);
[~, ib] = ismember(T.protein2, names);
edges = [ia, ib];
n = numel(names); m = size(edges, 1);
adjacency = @(e) sparse([e(:, 1); e(:, 2)], [e(:, 2); e(:, 1)], 1, n, n);

% Hide 10% of the edges, chosen at random
[~, order] = sort(sbp.uniform(11, m));
hidden = edges(order(1:floor(m / 10)), :);
train = edges(order(floor(m / 10) + 1:end), :);

% The same number of non-edges: random pairs not linked in the full network
full_net = adjacency(edges);
u = sbp.uniform(12, 6 * size(hidden, 1));
non_edges = zeros(size(hidden));
seen = false(n);
found = 0; k = 1;
while found < size(hidden, 1)
    a = floor(u(k) * n) + 1; b = floor(u(k + 1) * n) + 1;
    k = k + 2;
    lo = min(a, b); hi = max(a, b);
    if lo ~= hi && full_net(lo, hi) == 0 && ~seen(lo, hi)
        seen(lo, hi) = true;
        found = found + 1;
        non_edges(found, :) = [lo, hi];
    end
end

% Similarity indices, from the training network only
A = adjacency(train);
degree = full(sum(A, 2));
common = A * A;                                   % CN for every pair
w = zeros(n, 1);
w(degree > 1) = 1 ./ log(degree(degree > 1));
adamic_adar = A * spdiags(w, 0, n, n) * A;

positive = scores(hidden, common, adamic_adar, degree);
negative = scores(non_edges, common, adamic_adar, degree);
indices = fieldnames(positive)';
results.hidden = size(hidden, 1);
for i = indices
    results.auc.(i{1}) = auc(positive.(i{1}), negative.(i{1}));
end
results.no_common_neighbour = mean(positive.common_neighbours == 0);
fprintf('%d hidden edges, and as many non-edges\n', size(hidden, 1));
for i = indices
    fprintf('  AUC, %s: %.3f\n', strrep(i{1}, '_', ' '), results.auc.(i{1}));
end
fprintf(['%.1f%% of the hidden edges join nodes with no common neighbour ' ...
    'in the training network\n'], 100 * results.no_common_neighbour);

%% Plot
fig = figure('Position', [100 100 600 400]);
labels = {'common neighbours', 'Jaccard', 'Adamic–Adar', 'preferential attachment'};
bar(categorical(labels, labels), cellfun(@(i) results.auc.(i), indices), ...
    'FaceColor', [0.122 0.467 0.706]);
yline(0.5, ':k'); ylim([0, 1]); ylabel('AUC');
title('Predicting hidden STRING edges'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-link-prediction');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function s = scores(pairs, common, adamic_adar, degree)
x = pairs(:, 1); y = pairs(:, 2);
idx = sub2ind(size(common), x, y);
cn = full(common(idx));
union = degree(x) + degree(y) - cn;
jaccard = zeros(size(cn));
jaccard(union > 0) = cn(union > 0) ./ union(union > 0);
% Round away floating-point noise, so that exact ties stay ties
s.common_neighbours = cn;
s.jaccard = round(jaccard, 12);
s.adamic_adar = round(full(adamic_adar(idx)), 12);
s.preferential_attachment = degree(x) .* degree(y);
end

function a = auc(positive, negative)
% Probability that a positive scores above a negative (ties: 1/2), from
% the average ranks of all the scores (Mann-Whitney).
values = [positive; negative];
[sorted, order] = sort(values);
ranks = zeros(size(values));
i = 1;
while i <= numel(values)                          % average ranks over ties
    j = i;
    while j < numel(values) && sorted(j + 1) == sorted(i)
        j = j + 1;
    end
    ranks(order(i:j)) = (i + j) / 2;
    i = j + 1;
end
p = numel(positive);
a = (sum(ranks(1:p)) - p * (p + 1) / 2) / (p * numel(negative));
end
