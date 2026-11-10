%% Communities by modularity maximisation: Newman's spectral method
% Modularity (book §3.3.1) compares the edges within communities with the
% number expected at random, for the same degrees:
%     Q = (1/2m) sum_ij [A_ij - k_i k_j / 2m] delta(c_i, c_j).
% The matrix B = A - k k'/2m is the modularity matrix. Newman's method
% splits the network in two by the signs of the leading eigenvector of B,
% and then splits each part again, using the "generalised" modularity
% matrix of that part, until no split increases Q (Newman, 2006, PNAS).
% Unlike many other methods, it involves no random choices.

edges = readmatrix(fullfile(sbp.data_dir(), 'karate_edges.csv'));
factions = readtable(fullfile(sbp.data_dir(), 'karate_factions.csv'), ...
    'TextType', 'string');
faction = factions.faction';
n = numel(faction);
A = zeros(n);
for e = 1:size(edges, 1)
    A(edges(e, 1), edges(e, 2)) = 1;
    A(edges(e, 2), edges(e, 1)) = 1;
end
k = sum(A, 2);
two_m = sum(k);
B = A - k * k' / two_m;
Q = @(labels) sum(B .* (labels(:) == labels(:)'), 'all') / two_m;

communities = {}; queue = {1:n}; first_split = {};
while ~isempty(queue)
    group = queue{1}; queue(1) = [];
    parts = split_group(B, group, two_m);
    if isempty(parts)
        communities{end + 1} = group; %#ok<SAGROW>
    else
        if isempty(first_split), first_split = parts; end
        queue = [queue, parts]; %#ok<AGROW>
    end
end

% Label communities 1, 2, ... in order of their lowest-numbered member
[~, order] = sort(cellfun(@min, communities));
communities = communities(order);
labels = zeros(1, n);
for c = 1:numel(communities)
    labels(communities{c}) = c;
end
halves = zeros(1, n);
halves(first_split{1}) = 1; halves(first_split{2}) = 2;
if halves(1) ~= 1, halves = 3 - halves; end      % member 1 in the first half

first = repmat("Officer", 1, n);
first(halves == 1) = "Mr. Hi";
fprintf(['First split: Q = %.4f; it places %d of %d members on the side ' ...
    'they joined (misplaced: %s)\n'], Q(halves), sum(first == faction), n, ...
    strjoin(string(find(first ~= faction)), ', '));
fprintf('Final: %d communities, Q = %.4f\n', numel(communities), Q(labels));
for c = 1:numel(communities)
    fprintf('  community %d: members %s\n', c, strjoin(string(communities{c}), ', '));
end

%% Plot
fig = figure('Position', [100 100 600 500]);
palette = [0.122 0.467 0.706; 0.682 0.780 0.910; 0.839 0.153 0.157; ...
           1.000 0.596 0.588; 0.173 0.627 0.173; 0.596 0.875 0.541];
plot(graph(A), 'Layout', 'force', 'NodeColor', palette(labels, :), ...
    'MarkerSize', 8, 'EdgeAlpha', 0.3);
title(sprintf('%d communities, Q = %.3f', numel(communities), Q(labels)));
axis off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-modularity');

results.halves = halves;
results.q_halves = Q(halves);
results.communities = labels;
results.q = Q(labels);
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function parts = split_group(B, group, two_m)
% Split a group of nodes in two, or return {} if no split increases the
% modularity.
Bg = B(group, group);
Bg = Bg - diag(sum(Bg, 2));                      % generalised modularity matrix
[V, D] = eig((Bg + Bg') / 2);
[value, top] = max(diag(D));
parts = {};
if value < 1e-10, return; end
s = ones(numel(group), 1);
s(V(:, top) < 0) = -1;
if s' * Bg * s / (2 * two_m) < 1e-10, return; end   % the gain in Q
parts = {group(s > 0), group(s < 0)};
end
