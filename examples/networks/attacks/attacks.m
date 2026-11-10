%% Robustness of a network to random failures and targeted attacks
% We remove an increasing fraction of the nodes of the E. coli STRING
% network (combined score at least 700), and follow the size of the
% largest connected component (book §3.5.1). Nodes are removed either in
% random order (failures) or in decreasing order of their degree in the
% intact network (an attack on the hubs; ties broken by name). The random
% order comes from sbp.uniform, so that it is the same in MATLAB and
% Python. For comparison, we do the same for an Erdős–Rényi random graph
% with the same numbers of nodes and edges, also drawn with sbp.uniform.

T = readtable(fullfile(sbp.data_dir(), 'ecoli_string_ppi.csv'), 'TextType', 'string');
names = unique([T.protein1; T.protein2]);         % sorted
[~, ia] = ismember(T.protein1, names);
[~, ib] = ismember(T.protein2, names);
edges = [ia, ib];
n = numel(names); m = size(edges, 1);

steps = 0:19;                                     % remove 0%, 5%, ..., 95%
fractions = steps / 20;
counts = floor(steps * n / 20);                   % whole numbers of nodes
results.fractions = fractions;
labels = {'string', 'random_graph'};
networks = {edges, er_graph(n, m, 7)};
for net = 1:2
    e = networks{net};
    degree = accumarray(e(:), 1, [n, 1]);
    [~, by_degree] = sortrows([-degree, (1:n)']);  % ties: by name
    [~, random_order] = sort(sbp.uniform(2024, n));
    orders = {by_degree, random_order};
    strategies = {'attack', 'failure'};
    for s = 1:2
        curve = zeros(1, numel(counts));
        for c = 1:numel(counts)
            curve(c) = giant_fraction(e, n, orders{s}(1:counts(c)));
        end
        results.(labels{net}).(strategies{s}) = curve;
    end
    r = results.(labels{net});
    fprintf(['%s: largest component after removing 20%%, 50%%, 70%% of ' ...
        'nodes: failures %.2f, %.2f, %.2f; attack %.2f, %.2f, %.2f\n'], ...
        labels{net}, r.failure([5, 11, 15]), r.attack([5, 11, 15]));
end

%% Plot
fig = figure('Position', [100 100 600 450]); hold on
blue = [0.122 0.467 0.706]; red = [0.839 0.153 0.157];
plot(fractions, results.string.failure, '-o', 'Color', blue, 'MarkerSize', 3);
plot(fractions, results.string.attack, '-o', 'Color', red, 'MarkerSize', 3);
plot(fractions, results.random_graph.failure, ':o', 'Color', blue, 'MarkerSize', 3);
plot(fractions, results.random_graph.attack, ':o', 'Color', red, 'MarkerSize', 3);
legend({'STRING, failure', 'STRING, attack', 'random graph, failure', ...
    'random graph, attack'}, 'Box', 'off');
xlabel('Fraction of nodes removed');
ylabel('Largest component (fraction of nodes)'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-attacks');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function e = er_graph(n, m, seed)
% m distinct random edges among n nodes (Erdős–Rényi G(n, m)).
u = sbp.uniform(seed, 4 * m);
A = false(n);
e = zeros(m, 2);
found = 0; k = 1;
while found < m
    a = floor(u(k) * n) + 1; b = floor(u(k + 1) * n) + 1;
    k = k + 2;
    if a ~= b && ~A(min(a, b), max(a, b))
        A(min(a, b), max(a, b)) = true;
        found = found + 1;
        e(found, :) = [min(a, b), max(a, b)];
    end
end
e = sortrows(e);
end

function f = giant_fraction(e, n, removed)
% Size of the largest component, as a fraction of all n nodes.
G = graph(e(:, 1), e(:, 2), [], n);
G = rmnode(G, removed);
if numnodes(G) == 0, f = 0; return; end
f = max(accumarray(conncomp(G)', 1)) / n;
end
