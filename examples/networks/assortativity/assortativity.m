%% Degree assortativity, and comparing a network with randomised versions
% Assortativity (book §3.1.3) is the correlation between the degrees of the
% two nodes at either end of an edge: positive when hubs link to hubs,
% negative when hubs link to sparsely connected nodes. To see whether a
% value says something about a network, rather than about its degrees
% alone, we compare it with randomised networks that keep every node's
% degree (book §3.4.1). Each randomisation step picks two edges, a-b and
% c-d, and rewires them to a-d and c-b, unless that would create a
% self-loop or an edge that already exists. The random choices come from
% sbp.uniform, so that MATLAB and Python make exactly the same swaps.
%
% Two networks: Zachary's karate club, and the E. coli STRING network
% (combined score at least 700).

files = {'karate_edges.csv', 'ecoli_string_ppi.csv'};
columns = {{'member1', 'member2'}, {'protein1', 'protein2'}};
keys = {'karate', 'string'};
copies = [20, 3];
results = struct();
for net = 1:2
    [edges, n] = load_edges(files{net}, columns{net});
    k0 = accumarray(edges(:), 1, [n, 1]);
    r_rand = zeros(1, copies(net)); c_rand = r_rand;
    for seed = 1:copies(net)
        shuffled = randomise(edges, n, 1000 + seed, 10);
        assert(isequal(accumarray(shuffled(:), 1, [n, 1]), k0));
        r_rand(seed) = degree_correlation(shuffled, n);
        c_rand(seed) = clustering(shuffled, n);
    end
    r = struct('nodes', n, 'edges', size(edges, 1), ...
        'assortativity', degree_correlation(edges, n), ...
        'clustering', clustering(edges, n), ...
        'assortativity_random', r_rand, 'clustering_random', c_rand);
    results.(keys{net}) = r;
    fprintf('%s: %d nodes, %d edges\n', keys{net}, n, size(edges, 1));
    fprintf(['  assortativity %+.3f; randomised %+.3f (range %+.3f to ' ...
        '%+.3f, %d networks)\n'], r.assortativity, mean(r_rand), ...
        min(r_rand), max(r_rand), copies(net));
    fprintf('  mean clustering %.3f; randomised %.3f\n', r.clustering, ...
        mean(c_rand));
end

%% Plot
fig = figure('Position', [100 100 900 400]);
measures = {'assortativity', 'clustering'};
labels = {'Assortativity r', 'Mean clustering'};
for p = 1:2
    subplot(1, 2, p); hold on
    for net = 1:2
        r = results.(keys{net});
        h1 = plot(net - 1, r.(measures{p}), 'o', 'Color', [0.839 0.153 0.157], ...
            'MarkerFaceColor', [0.839 0.153 0.157], 'MarkerSize', 9);
        random = r.([measures{p} '_random']);
        h2 = plot(repmat(net - 0.85, size(random)), random, '_', ...
            'Color', [0.122 0.467 0.706], 'MarkerSize', 12);
    end
    yline(0, 'k');
    set(gca, 'XTick', [0, 1], 'XTickLabel', {'karate club', 'E. coli STRING'});
    xlim([-0.5, 1.5]); title(labels{p}); box off
    if p == 1, legend([h1, h2], {'network', 'randomised'}, 'Box', 'off'); end
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-assortativity');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function [edges, n] = load_edges(file, columns)
% Edges as pairs of node indices; nodes numbered in sorted order.
T = readtable(fullfile(sbp.data_dir(), file), 'TextType', 'string');
a = string(T.(columns{1})); b = string(T.(columns{2}));
names = unique([a; b]);                          % sorted
[~, ia] = ismember(a, names);
[~, ib] = ismember(b, names);
edges = [ia, ib];
n = numel(names);
end

function r = degree_correlation(edges, n)
% Newman's degree correlation coefficient.
k = accumarray(edges(:), 1, [n, 1]);
j = k(edges(:, 1)); l = k(edges(:, 2));
m = mean((j + l) / 2);
r = (mean(j .* l) - m ^ 2) / (mean((j .^ 2 + l .^ 2) / 2) - m ^ 2);
end

function c = clustering(edges, n)
% Mean clustering coefficient (nodes of degree < 2 count as 0).
A = sparse(edges(:, 1), edges(:, 2), 1, n, n);
A = A + A';
k = full(sum(A, 2));
triangles = full(sum((A * A) .* A, 2)) / 2;
cc = zeros(n, 1);
ok = k > 1;
cc(ok) = 2 * triangles(ok) ./ (k(ok) .* (k(ok) - 1));
c = mean(cc);
end

function edges = randomise(edges, n, seed, swaps_per_edge)
% Degree-preserving randomisation by double edge swaps.
m = size(edges, 1);
A = false(n);
A(sub2ind([n, n], edges(:, 1), edges(:, 2))) = true;
A(sub2ind([n, n], edges(:, 2), edges(:, 1))) = true;
u = sbp.uniform(seed, 3 * swaps_per_edge * m);
for t = 0:swaps_per_edge * m - 1
    e1 = floor(u(3 * t + 1) * m) + 1;
    e2 = floor(u(3 * t + 2) * m) + 1;
    a = edges(e1, 1); b = edges(e1, 2);
    c = edges(e2, 1); d = edges(e2, 2);
    if u(3 * t + 3) < 0.5                        % either orientation of c-d
        [c, d] = deal(d, c);
    end
    if numel(unique([a, b, c, d])) < 4 || A(a, d) || A(c, b)
        continue                                 % self-loop or existing edge
    end
    A(a, b) = false; A(b, a) = false; A(c, d) = false; A(d, c) = false;
    A(a, d) = true; A(d, a) = true; A(c, b) = true; A(b, c) = true;
    edges(e1, :) = [a, d];
    edges(e2, :) = [c, b];
end
end
