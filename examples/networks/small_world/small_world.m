%% Small-world networks (Watts and Strogatz, 1998)
% Start from a ring lattice of n nodes, each linked to its k nearest
% neighbours on either side, and rewire each edge with probability p. A
% little rewiring makes paths short, as in a random network, while keeping
% the high clustering of the lattice. We also compare with Erdős–Rényi
% random networks with the same numbers of nodes and edges.
% Requires: core MATLAB only.

n = 500; k = 5;                   % 500 nodes, 2k = 10 neighbours each
probs = logspace(-4, 0, 13);      % rewiring probabilities
reps = 10;                        % networks averaged for each p
rng(1);

[C0, L0] = clustering_and_path_length(ring_lattice(n, k));
C = zeros(reps, numel(probs));
L = zeros(reps, numel(probs));
for r = 1:reps
    for i = 1:numel(probs)
        [C(r, i), L(r, i)] = clustering_and_path_length( ...
            watts_strogatz(n, k, probs(i)));
    end
end
C_ratio = mean(C, 1) / C0;
L_ratio = mean(L, 1) / L0;

p_er = n * k / (n * (n - 1) / 2);  % same expected number of edges, n*k
er = zeros(reps, 2);
for r = 1:reps
    [er(r, 1), er(r, 2)] = clustering_and_path_length(erdos_renyi(n, p_er));
end
er = mean(er, 1);

fprintf('Ring lattice: C = %.4f, L = %.4f\n', C0, L0);
for i = 1:numel(probs)
    fprintf('  p = %8.4f: C(p)/C(0) = %.3f, L(p)/L(0) = %.3f\n', ...
        probs(i), C_ratio(i), L_ratio(i));
end
fprintf('Erdős–Rényi, same n and m: C = %.4f, L = %.4f\n', er(1), er(2));

%% Plot
fig = figure;
semilogx(probs, C_ratio, 'o-', 'Color', [0.122 0.467 0.706]);
hold on
semilogx(probs, L_ratio, 's-', 'Color', [1.000 0.498 0.055]);
xlabel('Rewiring probability p'); ylabel('Relative to the ring lattice');
title(sprintf('Watts–Strogatz networks, n = %d, %d neighbours', n, 2 * k));
legend({'C(p) / C(0)', 'L(p) / L(0)'}, 'Location', 'southwest', ...
    'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-small-world');

results.lattice = struct('C', C0, 'L', L0);
for p = [0.1 1]
    i = find(abs(probs - p) < 1e-12);
    name = strrep(sprintf('p_%g', p), '.', '_');
    results.ws.(name) = struct('C_ratio', C_ratio(i), ...
        'L_ratio', L_ratio(i));
end
results.er = struct('C', er(1), 'L', er(2));
results.probs = probs;
results.C_ratio = C_ratio;
results.L_ratio = L_ratio;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function A = ring_lattice(n, k)
A = false(n);
idx = (1:n)';
for j = 1:k
    nb = mod(idx - 1 + j, n) + 1;
    A(sub2ind([n n], idx, nb)) = true;
    A(sub2ind([n n], nb, idx)) = true;
end
end

function A = watts_strogatz(n, k, p)
% Rewire each lattice edge (i, i+j) with probability p, moving its far
% end to a random node, avoiding self-loops and duplicate edges.
A = ring_lattice(n, k);
for j = 1:k
    for i = 1:n
        old = mod(i - 1 + j, n) + 1;
        if A(i, old) && rand < p
            new = randi(n);                 % draw until the new end is
            while new == i || A(i, new)     % neither i nor a neighbour
                new = randi(n);
            end
            A(i, old) = false; A(old, i) = false;
            A(i, new) = true;  A(new, i) = true;
        end
    end
end
end

function A = erdos_renyi(n, p)
upper = triu(rand(n) < p, 1);
A = upper | upper';
end

function [C, L] = clustering_and_path_length(A)
% Mean clustering coefficient, and mean shortest-path length over all
% pairs of nodes that are connected by some path.
A = double(A);
deg = sum(A, 2);
tri = sum((A * A) .* A, 2) / 2;
pairs = deg .* (deg - 1) / 2;
c = zeros(size(deg));
c(pairs > 0) = tri(pairs > 0) ./ pairs(pairs > 0);
C = mean(c);
D = distances(graph(A));
d = D(triu(true(size(A)), 1));
L = mean(d(isfinite(d)));
end
