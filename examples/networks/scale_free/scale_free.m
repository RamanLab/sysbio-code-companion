%% Scale-free networks by preferential attachment (Barabási and Albert)
% Nodes are added one at a time; each new node links to m existing nodes,
% chosen with probability proportional to their degree. The resulting
% degree distribution has a heavy tail, P(k) ~ k^-3, with a few highly
% connected hubs. We compare it with an Erdős–Rényi network of the same
% size and mean degree.
% Requires: core MATLAB only.

n = 2000; m = 2;
reps = 10;
rng(1);

k_max = 200;
P_ba = zeros(1, k_max + 1); P_er = zeros(1, k_max + 1);
max_deg_ba = zeros(1, reps); max_deg_er = zeros(1, reps);
for r = 1:reps
    d = degrees(barabasi_albert(n, m), n);
    P_ba = P_ba + histcounts(min(d, k_max), -0.5:k_max + 0.5) / n / reps;
    max_deg_ba(r) = max(d);
    % Erdős–Rényi with the same expected mean degree
    mean_k = mean(d);
    upper = triu(rand(n) < mean_k / (n - 1), 1);
    d = sum(upper | upper', 2)';
    P_er = P_er + histcounts(min(d, k_max), -0.5:k_max + 0.5) / n / reps;
    max_deg_er(r) = max(d);
end

k = 0:k_max;
fprintf('Mean degree %.4f\n', mean_k);
fprintf(['Largest degree, averaged over %d networks: preferential ' ...
    'attachment %.1f, Erdős–Rényi %.1f\n'], reps, mean(max_deg_ba), ...
    mean(max_deg_er));
fprintf(['Fraction of nodes with degree >= 10: %.4f (theory for large ' ...
    'n: %.4f); Erdős–Rényi %.4f\n'], sum(P_ba(11:end)), ...
    m * (m + 1) / (10 * 11), sum(P_er(11:end)));

%% Plot. Right: the fraction of nodes with degree at least k, which is
% much less noisy in the tail than P(k) itself
ccdf_ba = fliplr(cumsum(fliplr(P_ba)));
ccdf_er = fliplr(cumsum(fliplr(P_er)));
blue = [0.122 0.467 0.706]; orange = [1.000 0.498 0.055];
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
plot(k, P_ba, 'o', 'Color', blue, 'MarkerFaceColor', blue, 'MarkerSize', 4);
hold on
plot(k, P_er, 's', 'Color', orange, 'MarkerFaceColor', orange, ...
    'MarkerSize', 4);
xlim([-0.5 30]);
xlabel('Degree k'); ylabel('Fraction of nodes with degree k');
title('Degree distribution');
legend({'preferential attachment', 'Erdős–Rényi'}, 'Box', 'off');
box off

subplot(1, 2, 2)
ok = ccdf_ba > 0 & k > 0;
loglog(k(ok), ccdf_ba(ok), 'o', 'Color', blue, 'MarkerFaceColor', blue, ...
    'MarkerSize', 4);
hold on
ok = ccdf_er > 0 & k > 0;
loglog(k(ok), ccdf_er(ok), 's', 'Color', orange, ...
    'MarkerFaceColor', orange, 'MarkerSize', 4);
kk = k(m + 1:end);
loglog(kk, m * (m + 1) ./ (kk .* (kk + 1)), '-', 'Color', [0.6 0.6 0.6]);
ylim([1e-5 1.5]);
xlabel('Degree k'); ylabel('Fraction of nodes with degree \geq k');
title('Tail, on logarithmic axes');
legend({'preferential attachment', 'Erdős–Rényi', ...
    'theory, m(m+1)/k(k+1)'}, 'Location', 'southwest', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-scale-free');

results.mean_degree = mean_k;
results.ba = struct('P2', P_ba(3), 'P3', P_ba(4), ...
    'P_ge_10', sum(P_ba(11:end)), 'mean_max_degree', mean(max_deg_ba));
results.er = struct('mean_max_degree', mean(max_deg_er));
results.er_P_ge_10 = sum(P_er(11:end));
results.P_ba = P_ba(1:51);
results.P_er = P_er(1:51);
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function edges = barabasi_albert(n, m)
% Edge list of a preferential-attachment network. 'ends' lists every
% edge end, so a node appears in it once per link: picking a uniformly
% random element of 'ends' picks a node with probability proportional
% to its degree. Nodes are numbered from 0, as in the Python version.
n_edges = m * (m + 1) / 2 + m * (n - m - 1);
edges = zeros(n_edges, 2);
e = 0;
for i = 0:m                                % start from a clique
    for j = 0:i - 1
        e = e + 1;
        edges(e, :) = [i j];
    end
end
ends = reshape(edges(1:e, :)', 1, []);
ends = [ends, zeros(1, 2 * n_edges - numel(ends))];
n_ends = 2 * e;
for new = m + 1:n - 1
    targets = [];
    while numel(targets) < m
        t = ends(randi(n_ends));
        if ~ismember(t, targets)
            targets(end + 1) = t; %#ok<AGROW>
        end
    end
    for t = targets
        e = e + 1;
        edges(e, :) = [new t];
        ends(n_ends + 1:n_ends + 2) = [new t];
        n_ends = n_ends + 2;
    end
end
end

function d = degrees(edges, n)
d = accumarray(edges(:) + 1, 1, [n 1])';
end
