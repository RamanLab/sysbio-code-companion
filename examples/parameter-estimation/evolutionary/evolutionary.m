%% A simple evolutionary algorithm for parameter estimation
% The same model and data as in fit_inhibition (D1 and D2 together), but
% now estimated by a (10 + 40) evolution strategy, which needs no starting
% guess and no gradients:
%   1. start from 50 random parameter sets and keep the best 10 (parents);
%   2. each generation, make 40 children, each a copy of a random parent
%      with small random changes to its parameters ('mutation');
%   3. keep the best 10 of parents and children; shrink the mutations a
%      little; repeat.
% Parameters are searched on a log scale, from 0.1 to about 316.
% Requires: core MATLAB only.

data = readmatrix(fullfile(sbp.data_dir(), 'competitive_inhibition.csv'));
S = [data(:, 1); data(:, 1)];
I = [10 * ones(20, 1); 20 * ones(20, 1)];
v = [data(:, 2); data(:, 3)];
delta = 0.1;
sse = @(log_p) sum(((10^log_p(1) * S ./ ...
    (10^log_p(2) * (1 + I / 10^log_p(3)) + S) - v) ./ (delta * v)) .^ 2);

rng(1);                            % fixed seed: the same run every time
lo = -1.0; hi = 2.5;               % bounds on log10 of each parameter
n_parents = 10; n_children = 40; n_generations = 60;
sigma = 0.5;                       % initial mutation size (log10 units)

% 1. Random start; keep the best 10
pop = lo + (hi - lo) * rand(50, 3);
cost = arrayfun(@(k) sse(pop(k, :)), (1:50)');
[~, order] = sort(cost);
parents = pop(order(1:n_parents), :);
parent_cost = cost(order(1:n_parents));
snapshots = {parents};
snapshot_gens = 0;
best_per_generation = zeros(1, n_generations + 1);
best_per_generation(1) = parent_cost(1);
n_evaluations = 50;

for g = 1:n_generations
    % 2. Children: mutated copies of randomly chosen parents
    pick = randi(n_parents, n_children, 1);
    children = parents(pick, :) + sigma * randn(n_children, 3);
    children = min(max(children, lo), hi);
    child_cost = arrayfun(@(k) sse(children(k, :)), (1:n_children)');
    n_evaluations = n_evaluations + n_children;
    % 3. Survival of the fittest, among parents and children together
    everyone = [parents; children];
    everyone_cost = [parent_cost; child_cost];
    [~, order] = sort(everyone_cost);
    parents = everyone(order(1:n_parents), :);
    parent_cost = everyone_cost(order(1:n_parents));
    sigma = sigma * 0.93;
    best_per_generation(g + 1) = parent_cost(1);
    if ismember(g, [5 20 n_generations])
        snapshots{end + 1} = parents; %#ok<SAGROW>
        snapshot_gens(end + 1) = g; %#ok<SAGROW>
    end
end

p = 10 .^ parents(1, :);
fprintf(['Best after %d generations (%d SSE evaluations): vmax = %.3f, ' ...
    'KM = %.3f, KI = %.3f, SSE = %.4f\n'], n_generations, n_evaluations, ...
    p(1), p(2), p(3), parent_cost(1));

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
semilogy(0:n_generations, best_per_generation, 'o-', ...
    'Color', [0.122 0.467 0.706], 'MarkerSize', 3);
hold on
yline(63.7765, '--', 'Color', [0.6 0.6 0.6]);
xlabel('Generation'); ylabel('Best SSE so far'); title('Convergence');
legend({'', 'least-squares optimum (fit\_inhibition)'}, 'Box', 'off');
box off

subplot(1, 2, 2)
colors = [0.733 0.733 0.733; 1.000 0.498 0.055;
          0.173 0.627 0.173; 0.122 0.467 0.706];
for k = 1:numel(snapshots)
    P = 10 .^ snapshots{k};
    loglog(P(:, 2), P(:, 3), 'o', 'Color', colors(k, :), ...
        'MarkerFaceColor', colors(k, :), 'MarkerSize', 5);
    hold on
end
xlim([0.1 316]); ylim([0.1 316]);
xlabel('K_M (mM)'); ylabel('K_I (mM)');
title('The 10 parents, as the search proceeds');
legend(compose('generation %d', snapshot_gens), 'Location', 'southeast', ...
    'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-evolutionary');

results.best = struct('SSE', parent_cost(1), 'vmax', p(1), 'KM', p(2), ...
    'KI', p(3));
results.n_evaluations = n_evaluations;
results.best_per_generation = best_per_generation;
sbp.save_results(mfilename('fullpath'), results);
