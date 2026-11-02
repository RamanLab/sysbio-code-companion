%% Stochastic production and degradation of an mRNA
%     0 -> M     at rate k          (transcription)
%     M -> 0     at rate gamma * M  (degradation)
% The deterministic (ODE) model, dM/dt = k - gamma*M, predicts a smooth
% rise to M = k/gamma. With few molecules, each reaction is a random
% event, simulated here with Gillespie's algorithm. At steady state the
% number of molecules follows a Poisson distribution with mean k/gamma.
% Requires: core MATLAB only.

k = 10; gamma = 1;                 % per minute
t_grid = linspace(0, 10, 201);     % minutes
n_runs = 2000;
rng(1);

changes = [+1; -1];
propensities = @(x) [k, gamma * x(1)];

runs = zeros(n_runs, numel(t_grid));
for r = 1:n_runs
    runs(r, :) = sbp.gillespie(0, changes, propensities, t_grid)';
end
final = runs(:, end);
ode = k / gamma * (1 - exp(-gamma * t_grid));

fprintf(['At t = %g min, over %d runs: mean %.3f, variance %.3f ' ...
    '(Poisson: both %g)\n'], t_grid(end), n_runs, mean(final), ...
    var(final, 1), k / gamma);

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
colors = [0.122 0.467 0.706; 0.173 0.627 0.173; 0.580 0.404 0.741];
for r = 1:3
    stairs(t_grid, runs(r, :), 'Color', colors(r, :));
    hold on
end
plot(t_grid, mean(runs, 1), '-', 'Color', [1.000 0.498 0.055], ...
    'LineWidth', 2);
plot(t_grid, ode, 'k--', 'LineWidth', 1.5);
ylim([0 25]);
xlabel('Time (min)'); ylabel('Number of mRNA molecules');
legend({'one simulation', '', '', ...
    sprintf('mean of %d simulations', n_runs), 'ODE'}, ...
    'Location', 'northwest', 'Box', 'off');
title('Trajectories');
box off

subplot(1, 2, 2)
m = 0:25;
counts = histcounts(final, -0.5:25.5) / n_runs;
bar(m, counts, 'FaceColor', [0.122 0.467 0.706], 'FaceAlpha', 0.7, ...
    'EdgeColor', 'none');
hold on
lambda = k / gamma;
plot(m, exp(-lambda) * lambda.^m ./ factorial(m), 'o-', ...
    'Color', [1.000 0.498 0.055], 'MarkerSize', 4, ...
    'MarkerFaceColor', [1.000 0.498 0.055]);
ylim([0 0.17]);
xlabel('Number of mRNA molecules'); ylabel('Probability');
legend({sprintf('simulated, t = %g min', t_grid(end)), ...
    sprintf('Poisson, mean %g', lambda)}, 'Box', 'off');
title('Distribution at steady state');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-birth-death');

results.steady_state = struct('mean', mean(final), ...
    'variance', var(final, 1));
mean_traj = mean(runs, 1);
results.mean_trajectory = mean_traj(1:20:end);
sbp.save_results(mfilename('fullpath'), results);
