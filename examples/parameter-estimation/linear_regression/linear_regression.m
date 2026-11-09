%% Linearising the Michaelis-Menten equation, and why we should not
% Three transformations turn v = vmax S / (KM + S) into a straight line
% (book §6.2.1):
%   Lineweaver-Burk  1/v vs 1/S:   1/v = (KM/vmax)(1/S) + 1/vmax
%   Eadie-Hofstee    v vs v/S:     v = -KM (v/S) + vmax
%   Hanes-Woolf      S/v vs S:     S/v = S/vmax + KM/vmax
% We estimate vmax and KM from each line, and by non-linear least squares,
% first for the data of Exercise 6.5, and then for 200 simulated datasets
% with known parameters (vmax = 2.5 mM/s, KM = 0.8 mM, noise of standard
% deviation 0.1 mM/s), to see how the transformations distort the errors.
% Requires: Optimization Toolbox (for lsqnonlin).

methods = {'Lineweaver-Burk', 'Eadie-Hofstee', 'Hanes-Woolf', 'non-linear'};

% Exercise 6.5 (the column headers in the book are swapped: S is the
% first row, v the second)
S = [0.15, 0.35, 0.65, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0];
v = [0.35, 0.70, 1.26, 1.62, 1.5, 2.21, 2.167, 2.33, 2.12];
exercise = estimate(S, v);
fprintf('Exercise 6.5:\n');
for k = 1:4
    vm = exercise(k, 1); km = exercise(k, 2);
    sse = sum((vm * S ./ (km + S) - v) .^ 2);
    fprintf('  %-16s vmax = %.3f, KM = %.3f, SSE = %.4f\n', methods{k}, ...
        vm, km, sse);
end

% 200 simulated datasets with known parameters
replicates = readmatrix(fullfile(sbp.data_dir(), 'mm_replicates.csv'));
true_values = [2.5, 0.8];
estimates = zeros(size(replicates, 1), 4, 2);
for r = 1:size(replicates, 1)
    estimates(r, :, :) = estimate(S, replicates(r, :));
end
med = squeeze(median(estimates, 1));                          % 4 x 2
err = squeeze(median(abs(estimates - reshape(true_values, 1, 1, 2)) ...
    ./ reshape(true_values, 1, 1, 2), 1));                    % relative
fprintf('200 simulated datasets (vmax = 2.5, KM = 0.8):\n');
for k = 1:4
    fprintf(['  %-16s median vmax = %.3f, KM = %.3f; median error ' ...
        '%.1f%%, %.1f%%\n'], methods{k}, med(k, 1), med(k, 2), ...
        100 * err(k, 1), 100 * err(k, 2));
end

%% Plot: the Lineweaver-Burk line for Exercise 6.5, and the spread of the
% KM estimates across the simulated datasets
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1); hold on
x = linspace(0, 1 / min(S), 50);
vm = exercise(1, 1); km = exercise(1, 2);
plot(1 ./ S, 1 ./ v, 'ko', 'MarkerFaceColor', 'k');
plot(x, km / vm * x + 1 / vm, 'Color', [0.839 0.153 0.157], 'LineWidth', 1.5);
xlabel('1/S (1/mM)'); ylabel('1/v (s/mM)');
title('Lineweaver-Burk plot, Exercise 6.5'); box off
subplot(1, 2, 2); hold on
groups = categorical(repmat({'LB', 'EH', 'HW', 'non-linear'}, ...
    size(estimates, 1), 1), {'LB', 'EH', 'HW', 'non-linear'});
boxchart(groups(:), reshape(estimates(:, :, 2), [], 1), 'MarkerStyle', 'none');
yline(true_values(2), ':k');
ylabel('Estimated KM (mM)'); title('KM from 200 simulated datasets'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-linear-regression');

results.methods = methods;
results.exercise = exercise;
results.median = med;
results.median_error = err;
results.estimates = estimates;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function est = estimate(S, v)
% vmax and KM by each of the four methods (one row per method).
c = polyfit(1 ./ S, 1 ./ v, 1);
lineweaver_burk = [1 / c(2), c(1) / c(2)];
c = polyfit(v ./ S, v, 1);
eadie_hofstee = [c(2), -c(1)];
c = polyfit(S, S ./ v, 1);
hanes_woolf = [1 / c(1), c(2) / c(1)];
% Non-linear least squares, started from the Hanes-Woolf estimate
opts = optimoptions('lsqnonlin', 'Display', 'off', ...
    'FunctionTolerance', 1e-14, 'StepTolerance', 1e-14, ...
    'OptimalityTolerance', 1e-14);
p = lsqnonlin(@(p) p(1) * S ./ (p(2) + S) - v, hanes_woolf, [], [], opts);
est = [lineweaver_burk; eadie_hofstee; hanes_woolf; p];
end
