%% Fitting an ODE model to a time course
% A gene is switched on at t = 0; its mRNA m and protein p follow
%     dm/dt = k_m - g_m m,    dp/dt = k_p m - g_p p,    m(0) = p(0) = 0.
% We estimate the four parameters from measurements of both mRNA and
% protein, and then from the protein alone. Each evaluation of the
% residuals solves the ODEs; the fits start from the same eight guesses.
% Requires: Optimization Toolbox (for lsqnonlin).

data = readmatrix(fullfile(sbp.data_dir(), 'gene_expression_timecourse.csv'));
t = data(:, 1); m_obs = data(:, 2); p_obs = data(:, 3);
delta = 0.1;                      % relative measurement error (10%)

starts = [1 1 1 1; 5 0.5 5 0.5; 20 2 1 0.1; 1 0.1 20 2; 10 0.3 3 1;
          3 3 3 0.3; 50 5 0.5 0.05; 0.5 0.05 50 5];
opts = optimoptions('lsqnonlin', 'Display', 'off', ...
    'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, ...
    'OptimalityTolerance', 1e-12);
names = {'mRNA and protein', 'protein only'};
fits = cell(1, 2);
for f = 1:2
    use_mrna = (f == 1);
    rows = zeros(size(starts, 1), 5);
    for k = 1:size(starts, 1)
        res = @(p) residuals(p, t, m_obs, p_obs, delta, use_mrna);
        [p, sse] = lsqnonlin(res, starts(k, :), 1e-6 * ones(1, 4), ...
            100 * ones(1, 4), opts);
        rows(k, :) = [p sse];
    end
    fits{f} = rows;
    fprintf('Fits to %s (k_m, g_m, k_p, g_p, SSE):\n', names{f});
    fprintf('  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n', rows');
end

% Best fits, and what the protein data alone can determine
both = fits{1};
[~, i] = min(both(:, 5));
best_both = both(i, :);
prot = fits{2};
good = prot(prot(:, 5) < min(prot(:, 5)) + 1e-6, :);   % equally good fits
product = good(:, 1) .* good(:, 3);                     % k_m * k_p
rates = sort(good(:, [2 4]), 2);                        % {g_m, g_p}
fprintf(['Protein only: %d of %d fits reach the best SSE, %.4f; across ' ...
    'them k_m*k_p = %.3f to %.3f, and the two degradation rates are ' ...
    '%.4f and %.4f\n'], size(good, 1), size(prot, 1), min(prot(:, 5)), ...
    min(product), max(product), mean(rates(:, 1)), mean(rates(:, 2)));

%% Plot
blue = [0.122 0.467 0.706]; orange = [1.000 0.498 0.055];
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
tt = linspace(0, 12, 241)';
Y = simulate(best_both(1:4), tt);
plot(t, p_obs, 'o', 'Color', blue, 'MarkerFaceColor', blue);
hold on
plot(tt, Y(:, 2), 'Color', blue);
plot(t, m_obs * 10, 's', 'Color', orange, 'MarkerFaceColor', orange, ...
    'MarkerSize', 4);
plot(tt, Y(:, 1) * 10, 'Color', orange);
xlabel('Time (h)'); ylabel('Molecules per cell');
title('Fit to mRNA and protein');
legend({'protein (data)', 'protein (fit)', 'mRNA × 10 (data)', ...
    'mRNA × 10 (fit)'}, 'Location', 'northwest', 'Box', 'off');
box off
subplot(1, 2, 2)
k = linspace(0.5, 100, 400);
loglog(k, mean(product) ./ k, ':', 'Color', [0.6 0.6 0.6]);
hold on
best = both(:, 5) < min(both(:, 5)) + 1e-6;
loglog(both(best, 1), both(best, 3), 's', 'Color', orange, ...
    'MarkerFaceColor', orange, 'MarkerSize', 8);
loglog(good(:, 1), good(:, 3), 'o', 'Color', blue, ...
    'MarkerFaceColor', blue, 'MarkerSize', 8);
xlim([0.5 100]); ylim([0.5 100]);
xlabel('k_m (molecules/h)'); ylabel('k_p (1/h)');
title('Estimates from eight starting points');
legend({'k_m k_p = constant', 'best fits, mRNA and protein', ...
    'best fits, protein only'}, 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-fit-timecourse');

results.best_both = best_both(1:4);
results.sse_both = best_both(5);
results.sse_protein = min(prot(:, 5));
results.product_protein = mean(product);
results.rates_protein = mean(rates, 1);
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function Y = simulate(p, t)
% Columns: mRNA, protein, at the times t.
rhs = @(~, y) [p(1) - p(2) * y(1); p(3) * y(1) - p(4) * y(2)];
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
[~, Y] = ode45(rhs, [0; t(:)], [0 0], opts);
Y = Y(2:end, :);                  % drop t = 0
end

function r = residuals(p, t, m_obs, p_obs, delta, use_mrna)
Y = simulate(p, t);
r = (Y(:, 2) - p_obs) ./ (delta * p_obs);
if use_mrna
    r = [(Y(:, 1) - m_obs) ./ (delta * m_obs); r];
end
end
