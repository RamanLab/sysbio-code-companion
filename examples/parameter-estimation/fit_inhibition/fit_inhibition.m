%% Fitting a rate law to data by weighted least squares
% Competitive inhibition:  v = vmax*S / (KM*(1 + I/KI) + S).
% Rates were measured at 20 substrate concentrations, with the inhibitor
% at I = 10 mM (dataset D1) and at I = 20 mM (dataset D2). We estimate
% vmax, KM and KI from D1 alone, and from both datasets together.
% Requires: Optimization Toolbox (for lsqnonlin).

data = readmatrix(fullfile(sbp.data_dir(), 'competitive_inhibition.csv'));
S = data(:, 1); v10 = data(:, 2); v20 = data(:, 3);
delta = 0.1;                      % relative measurement error (10%)

rate = @(p, S, I) p(1) * S ./ (p(2) * (1 + I / p(3)) + S);
% Weighted residuals; their sum of squares is the SSE (book Eq. 6.5)
residuals = @(p, S, I, v) (rate(p, S, I) - v) ./ (delta * v);
opts = optimoptions('lsqnonlin', 'Display', 'off', ...
    'StepTolerance', 1e-12, 'FunctionTolerance', 1e-12, ...
    'OptimalityTolerance', 1e-12);

% The same eight starting guesses for every fit: vmax = 20 and a spread
% of values for KM and KI
KM_KI = [1 1; 2 8; 5 2; 5 20; 10 5; 20 40; 40 10; 40 80];
starts = [20 * ones(8, 1), KM_KI];

names = {'D1', 'D2', 'D1_D2'};
datasets = {{S, 10 * ones(20, 1), v10}, ...
            {S, 20 * ones(20, 1), v20}, ...
            {[S; S], [10 * ones(20, 1); 20 * ones(20, 1)], [v10; v20]}};

for d = 1:3
    [s, i, v] = datasets{d}{:};
    P = zeros(8, 3);
    SSE = zeros(8, 1);
    for k = 1:8
        [P(k, :), SSE(k)] = lsqnonlin(@(p) residuals(p, s, i, v), ...
            starts(k, :), [0 0 0], [Inf Inf Inf], opts);
    end
    [~, best] = min(SSE);
    % KM*(1 + I/KI) at the first inhibitor level of the dataset
    KM_app = P(:, 2) .* (1 + i(1) ./ P(:, 3));
    fprintf('%s: SSE = %.4f (spread over starts %.1e)\n', names{d}, ...
        SSE(best), max(SSE) - min(SSE));
    fprintf(['   vmax = %.4f;  KM ranges %.3g to %.3g;  ' ...
        'KI ranges %.3g to %.3g\n'], P(best, 1), min(P(:, 2)), ...
        max(P(:, 2)), min(P(:, 3)), max(P(:, 3)));
    % What every start agrees on; KM and KI only if they are pinned down
    results.(names{d}) = struct('SSE', SSE(best), 'vmax', P(best, 1), ...
        'KM_app', KM_app(best));
    if strcmp(names{d}, 'D1_D2')
        results.D1_D2.KM = P(best, 2);
        results.D1_D2.KI = P(best, 3);
    else
        results.([names{d} '_estimates']) = struct('KM', P(:, 2)', ...
            'KI', P(:, 3)');
    end
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
blue = [0.122 0.467 0.706]; orange = [1.000 0.498 0.055];
r = results.D1_D2;
p = [r.vmax r.KM r.KI];
S_fine = linspace(0, 21, 200);

subplot(1, 2, 1)
hold on
errorbar(S, v10, delta * v10, 'o', 'Color', blue, 'MarkerSize', 4, ...
    'MarkerFaceColor', blue);
errorbar(S, v20, delta * v20, 'o', 'Color', orange, 'MarkerSize', 4, ...
    'MarkerFaceColor', orange);
plot(S_fine, rate(p, S_fine, 10), '-', 'Color', blue);
plot(S_fine, rate(p, S_fine, 20), '-', 'Color', orange);
xlabel('S (mM)'); ylabel('v (mmol L^{-1} s^{-1})');
title('Fit to D1 and D2 together');
legend({'data, I = 10 mM', 'data, I = 20 mM'}, 'Location', 'southeast', ...
    'Box', 'off');
box off

subplot(1, 2, 2)
KI_line = logspace(-1, 2.3, 200);
loglog(results.D1.KM_app ./ (1 + 10 ./ KI_line), KI_line, '-', ...
    'Color', [0.6 0.6 0.6]);
hold on
est = results.D1_estimates;
loglog(est.KM, est.KI, 'o', 'Color', blue, 'MarkerFaceColor', blue);
loglog(p(2), p(3), 'p', 'Color', [0.839 0.153 0.157], ...
    'MarkerFaceColor', [0.839 0.153 0.157], 'MarkerSize', 14);
xlabel('K_M (mM)'); ylabel('K_I (mM)');
title('Where the estimates land');
legend({'K_M(1 + 10/K_I) = best K_M^{app} for D1', ...
    'estimates from D1, 8 starts', 'estimate from D1 and D2'}, ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 9);
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-fit-inhibition');

sbp.save_results(mfilename('fullpath'), results);
