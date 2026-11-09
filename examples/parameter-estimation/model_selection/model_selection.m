%% Model selection: which kind of inhibition do the data support?
% The data are those of fit_inhibition: initial rates at 20 substrate
% concentrations, with an inhibitor at 10 mM and at 20 mM. We fit four
% rate laws, with three or four parameters, by weighted least squares
% (10% relative error), and compare them by
%   * the weighted SSE, chi^2 (book Eq. 6.9, with sigma = 0.1 v);
%   * the Akaike Information Criterion, AIC = chi^2 + 2k, for k parameters,
%     with the correction for small samples, AICc = AIC + 2k(k+1)/(n-k-1);
%   * Akaike weights, the relative support for each model;
%   * the cross-correlation of the parameter estimates (book Eq. 6.11),
%     from the covariance matrix V = (J'J)^-1, where J is the Jacobian of
%     the weighted residuals at the optimum.
% Requires: Optimization Toolbox (for lsqnonlin).

data = readmatrix(fullfile(sbp.data_dir(), 'competitive_inhibition.csv'));
S = [data(:, 1); data(:, 1)];
I = [repmat(10, 20, 1); repmat(20, 20, 1)];
v = [data(:, 2); data(:, 3)];
n = numel(v);
delta = 0.1;

names = {'competitive', 'uncompetitive', 'noncompetitive', 'mixed'};
params = {{'vmax', 'KM', 'KI'}, {'vmax', 'KM', 'KI'}, {'vmax', 'KM', 'KI'}, ...
    {'vmax', 'KM', 'KI', 'KI2'}};
rates = {@(p, S, I) p(1) * S ./ (p(2) * (1 + I / p(3)) + S), ...
         @(p, S, I) p(1) * S ./ (p(2) + S .* (1 + I / p(3))), ...
         @(p, S, I) p(1) * S ./ ((p(2) + S) .* (1 + I / p(3))), ...
         @(p, S, I) p(1) * S ./ (p(2) * (1 + I / p(3)) + S .* (1 + I / p(4)))};

opts = optimoptions('lsqnonlin', 'Display', 'off', ...
    'FunctionTolerance', 1e-14, 'StepTolerance', 1e-14, ...
    'OptimalityTolerance', 1e-14, 'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 1e4);
results = struct();
aicc = zeros(1, numel(names));
for m = 1:numel(names)
    rate = rates{m};
    k = numel(params{m});
    resid = @(p) (rate(p, S, I) - v) ./ (delta * v);
    best_p = []; best_cost = inf;
    for scale_km = [0.2, 1, 5]                   % nine starting guesses
        for scale_ki = [0.2, 1, 5]
            p0 = [20, 5 * scale_km, repmat(5 * scale_ki, 1, k - 2)];
            if k == 4, p0(4) = 10 * p0(4); end
            [p, cost] = lsqnonlin(resid, p0, zeros(1, k), [], opts);
            if cost < best_cost, best_p = p; best_cost = cost; end
        end
    end
    p = best_p;
    chi2 = sum(resid(p) .^ 2);
    J = jacobian(resid, p);
    V = inv(J' * J);
    sd = sqrt(diag(V))';
    kappa = V ./ (sd' * sd);
    aic = chi2 + 2 * k;
    aicc(m) = aic + 2 * k * (k + 1) / (n - k - 1);
    results.(names{m}) = struct('parameters', p, 'sd', sd, 'chi2', chi2, ...
        'aic', aic, 'aicc', aicc(m), 'kappa', kappa);
end

weights = exp(-(aicc - min(aicc)) / 2);
weights = weights / sum(weights);
for m = 1:numel(names)
    r = results.(names{m});
    results.(names{m}).weight = weights(m);
    fprintf('%-15s chi2 = %6.2f  AIC = %6.2f  AICc = %6.2f  weight = %.2f\n', ...
        names{m}, r.chi2, r.aic, r.aicc, weights(m));
    parts = arrayfun(@(j) sprintf('%s = %.3g ± %.2g', params{m}{j}, ...
        r.parameters(j), r.sd(j)), 1:numel(params{m}), 'UniformOutput', false);
    fprintf('   %s\n', strjoin(parts, ', '));
    fprintf('   largest |cross-correlation| = %.3f\n', ...
        max(max(abs(r.kappa - eye(numel(params{m}))))));
end

%% Plot: data and the two best models, and the Akaike weights
fig = figure('Position', [100 100 1000 400]);
subplot(1, 5, 1:3); hold on
s_fine = linspace(0.5, 20, 200)';
markers = {'o', 's'}; levels = [10, 20];
colours = {[0.122 0.467 0.706], [0.839 0.153 0.157]};
shown = {'competitive', 'noncompetitive'};
for l = 1:2
    keep = I == levels(l);
    errorbar(S(keep), v(keep), delta * v(keep), markers{l}, 'Color', 'k', ...
        'MarkerSize', 4, 'DisplayName', sprintf('data, I = %d mM', levels(l)));
    for m = 1:2
        rate = rates{strcmp(names, shown{m})};
        h = plot(s_fine, rate(results.(shown{m}).parameters, s_fine, levels(l)), ...
            'Color', colours{m}, 'DisplayName', shown{m});
        if l == 2, h.HandleVisibility = 'off'; end
    end
end
xlabel('S (mM)'); ylabel('v (mmol/L/s)');
legend('Box', 'off', 'Location', 'southeast'); box off
subplot(1, 5, 4:5)
bar(categorical(names, names), weights, 'FaceColor', [0.122 0.467 0.706]);
ylabel('Akaike weight'); title('Support for each model'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-model-selection');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function J = jacobian(resid, p)
% Central differences of the weighted residuals.
r0 = resid(p);
J = zeros(numel(r0), numel(p));
for j = 1:numel(p)
    h = 1e-6 * p(j);
    up = p; up(j) = up(j) + h;
    down = p; down(j) = down(j) - h;
    J(:, j) = (resid(up) - resid(down)) / (2 * h);
end
end
