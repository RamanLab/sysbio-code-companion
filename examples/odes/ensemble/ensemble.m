%% Handling uncertainty with an ensemble of models
% Parameters estimated from data are uncertain, and so are the predictions
% made with them. One way to see how uncertain is to build an ensemble:
% many parameter sets that are all consistent with the data, each making
% its own prediction (book §5.6.2). Predictions on which the members agree
% can be trusted; those on which they disagree cannot, however good the fit.
%
% The model and data are those of fit_timecourse (gene expression, mRNA and
% protein measured for 12 h). Near the best fit, the parameters (in logs)
% are approximately Gaussian, with covariance (J'J)^-1, where J is the
% Jacobian of the weighted residuals (see sloppiness, §6.5.2). We draw 200
% parameter sets from this distribution, and compare their predictions:
%   * the protein level at 12 h (within the data) and at 48 h (beyond it);
%   * the protein level at 48 h if translation is halved;
%   * the half-lives of the mRNA and of the protein.
% The Gaussian numbers come from sbp.uniform (Box-Muller), so that the
% ensemble is the same in MATLAB and Python.

theta = [9.4984, 0.96342, 5.2262, 0.20479];       % fit_timecourse
data = readmatrix(fullfile(sbp.data_dir(), 'gene_expression_timecourse.csv'));
t_obs = data(:, 1)'; m_obs = data(:, 2)'; p_obs = data(:, 3)';
delta = 0.1;
resid = @(log_p) residuals(exp(log_p), t_obs, m_obs, p_obs, delta);

% Covariance of log(theta): (J'J)^-1, J by central differences
log_theta = log(theta);
J = zeros(2 * numel(t_obs), 4);
for j = 1:4
    up = log_theta; up(j) = up(j) + 1e-5;
    down = log_theta; down(j) = down(j) - 1e-5;
    J(:, j) = (resid(up) - resid(down)) / 2e-5;
end
L = chol(inv(J' * J), 'lower');

% 200 parameter sets: log(theta) + L z, z standard Gaussian
members = 200;
u = sbp.uniform(31, 2 * 4 * members);
z = sqrt(-2 * log(u(1:2:end))) .* cos(2 * pi * u(2:2:end));
z = reshape(z, 4, members)';                      % members x parameters
members_theta = exp(log_theta + z * L');

t = [12, 48];
names = {'protein_12h', 'protein_48h', 'protein_48h_half_translation', ...
    'mrna_half_life', 'protein_half_life'};
pred = zeros(members, numel(names));
for k = 1:members
    p = members_theta(k, :);
    x = simulate(p, t);
    halved = simulate(p .* [1, 1, 0.5, 1], t);
    pred(k, :) = [x(2, 1), x(2, 2), halved(2, 2), log(2) / p(2), log(2) / p(4)];
end

fprintf('Prediction (200 members)          median   5%% to 95%%       spread\n');
for c = 1:numel(names)
    s = sort(pred(:, c));
    % median, and the 10th and 190th of the 200 sorted values (the 5th
    % and 95th percentiles)
    r = struct('median', median(s), 'low', s(10), 'high', s(190), ...
        'spread_percent', 100 * (s(190) - s(10)) / median(s));
    results.(names{c}) = r;
    fprintf('  %-30s %7.2f   %6.2f to %6.2f   %5.1f%%\n', names{c}, ...
        r.median, r.low, r.high, r.spread_percent);
end

%% Plot: the ensemble of protein time courses
fig = figure('Position', [100 100 650 450]); hold on
tt = linspace(0, 48, 97);
for k = 1:50
    x = simulate(members_theta(k, :), tt(2:end));
    plot(tt, [0, x(2, :)], 'Color', [0.122 0.467 0.706 0.15]);
end
h = plot(t_obs, p_obs, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
xline(12, ':k');
xlabel('Time (h)'); ylabel('Protein (molecules per cell)');
title('50 members of the ensemble');
legend(h, 'data (12 h)', 'Box', 'off'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-ensemble');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function x = simulate(p, t)
% Rows: mRNA, protein, at the times t (all > 0).
rhs = @(~, y) [p(1) - p(2) * y(1); p(3) * y(1) - p(4) * y(2)];
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
[~, y] = ode45(rhs, [0, t], [0; 0], opts);
x = y(2:end, :)';
end

function r = residuals(p, t, m_obs, p_obs, delta)
x = simulate(p, t);
r = [(x(1, :) - m_obs) ./ (delta * m_obs), (x(2, :) - p_obs) ./ (delta * p_obs)]';
end
