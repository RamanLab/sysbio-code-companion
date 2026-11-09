%% Local sensitivity analysis of a gene expression model
% The model is that of fit_timecourse: a gene switched on at t = 0, with
%     dm/dt = k_m - g_m m,    dp/dt = k_p m - g_p p,    m(0) = p(0) = 0,
% at the parameters fitted there to mRNA and protein data. The local
% sensitivity of state x_i to parameter theta_j (book Eq. 6.12) is
% S_ij(t) = dx_i(t)/dtheta_j, which we compute by central differences:
% solve the ODEs with theta_j raised and lowered by 0.01%. We report the
% normalised sensitivity (theta_j / x_i) dx_i/dtheta_j, the percentage
% change in x_i for a 1% change in theta_j, which can be compared across
% parameters of different units.

names = {'k_m', 'g_m', 'k_p', 'g_p'};
theta = [9.4984, 0.96342, 5.2262, 0.20479];      % fit_timecourse
t = linspace(0.5, 24, 48);

x = simulate(theta, t);
normalised = zeros(2, numel(t), numel(theta));    % state x time x parameter
for j = 1:numel(theta)
    h = 1e-4 * theta(j);
    up = theta; up(j) = up(j) + h;
    down = theta; down(j) = down(j) - h;
    dx = (simulate(up, t) - simulate(down, t)) / (2 * h);
    normalised(:, :, j) = theta(j) * dx ./ x;
end

fprintf('Normalised sensitivity of protein to each parameter:\n');
fprintf('   t (h)  %s\n', sprintf('%8s', names{:}));
for k = [1, 4, 12, 24, 48]
    fprintf('  %5.1f   %s\n', t(k), sprintf('%8.3f', squeeze(normalised(2, k, :))));
end
fprintf(['At steady state, p* = k_m k_p / (g_m g_p): the sensitivities ' ...
    'approach +1, -1, +1, -1.\n']);

%% Plot
fig = figure('Position', [100 100 1000 400]);
colours = [0.122 0.467 0.706; 1.000 0.498 0.055; 0.173 0.627 0.173; ...
           0.839 0.153 0.157];
titles = {'mRNA', 'Protein'};
for i = 1:2
    subplot(1, 2, i); hold on
    for j = 1:numel(names)
        plot(t, normalised(i, :, j), 'Color', colours(j, :), 'LineWidth', 1.5);
    end
    yline(0, 'k');
    xlabel('Time (h)'); title(titles{i}); box off
    if i == 1, ylabel('Normalised sensitivity'); end
end
legend(names, 'Box', 'off', 'Interpreter', 'none');
sbp.save_figure(fig, mfilename('fullpath'), 'fig-sensitivity');

results.t = t;
results.mrna = squeeze(normalised(1, :, :));
results.protein = squeeze(normalised(2, :, :));
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function x = simulate(p, t)
% Rows: mRNA, protein, at the times t.
rhs = @(~, y) [p(1) - p(2) * y(1); p(3) * y(1) - p(4) * y(2)];
opts = odeset('RelTol', 1e-11, 'AbsTol', 1e-11);
[~, y] = ode45(rhs, [0, t], [0; 0], opts);
x = y(2:end, :)';
end
