%% The Lotka-Volterra predator-prey model (book Eqs. 1.4 and 1.5)
%     dD/dt = b*D - p*L*D      deer (prey): born at rate b, eaten by lions
%     dL/dt = r*D*L - d*L      lions (predators): grow by eating, die at rate d
% The populations oscillate. Each starting point gives its own closed orbit
% around the steady state (D*, L*) = (d/r, b/p), because the quantity
%     V = r*D - d*ln(D) + p*L - b*ln(L)
% stays constant along every solution.
% Requires: core MATLAB only.

b = 1.0; p = 0.1; r = 0.02; d = 0.5;  % per year (p, r per animal per year)
D_star = d / r; L_star = b / p;       % steady state: 25 deer, 10 lions
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);

lotka_volterra = @(t, y) [b * y(1) - p * y(2) * y(1);
                          r * y(1) * y(2) - d * y(2)];
V = @(D, L) r * D - d * log(D) + p * L - b * log(L);

t = linspace(0, 30, 301);
starts = [40 9; 60 9; 80 9];
Y = cell(1, 3);
for k = 1:3
    [~, Y{k}] = ode45(lotka_volterra, t, starts(k, :), opts);
end

% Period of the first orbit: times at which the deer population passes
% upwards through D*, found by the solver as 'events'
crossing = @(t, y) deal(y(1) - D_star, 0, 1);  % value, terminal, direction
[~, ~, te] = ode45(lotka_volterra, [0 30], starts(1, :), ...
    odeset(opts, 'Events', crossing));
period = mean(diff(te));

D = Y{1}(:, 1); L = Y{1}(:, 2);
drift = max(abs(V(D, L) - V(D(1), L(1))));
fprintf('Steady state: D* = %g deer, L* = %g lions\n', D_star, L_star);
fprintf(['Period of the orbit from (40, 9): %.4f years (small ' ...
    'oscillations: 2*pi/sqrt(b*d) = %.4f)\n'], period, 2 * pi / sqrt(b * d));
fprintf('Largest change in V along the solution: %.1e\n', drift);

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
plot(t, D, '-', 'Color', [0.122 0.467 0.706]);
hold on
plot(t, L, '-', 'Color', [0.839 0.153 0.157]);
ylim([0 50]);
xlabel('Time (years)'); ylabel('Population');
title('Starting from 40 deer and 9 lions');
legend({'deer, D', 'lions, L'}, 'Location', 'north', 'Orientation', ...
    'horizontal', 'Box', 'off');
box off

subplot(1, 2, 2)
colors = [0.122 0.467 0.706; 0.173 0.627 0.173; 1.000 0.498 0.055];
for k = 1:3
    plot(Y{k}(:, 1), Y{k}(:, 2), '-', 'Color', colors(k, :));
    hold on
end
plot(D_star, L_star, 'k+', 'MarkerSize', 10, 'LineWidth', 2);
ylim([0 32]);
xlabel('Deer, D'); ylabel('Lions, L'); title('Phase plane');
legend({'start (40, 9)', 'start (60, 9)', 'start (80, 9)', ...
    'steady state'}, 'Location', 'northeast', 'NumColumns', 2, ...
    'Box', 'off', 'FontSize', 9);
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-lotka-volterra');

results.steady_state = [D_star L_star];
results.period = period;
results.V_drift = drift;
results.D_every_5_years = D(1:50:end)';
results.L_every_5_years = L(1:50:end)';
sbp.save_results(mfilename('fullpath'), results);
