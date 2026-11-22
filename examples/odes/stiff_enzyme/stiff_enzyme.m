%% Stiffness: when fast binding makes a simple model expensive to solve
% The same enzyme model as before, but now binding and unbinding (k1, k_1)
% are much faster than catalysis (k2). The fast and slow time scales make
% the ODEs 'stiff'. We compare a non-stiff solver (ode45) with a stiff
% one (ode15s) by counting how often each one evaluates the right-hand
% side of the ODEs.
% Requires: core MATLAB only.

k2 = 1.0; E0 = 0.1; S0 = 10.0; t_end = 50.0;
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);

% y = [S; ES; P]
enzyme = @(t, y, k1, k_1) [-k1*(E0 - y(2))*y(1) + k_1*y(2)
                           k1*(E0 - y(2))*y(1) - (k_1 + k2)*y(2)
                           k2*y(2)];

%% How the cost grows as binding gets faster (k1 = k_1 = k)
speeds = [1 10 100 1000];
cost_nonstiff = zeros(size(speeds));
cost_stiff = zeros(size(speeds));
final_P = zeros(size(speeds));
for i = 1:numel(speeds)
    k = speeds(i);
    f = @(t, y) enzyme(t, y, k, k);
    nonstiff = ode45(f, [0 t_end], [S0; 0; 0], opts);
    stiff = ode15s(f, [0 t_end], [S0; 0; 0], opts);
    cost_nonstiff(i) = nonstiff.stats.nfevals;
    cost_stiff(i) = stiff.stats.nfevals;
    final_P(i) = stiff.y(3, end);
    fprintf(['k = %5g: ode45 used %7d evaluations, ode15s %5d;  ' ...
        'P(%g) = %.4f mM\n'], k, cost_nonstiff(i), cost_stiff(i), ...
        t_end, final_P(i));
end

%% The fast transient, for k = 100, on a logarithmic time axis
t = logspace(-4, log10(t_end), 41);
[~, y] = ode15s(@(t, y) enzyme(t, y, 100, 100), [0 t], [S0; 0; 0], opts);
y = y(2:end, :);               % drop t = 0, which we added as the start

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
semilogx(t, y(:, 2) / E0, '-', 'Color', [0.000 0.620 0.451], 'LineWidth', 2);
hold on
semilogx(t, y(:, 3) / S0, '-', 'Color', [0.000 0.447 0.698], 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Fraction');
title('Two time scales (k_1 = k_{-1} = 100)');
legend({'ES / E_0 (fast)', 'P / S_0 (slow)'}, 'Location', 'northwest', ...
    'Box', 'off');
box off

subplot(1, 2, 2)
loglog(speeds, cost_nonstiff, 'o-', 'Color', [0.902 0.624 0.000]);
hold on
loglog(speeds, cost_stiff, 's-', 'Color', [0.000 0.447 0.698]);
xlabel('Binding rate constants k_1 = k_{-1}');
ylabel('Function evaluations');
title('Cost of solving to t = 50 s');
legend({'non-stiff solver', 'stiff solver'}, 'Location', 'northwest', ...
    'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-stiff');

results.speeds = speeds;
results.final_P = final_P;
results.t = t;
results.ES_k100 = y(:, 2)';
results.P_k100 = y(:, 3)';
results.evaluations_nonstiff = cost_nonstiff;
results.evaluations_stiff = cost_stiff;
sbp.save_results(mfilename('fullpath'), results);
