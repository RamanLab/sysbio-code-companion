%% Robust adaptation by integral feedback
% A signal u drives an output y, which is removed by a controller x; y in
% turn makes x, and x is degraded by an enzyme with Michaelis constant K:
%     dy/dt = c u - e x y
%     dx/dt = a y - b x / (K + x)
% When the enzyme is saturated (K small), x integrates the difference
% between a y and b, and y returns to b/a after any change in u, whatever
% the other parameters: perfect adaptation. We compare K = 0.01 with
% K = 1, first in time, then for 81 combinations of a, b, c and e, each
% at half, once and twice its nominal value of 1.
% Requires: core MATLAB only.

K_values = [0.01 1];
u_of_t = @(t) 1 + (t >= 20 && t < 60);   % u doubles at t = 20, falls at 60

%% Time courses with the nominal parameters
t = linspace(0, 100, 2001);
courses = zeros(2, numel(t)); peaks = zeros(1, 2);
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', 0.1);
for k = 1:2
    K = K_values(k);
    y0 = steady_y(1, 1, 1, 1, 1, K);
    x0 = 1 / y0;                         % from c u = e x y
    rhs = @(t, z) [u_of_t(t) - z(2) * z(1); z(1) - z(2) / (K + z(2))];
    [~, Z] = ode15s(rhs, t, [y0 x0], opts);
    courses(k, :) = Z(:, 1)';
    peaks(k) = max(courses(k, t >= 20 & t < 60));
    fprintf(['K = %g: y = %.4f before the step, peak %.4f, %.4f just ' ...
        'before u falls again\n'], K, y0, peaks(k), ...
        courses(k, abs(t - 59.95) < 1e-9));
end

%% 81 parameter sets: how much does the steady-state y change when u
% doubles?
factors = [0.5 1 2];
[e_, c_, b_, a_] = ndgrid(factors);      % a varies slowest, as in Python
params = [a_(:) b_(:) c_(:) e_(:)];
ratio = zeros(81, 2); level = zeros(81, 2);
for k = 1:2
    for i = 1:81
        p = num2cell(params(i, :));
        level(i, k) = steady_y(1, p{:}, K_values(k));
        ratio(i, k) = steady_y(2, p{:}, K_values(k)) / level(i, k);
    end
    fprintf(['K = %g: y(u = 2) / y(u = 1) ranges from %.4f to %.4f; ' ...
        'the level y(u = 1) from %.3f to %.3f\n'], K_values(k), ...
        min(ratio(:, k)), max(ratio(:, k)), min(level(:, k)), ...
        max(level(:, k)));
end

%% Plot
colours = [0.122 0.467 0.706; 1.000 0.498 0.055];
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
patch([20 60 60 20], [0.3 0.3 2.2 2.2], [0.93 0.93 0.93], ...
    'EdgeColor', 'none');
hold on
h = plot(t, courses);
set(h, {'Color'}, num2cell(colours, 2));
text(40, 2.05, 'u = 2', 'HorizontalAlignment', 'center', ...
    'Color', [0.4 0.4 0.4]);
ylim([0.3 2.2]);
xlabel('Time'); ylabel('Output y');
title('Response to a step in the signal');
legend(h, {'K = 0.01', 'K = 1'}, 'Box', 'off');
box off
subplot(1, 2, 2)
hold on
for k = 1:2
    plot(level(:, k), ratio(:, k), 'o', 'Color', colours(k, :), ...
        'MarkerFaceColor', colours(k, :));
end
yline(1, ':', 'Color', [0.6 0.6 0.6]);
set(gca, 'XScale', 'log');
xticks([0.2 0.5 1 2 4]);
xlabel('Steady-state y before the step'); ylabel('y after / y before');
title('81 parameter sets');
legend({'K = 0.01', 'K = 1'}, 'Box', 'off', 'Location', 'northwest');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-integral-feedback');

results.peak = peaks;
results.ratio_K001 = ratio(:, 1)';
results.ratio_K1 = ratio(:, 2)';
results.level_K001 = level(:, 1)';
results.level_K1 = level(:, 2)';
sbp.save_results(mfilename('fullpath'), results);

%% Local function
function y = steady_y(u, a, b, c, e, K)
% At steady state, r = x / (K + x) = a y / b solves
% e K b r^2 + a c u r - a c u = 0.
p = a * c * u;
r = (-p + sqrt(p^2 + 4 * e * K * b * p)) / (2 * e * K * b);
y = r * b / a;
end
