%% Solving dx/dt = a - b*x with the forward and backward Euler methods
% This simple ODE has an exact solution, so we can see how the step size
% dt affects accuracy and, for forward Euler, stability.
% Requires: core MATLAB only.

a = 20; b = 2; c = 5;   % dx/dt = a - b*x,  x(0) = c
t_end = 4;
exact = @(t) a/b - (a/b - c) * exp(-b*t);

%% Forward Euler with different step sizes
steps = [0.05 0.25 0.9 1.1];
max_error = zeros(size(steps));
for k = 1:numel(steps)
    [t, x] = forward_euler(a, b, c, t_end, steps(k));
    max_error(k) = max(abs(x - exact(t)));
    fprintf('forward Euler, dt = %4g: max error = %.4g\n', ...
        steps(k), max_error(k));
end
fprintf('Forward Euler is stable only for dt < 2/b = %g\n', 2/b);

%% Forward vs backward Euler at a step size beyond the stability limit
[t_f, x_f] = forward_euler(a, b, c, t_end, 1.1);
[t_b, x_b] = backward_euler(a, b, c, t_end, 1.1);

%% Plot
colors = [0.000 0.447 0.698; 0.000 0.620 0.451;
          0.902 0.624 0.000; 0.835 0.369 0.000];
t_fine = linspace(0, 4.4, 441);
fig = figure('Position', [100 100 1000 400]);

subplot(1, 2, 1)
plot(t_fine, exact(t_fine), 'k-', 'LineWidth', 2);
hold on
for k = 1:3                     % dt = 1.1 is in the right panel
    [t, x] = forward_euler(a, b, c, t_end, steps(k));
    plot(t, x, 'o-', 'Color', colors(k, :), 'MarkerSize', 4);
end
ylim([0 20]);
xlabel('t'); ylabel('x(t)'); title('Forward Euler');
legend(['exact', compose('\\Deltat = %g', steps(1:3))], ...
    'Location', 'southeast', 'Box', 'off');
box off

subplot(1, 2, 2)
plot(t_fine, exact(t_fine), 'k-', 'LineWidth', 2);
hold on
plot(t_f, x_f, 'o-', 'Color', colors(4, :), 'MarkerSize', 4);
plot(t_b, x_b, 's-', 'Color', colors(1, :), 'MarkerSize', 4);
ylim([-2 25]);
xlabel('t'); title('Forward vs backward Euler');
legend({'exact', 'forward, \Deltat = 1.1', 'backward, \Deltat = 1.1'}, ...
    'Location', 'northwest', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-euler');

results.steps = steps;
results.max_error_forward = max_error;
results.forward_dt_1_1 = x_f;
results.backward_dt_1_1 = x_b;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function [t, x] = forward_euler(a, b, c, t_end, dt)
% x(t + dt) = x(t) + f(x(t)) * dt
n = round(t_end / dt);
t = dt * (0:n);
x = zeros(1, n + 1);
x(1) = c;
for i = 1:n
    x(i + 1) = x(i) + (a - b * x(i)) * dt;
end
end

function [t, x] = backward_euler(a, b, c, t_end, dt)
% x(t + dt) = x(t) + f(x(t + dt)) * dt, solved for x(t + dt).
% For a linear f this can be done by hand:
% x(t + dt) = (x(t) + a*dt) / (1 + b*dt).
n = round(t_end / dt);
t = dt * (0:n);
x = zeros(1, n + 1);
x(1) = c;
for i = 1:n
    x(i + 1) = (x(i) + a * dt) / (1 + b * dt);
end
end
