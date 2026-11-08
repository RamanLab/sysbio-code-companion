%% Bistability and hysteresis in the toggle switch
% The same toggle switch, du/dt = a1/(1 + v^n) - u, dv/dt = a2/(1 + u^n) - v,
% now with a2 = 5 fixed and a1, the strength of the promoter making u,
% varied from 1 to 12 (in an experiment, an inducer does this). For each a1
% we find all steady states and their stability. Then we simulate the
% switch while slowly raising a1 and lowering it again: the state the
% switch is in depends on where it has been.
% Requires: core MATLAB only.

a2 = 5; n = 2;
a1_values = round(1:0.05:12, 10);

branches = cell(size(a1_values));
n_states = zeros(size(a1_values));
for k = 1:numel(a1_values)
    branches{k} = steady_states(a1_values(k), a2, n);
    n_states(k) = size(branches{k}, 1);
end
bistable = a1_values(n_states == 3);
fprintf('Bistable (three steady states) for a1 from %.2f to %.2f\n', ...
    min(bistable), max(bistable));

%% Hysteresis: raise a1 step by step, letting the switch settle each
% time from where it was, then lower it again
ramp = round(1:0.25:12, 10);
up = sweep(ramp, [0 a2], a2, n);                   % start with u off
down = flipud(sweep(fliplr(ramp), up(end, :), a2, n));
% u is 'on' when it exceeds v
switch_up = ramp(find(up(:, 1) > up(:, 2), 1));    % first a1 with u on
switch_down = ramp(find(down(:, 1) > down(:, 2), 1));  % lowest a1 with u on
fprintf(['Raising a1, u switches on at a1 = %g; lowering a1, u stays on ' ...
    'down to a1 = %g\n'], switch_up, switch_down);
up = up(:, 1)'; down = down(:, 1)';

%% Plot
fig = figure('Position', [100 100 700 450]);
hold on
xregion = [min(bistable) max(bistable)];
patch(xregion([1 2 2 1]), [-1 -1 13 13], [0.93 0.93 0.93], ...
    'EdgeColor', 'none');
for k = 1:numel(a1_values)
    b = branches{k};
    plot(a1_values(k) * ones(sum(b(:, 2)), 1), b(b(:, 2) == 1, 1), ...
        'k.', 'MarkerSize', 5);
    plot(a1_values(k) * ones(sum(~b(:, 2)), 1), b(b(:, 2) == 0, 1), ...
        '.', 'Color', [0.73 0.73 0.73], 'MarkerSize', 5);
end
h1 = plot(ramp, up, '^', 'Color', [0.122 0.467 0.706]);
h2 = plot(ramp, down, 'v', 'Color', [1.000 0.498 0.055]);
ylim([-0.5 12.5]);
xlabel('Promoter strength a1'); ylabel('Steady-state level of u');
title('Bistability (shaded) and hysteresis');
legend([h1 h2], {'simulated, a1 increasing', 'simulated, a1 decreasing'}, ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 9);
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-hysteresis');

results.bistable_range = [min(bistable) max(bistable)];
results.n_states = n_states;
results.switch_up = switch_up;
results.switch_down = switch_down;
results.up = up;
results.down = down;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function out = steady_states(a1, a2, n)
% All steady values of u, as roots of g(u) on [0, a1], with stability:
% one row [u, is_stable] per steady state.
g = @(u) a1 ./ (1 + (a2 ./ (1 + u.^n)).^n) - u;
grid = linspace(0, a1, 4001);
vals = g(grid);
out = zeros(0, 2);
for k = find(sign(vals(1:end-1)) ~= sign(vals(2:end)))
    u = fzero(g, [grid(k) grid(k + 1)]);
    if ~isempty(out) && u - out(end, 1) <= 1e-6
        continue
    end
    v = a2 / (1 + u^n);
    J = [-1, -a1 * n * v^(n - 1) / (1 + v^n)^2;
         -a2 * n * u^(n - 1) / (1 + u^n)^2, -1];
    out(end + 1, :) = [u, max(real(eig(J))) < 0]; %#ok<AGROW>
end
end

function out = sweep(a1_path, y0, a2, n)
% Settle at each a1 in turn, starting from the previous state.
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
y = y0;
out = zeros(numel(a1_path), 2);
for k = 1:numel(a1_path)
    a1 = a1_path(k);
    f = @(t, y) [a1 / (1 + y(2)^n) - y(1); a2 / (1 + y(1)^n) - y(2)];
    [~, Y] = ode45(f, [0 100], y, opts);
    y = Y(end, :);
    out(k, :) = y;
end
end
