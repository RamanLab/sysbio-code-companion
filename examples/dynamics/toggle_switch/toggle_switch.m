%% Steady states, stability and the phase plane: a genetic toggle switch
% Two genes repress each other (Gardner, Cantor and Collins, 2000):
%     du/dt = a1/(1 + v^n) - u
%     dv/dt = a2/(1 + u^n) - v
% We find every steady state, classify each by the eigenvalues of the
% Jacobian matrix there, and draw the phase plane.
% Requires: core MATLAB only.

a1 = 10; a2 = 10; n = 2;
toggle = @(t, y) [a1 / (1 + y(2)^n) - y(1); a2 / (1 + y(1)^n) - y(2)];
jacobian = @(u, v) [-1, -a1 * n * v^(n - 1) / (1 + v^n)^2;
                    -a2 * n * u^(n - 1) / (1 + u^n)^2, -1];

% Steady states: on the v-nullcline v = a2/(1 + u^n), so the steady values
% of u are the roots of g(u) = a1/(1 + v(u)^n) - u, which lie in [0, a1].
g = @(u) a1 ./ (1 + (a2 ./ (1 + u.^n)).^n) - u;
grid = linspace(0, a1, 10001);
vals = g(grid);
roots = [];
for k = find(sign(vals(1:end-1)) ~= sign(vals(2:end)))
    u = fzero(g, [grid(k) grid(k + 1)]);
    if isempty(roots) || u - roots(end) > 1e-6  % a root on a grid point is
        roots(end + 1) = u; %#ok<AGROW>          % found twice; keep it once
    end
end
for k = 1:numel(roots)
    u = roots(k);
    v = a2 / (1 + u^n);
    e = sort(real(eig(jacobian(u, v))))';
    states(k) = struct('u', u, 'v', v, 'eigenvalues', e, ...
        'stable', max(e) < 0); %#ok<SAGROW>
    kinds = {'unstable (saddle)', 'stable'};
    fprintf(['Steady state u = %.4f, v = %.4f: eigenvalues %.4f, ' ...
        '%.4f -> %s\n'], ...
        u, v, e(1), e(2), kinds{states(k).stable + 1});
end

%% Trajectories from a grid of starting points (off the diagonal u = v,
% which by symmetry leads to the saddle)
[U0, V0] = ndgrid(linspace(0.25, 11.75, 8), linspace(0.5, 12, 8));
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
trajs = cell(numel(U0), 1);
ends_high_u = false(numel(U0), 1);
for k = 1:numel(U0)
    [~, trajs{k}] = ode45(toggle, linspace(0, 20, 401), [U0(k) V0(k)], opts);
    ends_high_u(k) = trajs{k}(end, 1) > trajs{k}(end, 2);
end

%% Plot
fig = figure('Position', [100 100 600 550]);
[U, V] = meshgrid(linspace(0, 12, 25));
quiver(U, V, a1 ./ (1 + V.^n) - U, a2 ./ (1 + U.^n) - V, ...
    'Color', [0.8 0.8 0.8]);
hold on
for k = 1:numel(trajs)
    if ends_high_u(k), c = [0.122 0.467 0.706]; else, c = [1 0.498 0.055]; end
    plot(trajs{k}(:, 1), trajs{k}(:, 2), '-', 'Color', c, 'LineWidth', 0.8);
end
x = linspace(0, 12, 400);
h1 = plot(a1 ./ (1 + x.^n), x, 'k-', 'LineWidth', 2);
h2 = plot(x, a2 ./ (1 + x.^n), 'k--', 'LineWidth', 2);
for k = 1:numel(states)
    if states(k).stable, fc = 'k'; else, fc = 'w'; end
    plot(states(k).u, states(k).v, 'o', 'MarkerSize', 10, ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', fc, 'LineWidth', 1.5);
end
axis([0 12 0 12]);
xlabel('Repressor 1, u'); ylabel('Repressor 2, v');
title('Two stable states (filled) and a saddle (open)');
legend([h1 h2], {'u-nullcline (du/dt = 0)', 'v-nullcline (dv/dt = 0)'}, ...
    'Location', 'northeast', 'FontSize', 9);
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-toggle-switch');

results.steady_states = states;
results.fraction_to_high_u = mean(ends_high_u);
sbp.save_results(mfilename('fullpath'), results);
