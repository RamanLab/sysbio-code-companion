%% Biochemical systems theory: an S-system approximation of a pathway
% A two-step pathway with feedback inhibition, written with familiar rate
% laws (book §5.3):
%     v1 = V1 / (1 + (X2/Ki)^h)       (supply, inhibited by the end product)
%     v2 = V2 X1 / (K2 + X1)          (conversion of X1 to X2)
%     v3 = V3 X2 / (K3 + X2)          (demand for X2)
%     dX1/dt = v1 - v2,   dX2/dt = v2 - v3.
% In the S-system form of BST (book Eq. 5.21), each rate is replaced by a
% power law, v ~ gamma X^f, whose kinetic order f is the elasticity
% d ln v / d ln X at the steady state, and gamma matches the rate there:
%     dX1/dt = a1 X2^g12 - b1 X1^h11,   dX2/dt = a2 X1^g21 - b2 X2^h22.
% We compare the two models when the supply V1 is raised by 10%, 50% and
% 200%. Note that the steady state of an S-system can be found with linear
% algebra, since in logarithms its steady-state equations are linear.

p.V1 = 2.0; p.Ki = 1.0; p.h = 2.0;
p.V2 = 3.0; p.K2 = 1.0;
p.V3 = 2.5; p.K3 = 1.0;

xs = steady_state(p, p.V1);
x1s = xs(1); x2s = xs(2);
v = rates(p, x1s, x2s, p.V1);
flux = v(3);
% Kinetic orders: elasticities at the steady state
g12 = -p.h * (x2s / p.Ki) ^ p.h / (1 + (x2s / p.Ki) ^ p.h);
h11 = p.K2 / (p.K2 + x1s); g21 = h11;
h22 = p.K3 / (p.K3 + x2s);
% Rate constants: each power law equals the flux at the steady state
a1 = flux / x2s ^ g12;
b1 = flux / x1s ^ h11; a2 = b1;
b2 = flux / x2s ^ h22;
fprintf('Steady state: X1 = %.4f, X2 = %.4f, flux = %.4f\n', x1s, x2s, flux);
fprintf('Kinetic orders: g12 = %.3f, h11 = g21 = %.3f, h22 = %.3f\n', ...
    g12, h11, h22);

bst_model = @(t, x, f) [f * a1 * x(2) ^ g12 - b1 * x(1) ^ h11;
                       a2 * x(1) ^ g21 - b2 * x(2) ^ h22];

factors = [1.1, 1.5, 3.0];
t = linspace(0, 10, 201);
results.steady_state = [x1s, x2s];
results.kinetic_orders = [g12, h11, h22];
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
fig = figure('Position', [100 100 1200 360]);
colours = {[0.122 0.467 0.706], [0.839 0.153 0.157]};
for k = 1:numel(factors)
    f = factors(k);
    [~, full] = ode45(@(t, x) full_model(p, x, f * p.V1), t, [x1s; x2s], opts);
    [~, approx] = ode45(@(t, x) bst_model(t, x, f), t, [x1s; x2s], opts);
    exact_ss = steady_state(p, f * p.V1);
    % S-system steady state, in logs: g12 y2 - h11 y1 = ln(b1 / (f a1)),
    % and g21 y1 - h22 y2 = ln(b2 / a2), with y = ln X
    bst_ss = exp([-h11, g12; g21, -h22] \ [log(b1 / (f * a1)); log(b2 / a2)])';
    key = sprintf('supply_up_%d', round(100 * (f - 1)));
    results.(key) = struct('full', exact_ss, 's_system', bst_ss, ...
        'error_percent', 100 * (bst_ss ./ exact_ss - 1));
    fprintf(['Supply x%g: new steady state, full model X1 = %.4f, X2 = %.4f; ' ...
        'S-system X1 = %.4f, X2 = %.4f\n'], f, exact_ss, bst_ss);
    subplot(1, 3, k); hold on
    for i = 1:2
        plot(t, full(:, i), 'Color', colours{i}, ...
            'DisplayName', sprintf('X%d, full model', i));
        plot(t, approx(:, i), '--', 'Color', colours{i}, ...
            'DisplayName', sprintf('X%d, S-system', i));
    end
    title(sprintf('Supply raised %d%%', round(100 * (f - 1))));
    xlabel('Time'); box off
    if k == 1, ylabel('Concentration'); legend('Box', 'off', 'FontSize', 8); end
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-s-system');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function v = rates(p, x1, x2, v1max)
v = [v1max / (1 + (x2 / p.Ki) ^ p.h), p.V2 * x1 / (p.K2 + x1), ...
     p.V3 * x2 / (p.K3 + x2)];
end

function dx = full_model(p, x, v1max)
v = rates(p, x(1), x(2), v1max);
dx = [v(1) - v(2); v(2) - v(3)];
end

function xs = steady_state(p, v1max)
% v1 = v3 fixes X2; then v2 = v3 fixes X1.
g = @(x) v1max / (1 + (x / p.Ki) ^ p.h) - p.V3 * x / (p.K3 + x);
x2 = fzero(g, [1e-9, 100], optimset('TolX', 1e-14));
flux = p.V3 * x2 / (p.K3 + x2);
xs = [flux * p.K2 / (p.V2 - flux), x2];
end
