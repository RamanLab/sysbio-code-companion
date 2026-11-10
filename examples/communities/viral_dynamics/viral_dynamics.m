%% Within-host dynamics of an infection: the target-cell-limited model
% The simplest model of viral dynamics in a host (book §11.2.2.2, after
% Perelson): target cells T are infected by virus V, at rate beta T V;
% infected cells I produce virus at rate p and die at rate delta; free
% virus is cleared at rate c.
%     dT/dt = -beta T V
%     dI/dt =  beta T V - delta I
%     dV/dt =  p I - c V
% The parameters are those estimated by Baccam et al. (2006) for influenza
% A in people. The basic reproductive number, R0 = beta p T0 / (c delta),
% is the number of cells infected by the virus from one infected cell, at
% the start. An antiviral (a neuraminidase inhibitor) that blocks virus
% release with a given efficacy multiplies p by (1 - efficacy); we compare
% treatment started on day 1, 2 and 3 of the infection.

par.beta = 2.7e-5; par.p = 1.2e-2; par.c = 3.0; par.delta = 4.0;  % per day
T0 = 4e8; V0 = 9.3e-2;
R0 = par.beta * par.p * T0 / (par.c * par.delta);
fprintf('R0 = %.1f\n', R0);

results.R0 = R0;
names = {'untreated', 'day_1', 'day_2', 'day_3'};
starts = [Inf, 1, 2, 3];
efficacy = 0.98;
fig = figure('Position', [100 100 650 450]);
colours = {[0 0 0], [0.173 0.627 0.173], [0.122 0.467 0.706], ...
           [0.839 0.153 0.157]};
for s = 1:numel(names)
    [t, y] = simulate(par, [T0; 0; V0], starts(s), efficacy);
    T = y(1, :); V = y(3, :);
    [peak, k] = max(V);
    dt = t(2) - t(1);
    % (the day of the peak is printed, but not stored: on a grid of
    % 0.01 days, two solvers could disagree about which point is highest)
    results.(names{s}) = struct('peak_virus', peak, ...
        'target_cells_left', T(end) / T0, ...
        'virus_area', sum((V(2:end) + V(1:end - 1)) / 2) * dt);
    fprintf('%-10s peak %.3g TCID50/mL on day %.2f; %.1f%% of target cells left on day 10\n', ...
        names{s}, peak, t(k), 100 * T(end) / T0);
    label = strrep(strrep(names{s}, '_', ' '), 'day', 'treated, day');
    semilogy(t, max(V, 1e-3), 'Color', colours{s}, 'LineWidth', 1.5, ...
        'DisplayName', label);
    hold on
end
ylim([1e-2, 1e7]); xlabel('Time (days)'); ylabel('Virus (TCID50/mL)');
legend('Box', 'off');
title(sprintf('Influenza A, target-cell-limited model (R0 = %.1f)', R0)); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-viral-dynamics');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function [t, y] = simulate(par, y0, start, efficacy)
% Up to day `start`, untreated; from then on, virus production is reduced
% by the antiviral. Returns T, I and V (rows) every 0.01 days.
t = linspace(0, 10, 1001);
before = t <= start;
y = zeros(3, numel(t));
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-8);
rhs = @(production) @(~, x) [-par.beta * x(1) * x(3);
    par.beta * x(1) * x(3) - par.delta * x(2);
    production * x(2) - par.c * x(3)];
tb = t(before);
[~, x] = ode15s(rhs(par.p), tb, y0, opts);
y(:, before) = x';
if start < t(end)
    [~, x] = ode15s(rhs(par.p * (1 - efficacy)), [start, t(~before)], x(end, :)', opts);
    y(:, ~before) = x(2:end, :)';
end
end
