%% Over-expression targets by FSEOF, and the production envelope
% Flux Scanning based on Enforced Objective Flux (FSEOF; book Section
% 10.3.1) for succinate secretion by the E. coli core model on glucose:
%   1. find the maximum growth rate, and the maximum succinate secretion;
%   2. enforce succinate secretion at fractions f = 0, 0.1, ..., 0.9 of its
%      maximum, and maximise growth each time;
%   3. pick the reactions whose flux rises steadily with f: candidates for
%      over-expression.
% Each FBA problem has many optimal flux distributions, so at each step we
% take the one with the smallest Euclidean norm, which is unique (as for
% MoMA in Section 9.1). The production envelope (Section 10.6.1) shows the
% largest and smallest succinate secretion possible at each growth rate.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = full(model.S); lb = model.lb; ub = model.ub; ids = model.rxns;
n = numel(ids);
bio = find(strcmp(ids, 'Biomass_Ecoli_core'));
prod = find(strcmp(ids, 'EX_succ_e'));

mu_max = optimum(S, bio, lb, ub, 1);
succ_max = optimum(S, prod, lb, ub, 1);
fprintf(['Maximum growth %.4f /h; maximum succinate secretion %.4f ' ...
    'mmol/gDW/h\n'], mu_max, succ_max);

%% FSEOF
fractions = round(0:0.1:0.9, 10);
growth = zeros(size(fractions)); F = zeros(numel(fractions), n);
for k = 1:numel(fractions)
    lb_f = lb; lb_f(prod) = fractions(k) * succ_max;   % enforce succinate
    mu = optimum(S, bio, lb_f, ub, 1);
    lb_f(bio) = mu * (1 - 1e-9);
    ub_f = ub; ub_f(bio) = mu;
    growth(k) = mu;
    F(k, :) = abs(nearest(S, zeros(n, 1), lb_f, ub_f))';
end
internal = ~startsWith(ids, {'EX_', 'Biomass'})';
steps = diff(F, 1, 1);
up = ids(internal & all(steps > 1e-4, 1))';
down = ids(internal & all(steps < -1e-4, 1))';
fprintf(['Growth falls from %.3f to %.3f /h as succinate secretion is ' ...
    'enforced\n'], growth(1), growth(end));
fprintf('Fluxes that rise with succinate secretion (%d): %s\n', ...
    numel(up), strjoin(up, ', '));
fprintf('Fluxes that fall (%d): %s\n', numel(down), strjoin(down, ', '));

%% Production envelope
mu_grid = linspace(0, mu_max, 21);
env_max = zeros(size(mu_grid)); env_min = zeros(size(mu_grid));
for k = 1:numel(mu_grid)
    lb_m = lb; lb_m(bio) = mu_grid(k) * (1 - 1e-9);
    env_max(k) = optimum(S, prod, lb_m, ub, 1);
    env_min(k) = optimum(S, prod, lb_m, ub, -1);
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
fill([mu_grid fliplr(mu_grid)], [env_min fliplr(env_max)], ...
    [0.776 0.859 0.937], 'EdgeColor', 'none');
hold on
plot(mu_grid, env_max, 'Color', [0.122 0.467 0.706]);
h = plot(growth, fractions * succ_max, 'o', 'Color', [0.839 0.153 0.157], ...
    'MarkerFaceColor', [0.839 0.153 0.157]);
xlabel('Growth rate (1/h)'); ylabel('Succinate secretion (mmol/gDW/h)');
title('Production envelope');
legend(h, {'FSEOF steps'}, 'Box', 'off');
box off
subplot(1, 2, 2)
show = {'PPC', 'PFK', 'GAPD', 'SUCCt3', 'CS', 'AKGDH'};
hold on
for k = 1:numel(show)
    plot(fractions * succ_max, F(:, strcmp(ids, show{k})), 'o-', ...
        'MarkerSize', 3);
end
ylim([0 26]);
xlabel('Enforced succinate secretion (mmol/gDW/h)');
ylabel('|flux| (mmol/gDW/h)');
title('Fluxes as succinate is enforced');
legend(show, 'Location', 'northwest', 'NumColumns', 2, 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-fseof');

results.growth_max = mu_max;
results.succinate_max = succ_max;
results.fseof_growth = growth;
results.up = up;
results.down = down;
results.envelope_growth = mu_grid;
results.envelope_max = env_max;
results.envelope_min = env_min;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function f = optimum(S, j, lb, ub, sense)
% Largest (sense = 1) or smallest (sense = -1) value of flux j.
c = zeros(size(S, 2), 1); c(j) = -sense;
opts = optimoptions('linprog', 'Display', 'none');
[~, f] = linprog(c, [], [], S, zeros(size(S, 1), 1), lb, ub, opts);
f = -sense * f;
end

function v = nearest(S, target, lb, ub)
% argmin ||v - target||^2 with S v = 0 and lb <= v <= ub. Dependent rows
% of S are dropped, and the active-set algorithm makes quadprog exact.
n = numel(lb);
[~, R, order] = qr(S', 'vector');
S = S(sort(order(abs(diag(R)) > 1e-9 * abs(R(1, 1)))), :);
opts = optimoptions('quadprog', 'Algorithm', 'active-set', ...
    'Display', 'none');
x0 = min(max(target, lb), ub);
[v, ~, exitflag] = quadprog(eye(n), -target, [], [], S, ...
    zeros(size(S, 1), 1), lb, ub, x0, opts);
assert(exitflag > 0, 'quadprog stopped (exit flag %d)', exitflag);
end
