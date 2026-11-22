%% Flux variability analysis (FVA) of the toy metabolic network
% FBA gives one optimal flux distribution, but often many exist. FVA finds,
% for every reaction, the smallest and largest flux compatible with
% (a chosen fraction of) the optimal growth rate.
% Requires: Optimization Toolbox (for linprog).

%% The network (same as toy_fba.m; book Fig. 8.1)
rxns = {'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'b1', 'b2', 'vbio'};
%    v1  v2  v3  v4  v5  v6  v7  b1  b2 vbio
S = [-1  -1   0   0   0   0   0   1   0   0     % A
      1   0   1  -1  -1   0   0   0   0   0     % B
      0   1  -1   0   0   0   0   0   0   0     % C
      0   0   0   2   0  -1  -1   0   0   0     % D
      0   0   0   0   0   1   0   0  -1   0     % E
      0   0   0   0   1   0   0   0   0  -1     % F
      0   0   0   0   0   0   1   0   0  -2     % G
      1   0   1   0   0   0  -1   0   0 -10];   % ATP
lb = [0 0 0 -1000 0 -1000 0 0 -1000 0]';
ub = 1000 * ones(numel(rxns), 1);
n = numel(rxns);
c = zeros(n, 1);
c(strcmp(rxns, 'vbio')) = 1;
opts = optimoptions('linprog', 'Display', 'none');

%% Step 1: FBA, to find the optimal growth rate f*
[~, fval] = linprog(-c, [], [], S, zeros(size(S, 1), 1), lb, ub, opts);
f_star = -fval;

%% Step 2: FVA at 100% and at 90% of the optimum
[vmin100, vmax100] = fva(S, lb, ub, c, 1.0 * f_star, opts);
[vmin90,  vmax90]  = fva(S, lb, ub, c, 0.9 * f_star, opts);

fprintf('Optimal growth f* = %g\n', f_star);
fprintf('%5s %8s %8s   (at 100%% of f*)\n', 'rxn', 'min', 'max');
for j = 1:n
    fprintf('%5s %8.1f %8.1f\n', rxns{j}, vmin100(j), vmax100(j));
end

%% Plot: flux ranges at 100% and 90% of optimal growth
blue = [0.000 0.447 0.698];
orange = [0.902 0.624 0.000];
fig = figure;
hold on
y = 1:n;
for j = 1:n
    plot([vmin90(j) vmax90(j)], y(j) + [0.15 0.15], '-|', 'Color', orange, ...
        'LineWidth', 6, 'MarkerSize', 10);
    plot([vmin100(j) vmax100(j)], y(j) - [0.15 0.15], '-|', 'Color', blue, ...
        'LineWidth', 6, 'MarkerSize', 10);
end
set(gca, 'YTick', y, 'YTickLabel', rxns, 'YDir', 'reverse');
xlabel('Flux (mmol/gDW/h)');
title('Flux variability analysis');
h = [plot(NaN, NaN, '-', 'Color', orange, 'LineWidth', 6), ...
     plot(NaN, NaN, '-', 'Color', blue, 'LineWidth', 6)];
legend(h, {'90% of optimum', '100% of optimum'}, ...
    'Location', 'northeast', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-toy-fva');

results.growth = f_star;
results.fva100.min = cell2struct(num2cell(vmin100), rxns, 1);
results.fva100.max = cell2struct(num2cell(vmax100), rxns, 1);
results.fva90.min = cell2struct(num2cell(vmin90), rxns, 1);
results.fva90.max = cell2struct(num2cell(vmax90), rxns, 1);
sbp.save_results(mfilename('fullpath'), results);

%% Local function
function [vmin, vmax] = fva(S, lb, ub, c, f_min, opts)
% Min and max of every flux, subject to growth c'v >= f_min.
% The extra constraint is written as -c'v <= -f_min.
n = numel(lb);
vmin = zeros(n, 1);
vmax = zeros(n, 1);
beq = zeros(size(S, 1), 1);
for j = 1:n
    e = zeros(n, 1);
    e(j) = 1;
    [~, vmin(j)] = linprog(e, -c', -f_min, S, beq, lb, ub, opts);
    [~, fmax] = linprog(-e, -c', -f_min, S, beq, lb, ub, opts);
    vmax(j) = -fmax;
end
end
