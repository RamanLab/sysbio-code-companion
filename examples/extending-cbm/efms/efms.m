%% Elementary flux modes (EFMs) and extreme pathways of a small network
% The network is that of Figure 9.2 of the book. A and C can be exchanged
% with the surroundings (v1, v4r), as can F and G (v10, v11); v3r and v4r
% are reversible, and every other reaction is irreversible.
%
% An EFM is a steady-state flux distribution that uses a minimal set of
% reactions (book §9.8): no smaller set of reactions can carry a steady
% flux. For a network this small, we can apply the definition directly.
% For every set of reactions, we ask whether the stoichiometric matrix,
% restricted to those reactions, has exactly one steady state (up to
% scaling) that uses all of them, with every irreversible reaction running
% forwards. Real networks need cleverer algorithms, such as the double
% description method of efmtool, because the number of sets doubles with
% every reaction.
% Requires: Optimization Toolbox (for linprog).

reactions = {'v1', 'v2', 'v3r', 'v4r', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11'};
reversible = endsWith(reactions, 'r');
S = [
  % v1 v2 v3r v4r v5 v6 v7 v8 v9 v10 v11
     1 -1  0   0   0  0  0  0  0  0   0    % A
     0  1 -1   0  -1  0  0  0  0  0   0    % B
     0  0  1  -1   0  0  0  0 -1  0   0    % C
     0  0  0   0   1 -1  0 -1  0  0   0    % D
     0  0  0   0   0  1 -1  0  0  0   0    % E
     0  0  0   0   0  0  0  0  1 -1   0    % F
     0  0  0   0   0  0  1  2  1  0  -1];  % G
n = numel(reactions);

modes = zeros(0, n);
for mask = 1:2^n - 1
    support = logical(bitget(mask, 1:n));
    K = null(S(:, support));
    if size(K, 2) ~= 1, continue; end   % no steady state, or not a minimal one
    v = zeros(1, n);
    v(support) = K;
    if any(abs(v(support)) < 1e-9), continue; end   % a reaction is unused
    for sign = [1, -1]                  % a mode and its reverse
        w = sign * v;
        if all(w(~reversible) >= -1e-9)
            modes(end + 1, :) = w / min(abs(w(support))); %#ok<SAGROW>
        end
    end
end
modes = round(modes, 9) + 0;              % + 0 turns -0 into 0

% Systemic independence: is a mode a non-negative combination of the others?
extreme = false(size(modes, 1), 1);
opts = optimoptions('linprog', 'Display', 'none');
for i = 1:size(modes, 1)
    others = modes([1:i - 1, i + 1:end], :)';
    [~, ~, exitflag] = linprog(zeros(size(others, 2), 1), [], [], others, ...
        modes(i, :)', zeros(size(others, 2), 1), [], opts);
    extreme(i) = exitflag ~= 1;         % infeasible: not a combination
end

% Number the modes as the book does: extreme pathways first, the mode
% that only converts A to C first in each group, then by the last
% reaction used before v11 (v10, then v8, then v7)
keys = [~extreme, modes(:, 11) ~= 0, -abs(modes(:, 10:-1:5))];
[~, order] = sortrows(keys);
modes = modes(order, :); extreme = extreme(order);

% Yield of G on C, for the modes that take up C (v4r < 0)
fprintf('%d elementary flux modes, %d of them extreme pathways (marked *):\n', ...
    size(modes, 1), sum(extreme));
fprintf('      %s   G/C\n', sprintf('%5s', reactions{:}));
yields = nan(1, size(modes, 1));     % NaN: the mode takes up no C
marks = {' ', '*'};
for k = 1:size(modes, 1)
    e = modes(k, :);
    line = sprintf('%2d%s   %s', k, marks{extreme(k) + 1}, sprintf('%5.0f', e));
    if e(4) < 0
        yields(k) = e(11) / -e(4);
        line = [line, sprintf('  %4.1f', yields(k))]; %#ok<AGROW>
    end
    fprintf('%s\n', line);
end

%% Plot: the modes as a matrix
fig = figure('Position', [100 100 700 360]);
imagesc(modes, [-2 2]);
colormap(red_blue());
[rows, cols] = find(modes ~= 0);
for k = 1:numel(rows)
    text(cols(k), rows(k), sprintf('%.0f', modes(rows(k), cols(k))), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
labels = arrayfun(@(k) sprintf('EFM %d%s', k, marks{extreme(k) + 1}), ...
    1:size(modes, 1), 'UniformOutput', false);
set(gca, 'XTick', 1:n, 'XTickLabel', reactions, ...
    'YTick', 1:size(modes, 1), 'YTickLabel', strtrim(labels));
title('Elementary flux modes (* extreme pathways)');
sbp.save_figure(fig, mfilename('fullpath'), 'fig-efms');

results.reactions = reactions;
results.efms = modes;
results.extreme = extreme';
results.yield_G_on_C = yields;
sbp.save_results(mfilename('fullpath'), results);

function cmap = red_blue()
% Blue (negative) to white to red (positive), like Matplotlib's RdBu_r.
t = linspace(0, 1, 128)';
cmap = [[t; ones(128, 1)], [t; flipud(t)], [ones(128, 1); flipud(t)]];
end
