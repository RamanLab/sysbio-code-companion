%% A three-gene Boolean network (book Fig. 7.1, Eqs. 7.3)
%     g1* = g2 AND NOT g3
%     g2* = NOT g1
%     g3* = NOT g3 OR g2
% With synchronous updating, all genes are updated at once from the
% previous state. We list the successor of each of the 2^3 = 8 states,
% and find the attractors (the states the network ends up cycling
% through) and their basins (the states that lead to them).
% Requires: core MATLAB only.

n = 3;
update = @(x) [x(2) && ~x(3), ~x(1), ~x(3) || x(2)];
% State number <-> vector of 0/1, first gene as the leftmost digit
to_bits = @(s) bitget(s, n:-1:1);
to_number = @(x) sum(x .* 2.^(n-1:-1:0));
label = @(s) sprintf('%d', to_bits(s));

% Successor of every state (states are numbered 0 to 2^n - 1)
succ = zeros(1, 2^n);
for s = 0:2^n - 1
    succ(s + 1) = to_number(double(update(to_bits(s))));
    fprintf('%s => %s\n', label(s), label(succ(s + 1)));
end

% Attractors: follow each state until a state repeats; the repeating part
% is the attractor. Every state is assigned to the attractor it reaches.
attractors = {};
reaches = zeros(1, 2^n);
for s0 = 0:2^n - 1
    path = [];
    s = s0;
    while ~ismember(s, path)
        path(end + 1) = s; %#ok<AGROW>
        s = succ(s + 1);
    end
    cycle = path(find(path == s):end);
    key = min(cycle);
    if ~any(cellfun(@(a) min(a) == key, attractors))
        i = find(cycle == key);                     % start each cycle at
        attractors{end + 1} = cycle([i:end, 1:i - 1]); %#ok<SAGROW>
    end
    reaches(s0 + 1) = key;
end
basin_sizes = cellfun(@(a) sum(reaches == min(a)), attractors);
for k = 1:numel(attractors)
    a = attractors{k};
    fprintf('Attractor of period %d: %s; basin of %d states\n', numel(a), ...
        strjoin(arrayfun(label, a, 'UniformOutput', false), ' -> '), ...
        basin_sizes(k));
end

%% Plot the state transition map
labels = arrayfun(label, 0:2^n - 1, 'UniformOutput', false);
G = digraph(1:2^n, succ + 1, [], labels);   % MATLAB counts nodes from 1
% Same positions as the Python version, for states 000, 001, ..., 111
x = [3 1 2 3 0 1 0 2];
y = [1 1 1.5 0 1 0 0 0.5];
in_attractor = ismember(0:2^n - 1, [attractors{:}]);
fig = figure('Position', [100 100 700 350]);
h = plot(G, 'XData', x, 'YData', y, 'NodeLabel', labels, ...
    'MarkerSize', 12, 'NodeColor', 'k', 'EdgeColor', 'k', 'ArrowSize', 10);
highlight(h, find(in_attractor), 'NodeColor', [1.000 0.498 0.055]);
title('State transitions (orange: states in an attractor)');
axis off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-three-genes');

results.successor = arrayfun(label, succ, 'UniformOutput', false);
results.attractors = cellfun(@(a) arrayfun(label, a, ...
    'UniformOutput', false), attractors, 'UniformOutput', false);
results.basin_sizes = basin_sizes;
sbp.save_results(mfilename('fullpath'), results);
