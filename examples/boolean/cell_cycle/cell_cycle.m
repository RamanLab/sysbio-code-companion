%% A Boolean model of the mammalian cell cycle (Fauré et al., 2006)
% Ten genes and proteins; the update rules are the book's Eqs. 7.4. With
% 2^10 = 1024 possible states, we can examine every one of them:
%   1. synchronous updating, as in the previous example;
%   2. asynchronous updating, where one variable changes at a time, in
%      any order. The states then form a 'state transition graph', and the
%      attractors are the groups of states that, once entered, are never
%      left: its terminal strongly connected components.
% Requires: core MATLAB only.

names = {'CycD', 'Rb', 'E2F', 'CycE', 'CycA', 'p27', 'Cdc20', 'Cdh1', ...
         'Ubc', 'CycB'};
n = numel(names);
to_bits = @(s) double(bitget(s, n:-1:1));
to_number = @(x) sum(x .* 2.^(n-1:-1:0));

N = 2^n;
target = zeros(1, N);
for s = 0:N - 1
    target(s + 1) = to_number(double(update(to_bits(s))));
end

%% 1. Synchronous updating: follow every state to its attractor
keys = []; periods = []; reaches = zeros(1, N);
for s0 = 0:N - 1
    path = [];
    s = s0;
    while ~ismember(s, path)
        path(end + 1) = s; %#ok<AGROW>
        s = target(s + 1);
    end
    cycle = path(find(path == s):end);
    if ~ismember(min(cycle), keys)
        keys(end + 1) = min(cycle); %#ok<AGROW>
        periods(end + 1) = numel(cycle); %#ok<AGROW>
    end
    reaches(s0 + 1) = min(cycle);
end
[keys, order] = sort(keys);
periods = periods(order);
fprintf('Synchronous updating:\n');
for k = 1:numel(keys)
    b = to_bits(keys(k));
    sync_summary(k) = struct('period', periods(k), ...
        'basin', sum(reaches == keys(k)), 'CycD', b(1)); %#ok<SAGROW>
    fprintf('  attractor of period %d (CycD = %d), basin of %d states\n', ...
        periods(k), b(1), sync_summary(k).basin);
end

%% 2. Asynchronous updating: from each state, any one variable whose
% rule says it should change may change next
src = []; dst = [];
for s = 0:N - 1
    x = to_bits(s);
    fx = to_bits(target(s + 1));
    for i = find(x ~= fx)
        y = x;
        y(i) = fx(i);
        src(end + 1) = s + 1; %#ok<AGROW>
        dst(end + 1) = to_number(y) + 1; %#ok<AGROW>
    end
end
G = digraph(src, dst, [], N);
comp = conncomp(G, 'Type', 'strong');  % component of each state
C = condensation(G);                   % one node per component
terminal = find(outdegree(C) == 0)';
sizes = zeros(size(terminal)); cycd = zeros(size(terminal));
for k = 1:numel(terminal)
    members = find(comp == terminal(k)) - 1;  % back to states 0..N-1
    sizes(k) = numel(members);
    b = to_bits(min(members));
    cycd(k) = b(1);
end
[~, order] = sortrows([cycd' sizes']);
fprintf('Asynchronous updating:\n');
for k = 1:numel(order)
    async_summary(k) = struct('size', sizes(order(k)), ...
        'CycD', cycd(order(k))); %#ok<SAGROW>
    fprintf('  attractor of %d states (CycD = %d)\n', sizes(order(k)), ...
        cycd(order(k)));
end

%% A synchronous trajectory: start from a G1-like state, with CycD on
x0 = double(ismember(names, {'CycD', 'Rb', 'p27', 'Cdh1'}));
traj = zeros(21, n);
traj(1, :) = x0;
for t = 2:21
    traj(t, :) = double(update(traj(t - 1, :)));
end

fig = figure('Position', [100 100 800 360]);
imagesc(0:20, 1:n, traj', [0 1]);
colormap([0.97 0.98 1; 0.12 0.47 0.71]);   % OFF: light, ON: dark
set(gca, 'YTick', 1:n, 'YTickLabel', names, 'XTick', 0:2:20, 'Box', 'off');
xlabel('Time step');
title('Synchronous trajectory from a G1-like state (dark: ON, light: OFF)');
sbp.save_figure(fig, mfilename('fullpath'), 'fig-cell-cycle');

results.synchronous = sync_summary;
results.asynchronous = async_summary;
results.trajectory = traj;
sbp.save_results(mfilename('fullpath'), results);

%% Local function
function y = update(x)
% Book Eqs. 7.4 (Fauré et al., 2006)
x = num2cell(logical(x));
[CycD, Rb, E2F, CycE, CycA, p27, Cdc20, Cdh1, Ubc, CycB] = x{:};
y = [CycD, ...
     (~CycA && ~CycB && ~CycD && ~CycE) || (p27 && ~CycB && ~CycD), ...
     (~Rb && ~CycA && ~CycB) || (p27 && ~Rb && ~CycB), ...
     E2F && ~Rb, ...
     (E2F && ~Rb && ~Cdc20 && ~(Cdh1 && Ubc)) || ...
         (CycA && ~Rb && ~Cdc20 && ~(Cdh1 && Ubc)), ...
     (~CycD && ~CycE && ~CycA && ~CycB) || ...
         (p27 && ~(CycE && CycA) && ~CycB && ~CycD), ...
     CycB, ...
     (~CycA && ~CycB) || Cdc20 || (p27 && ~CycB), ...
     ~Cdh1 || (Cdh1 && Ubc && (Cdc20 || CycA || CycB)), ...
     ~Cdc20 && ~Cdh1];
end
