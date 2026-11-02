%% Synthetic lethal reaction pairs in the E. coli core model
% A pair of reactions is synthetic lethal if deleting both stops growth
% (below 1% of wild type) while deleting either one alone does not. We
% find all such pairs twice:
%   1. exhaustively, testing every pair of non-essential reactions;
%   2. as in Fast-SL (Pratapa et al., 2015), testing only pairs that can
%      possibly be lethal. If a growing flux distribution uses neither
%      reaction of a pair, deleting both leaves it intact, so the pair is
%      not lethal. Hence at least one reaction of a lethal pair carries flux
%      in the wild-type solution, and the other carries flux once the first
%      is deleted. Using sparse solutions (minimal total flux) keeps the
%      number of pairs to test small.
% Requires: COBRA Toolbox (run initCobraToolbox once per session first).

model = readCbModel(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
sol = optimizeCbModel(model, 'max');
cutoff = 0.01 * sol.f;
rxns = sort(model.rxns(~startsWith(model.rxns, {'EX_', 'Biomass'})));

is_essential = false(size(rxns));
for k = 1:numel(rxns)
    is_essential(k) = growth_without(model, rxns(k)) < cutoff;
end
candidates = rxns(~is_essential);

%% 1. Exhaustive: every pair of non-essential reactions
pairs = nchoosek(1:numel(candidates), 2);
lethal_exhaustive = {};
for p = 1:size(pairs, 1)
    pair = candidates(pairs(p, :))';
    if growth_without(model, pair) < cutoff
        lethal_exhaustive(end + 1, :) = pair; %#ok<SAGROW>
    end
end

%% 2. Fast-SL: only pairs (i, j) with i active in the wild type and j
% active once i is deleted
tested = {};
lethal_fastsl = {};
for i = active_without(model, {}, candidates)'
    for j = active_without(model, i, candidates)'
        pair = sort([i, j]);
        key = strjoin(pair, ' ');
        if ~strcmp(i{1}, j{1}) && ~ismember(key, tested)
            tested{end + 1} = key; %#ok<SAGROW>
            if growth_without(model, pair) < cutoff
                lethal_fastsl(end + 1, :) = pair; %#ok<SAGROW>
            end
        end
    end
end
lethal_fastsl = sortrows(lethal_fastsl);
same = isequal(lethal_fastsl, sortrows(lethal_exhaustive));

fprintf('%d non-essential reactions\n', numel(candidates));
fprintf('Exhaustive: %d pairs tested, %d synthetic lethal\n', ...
    size(pairs, 1), size(lethal_exhaustive, 1));
fprintf('Fast-SL:    %d pairs tested, %d synthetic lethal\n', ...
    numel(tested), size(lethal_fastsl, 1));
fprintf('Same pairs found: %d\n', same);
partners = cellfun(@(r) sum(strcmp(lethal_fastsl(:), r)), candidates);
[~, order] = sortrows([-partners, (1:numel(candidates))']);
top = order(1:5);
fprintf('Reactions in the most pairs: %s\n', strjoin(strcat( ...
    candidates(top), ' (', arrayfun(@num2str, partners(top), ...
    'UniformOutput', false), ')'), ', '));

%% Plot: which reactions form lethal pairs
involved = unique(lethal_fastsl(:));
M = zeros(numel(involved));
for p = 1:size(lethal_fastsl, 1)
    a = strcmp(involved, lethal_fastsl{p, 1});
    b = strcmp(involved, lethal_fastsl{p, 2});
    M(a, b) = 1; M(b, a) = 1;
end
fig = figure('Position', [100 100 1100 500]);
subplot(1, 3, [1 2])
imagesc(M, [0 1]);
colormap([0.97 0.98 1; 0.12 0.47 0.71]);
axis square
set(gca, 'XTick', 1:numel(involved), 'XTickLabel', involved, ...
    'XTickLabelRotation', 90, 'YTick', 1:numel(involved), ...
    'YTickLabel', involved, 'FontSize', 7, 'Box', 'off');
title(sprintf('%d synthetic lethal pairs', size(lethal_fastsl, 1)));
subplot(1, 3, 3)
b = bar([size(pairs, 1), numel(tested)], 'FaceColor', 'flat');
b.CData = [0.6 0.6 0.6; 0.122 0.467 0.706];
set(gca, 'XTickLabel', {'exhaustive', 'Fast-SL'});
ylabel('Double deletions simulated');
title('Cost of the search');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-synthetic-lethals');

results.n_candidates = numel(candidates);
results.n_pairs_exhaustive = size(pairs, 1);
results.lethal_pairs = num2cell(lethal_fastsl, 2)';
results.same_as_exhaustive = same;
results.n_pairs_fastsl = numel(tested);
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function g = growth_without(model, deleted)
model = changeRxnBounds(model, deleted, 0, 'b');
sol = optimizeCbModel(model, 'max');
g = 0;
if sol.stat == 1
    g = sol.f;
end
end

function active = active_without(model, deleted, candidates)
% Candidate reactions carrying flux in a minimal-total-flux solution
if ~isempty(deleted)
    model = changeRxnBounds(model, deleted, 0, 'b');
end
sol = optimizeCbModel(model, 'max', 'one');
v = sol.x(findRxnIDs(model, candidates));
active = candidates(abs(v) > 1e-9);
end
