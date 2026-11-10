%% Network motifs in the E. coli transcriptional regulatory network
% A motif is a small pattern of interactions that occurs far more often in
% a real network than in randomised networks with the same degrees (book
% §3.4). We count two three-node patterns in the network of transcription
% factors and their targets (data/ecoli_trn.csv, from RegulonDB via
% PRECISE-1K):
%   feed-forward loop (FFL)   X -> Y, Y -> Z and X -> Z
%   feedback loop (FBL)       X -> Y, Y -> Z and Z -> X
% With A the adjacency matrix (no self-loops), the number of FFLs is
% sum(A .* (A A)), and the number of FBLs is trace(A^3) / 3. We compare
% the counts with 20 randomised networks that keep every node's in- and
% out-degree (book §3.4.1), and classify FFLs as coherent or incoherent
% from the signs of their edges. Random choices come from sbp.uniform, so
% that MATLAB and Python make the same swaps.

trn = readtable(fullfile(sbp.data_dir(), 'ecoli_trn.csv'), 'TextType', 'string');
names = unique([trn.regulator; trn.target]);      % sorted
n = numel(names);
[~, src] = ismember(trn.regulator, names);
[~, dst] = ismember(trn.target, names);
sgn = zeros(size(src));
sgn(trn.effect == "+") = 1;
sgn(trn.effect == "-") = -1;
auto = src == dst;
fprintf('%d nodes, %d edges; %d regulators regulate themselves\n', n, ...
    height(trn), sum(auto));
src = src(~auto); dst = dst(~auto); sgn = sgn(~auto);

[ffl, fbl] = motif_counts(src, dst, n);
random_counts = zeros(20, 2);
for s = 1:20
    [rs, rd] = randomise(src, dst, n, 100 + s, 10);
    [random_counts(s, 1), random_counts(s, 2)] = motif_counts(rs, rd, n);
end
mu = mean(random_counts, 1);
sd = std(random_counts, 0, 1);
z = ([ffl, fbl] - mu) ./ sd;

% Coherent FFLs: the sign of X -> Z equals the product of the signs of
% X -> Y and Y -> Z. For each FFL whose three signs are all known, the
% product of the three signs is +1 if it is coherent and -1 if not.
S = sparse(src, dst, sgn, n, n);                  % +1, -1, 0
K = abs(S);                                       % sign known
signed = full(sum((K * K) .* K, 'all'));          % FFLs with all signs known
product = full(sum((S * S) .* S, 'all'));         % sum of their sign products
coherent = round((signed + product) / 2);
incoherent = round((signed - product) / 2);

fprintf('Feed-forward loops: %d; randomised %.1f ± %.1f; Z = %.1f\n', ...
    ffl, mu(1), sd(1), z(1));
fprintf('Feedback loops:     %d; randomised %.1f ± %.1f; Z = %.1f\n', ...
    fbl, mu(2), sd(2), z(2));
fprintf('FFLs with all signs known: %d coherent, %d incoherent\n', ...
    coherent, incoherent);

%% Plot
fig = figure('Position', [100 100 900 400]);
labels = {'Feed-forward loops', 'Feedback loops'};
real_counts = [ffl, fbl];
for k = 1:2
    subplot(1, 2, k); hold on
    histogram(random_counts(:, k), 8, 'FaceColor', [0.122 0.467 0.706]);
    xline(real_counts(k), 'Color', [0.839 0.153 0.157], 'LineWidth', 2);
    title(sprintf('%s (Z = %.1f)', labels{k}, z(k))); xlabel('Count'); box off
    if k == 1
        ylabel('Randomised networks');
        legend({'randomised', 'E. coli'}, 'Box', 'off');
    end
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-motifs');

results.nodes = n;
results.edges = height(trn);
results.autoregulated = sum(auto);
results.ffl = ffl;
results.fbl = fbl;
results.random = random_counts;
results.z = z;
results.coherent = coherent;
results.incoherent = incoherent;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function [ffl, fbl] = motif_counts(src, dst, n)
% Numbers of FFLs and FBLs.
A = sparse(src, dst, 1, n, n);
AA = A * A;
ffl = full(sum(AA .* A, 'all'));
fbl = round(full(trace(AA * A)) / 3);
end

function [src, dst] = randomise(src, dst, n, seed, swaps_per_edge)
% Swap targets between pairs of edges, a->b and c->d becoming a->d and
% c->b, unless that creates a self-loop or an existing edge.
m = numel(src);
A = false(n);
A(sub2ind([n, n], src, dst)) = true;
u = sbp.uniform(seed, 2 * swaps_per_edge * m);
for t = 0:swaps_per_edge * m - 1
    e1 = floor(u(2 * t + 1) * m) + 1;
    e2 = floor(u(2 * t + 2) * m) + 1;
    a = src(e1); b = dst(e1); c = src(e2); d = dst(e2);
    if a == d || c == b || b == d || A(a, d) || A(c, b)
        continue
    end
    A(a, b) = false; A(c, d) = false;
    A(a, d) = true; A(c, b) = true;
    dst(e1) = d; dst(e2) = b;
end
end
