%% Epidemics on networks, and vaccinating the hubs
% An SIR epidemic on a contact network: each infected person infects each
% susceptible contact with probability T (the transmissibility) and then
% recovers. We compare Erdős–Rényi and preferential-attachment (scale-free)
% networks with the same mean degree, 4, and then compare two ways of
% vaccinating 10% of the scale-free network: at random, or the people with
% the most contacts.
% Requires: core MATLAB only.

n = 2000; m = 2;
n_networks = 10; n_seeds = 50;
T_values = round(0:0.05:0.6, 10);
rng(1);

size_er = zeros(size(T_values)); size_ba = zeros(size(T_values));
Tc_er = zeros(1, n_networks); Tc_ba = zeros(1, n_networks);
for r = 1:n_networks
    ba = neighbour_lists(barabasi_albert(n, m), n);
    er = neighbour_lists(erdos_renyi(n, 2 * m), n);
    Tc_ba(r) = threshold(ba);
    Tc_er(r) = threshold(er);
    for t = 1:numel(T_values)
        for seed = randi(n, 1, n_seeds)
            size_ba(t) = size_ba(t) + outbreak(ba, T_values(t), seed, []) / n;
        end
        for seed = randi(n, 1, n_seeds)
            size_er(t) = size_er(t) + outbreak(er, T_values(t), seed, []) / n;
        end
    end
end
size_er = size_er / (n_networks * n_seeds);
size_ba = size_ba / (n_networks * n_seeds);
Tc = struct('er', mean(Tc_er), 'ba', mean(Tc_ba));
fprintf(['Epidemic threshold <k>/(<k^2> - <k>): Erdős–Rényi %.4f, ' ...
    'scale-free %.4f\n'], Tc.er, Tc.ba);
for t = [4 7 11]
    fprintf(['T = %.2f: mean fraction infected, Erdős–Rényi %.4f, ' ...
        'scale-free %.4f\n'], T_values(t), size_er(t), size_ba(t));
end

%% Vaccination of 10% of a scale-free network, with T = 0.5
T_vacc = 0.5; n_vacc = n / 10;
strategies = {'none', 'random', 'hubs'};
vacc = struct('none', [], 'random', [], 'hubs', []);
for r = 1:n_networks
    nbrs = neighbour_lists(barabasi_albert(n, m), n);
    degree = cellfun(@numel, nbrs);
    [~, by_degree] = sort(degree, 'descend');
    choices = {[], randperm(n, n_vacc), by_degree(1:n_vacc)};
    for s = 1:3
        others = setdiff(1:n, choices{s});
        for seed = others(randi(numel(others), 1, n_seeds))
            vacc.(strategies{s})(end + 1) = ...
                outbreak(nbrs, T_vacc, seed, choices{s}) / n;
        end
    end
end
for s = 1:3
    vacc.(strategies{s}) = mean(vacc.(strategies{s}));
end
fprintf(['Vaccinating 10%% at T = %g: mean fraction infected %.4f ' ...
    '(no vaccination), %.4f (random), %.4f (hubs)\n'], T_vacc, ...
    vacc.none, vacc.random, vacc.hubs);

%% Plot
blue = [0.122 0.467 0.706]; orange = [1.000 0.498 0.055];
fig = figure('Position', [100 100 1000 400]);
subplot(1, 5, 1:3)
plot(T_values, size_er, 's-', 'Color', orange, 'MarkerFaceColor', orange);
hold on
plot(T_values, size_ba, 'o-', 'Color', blue, 'MarkerFaceColor', blue);
xline(Tc.er, ':', 'Color', orange); xline(Tc.ba, ':', 'Color', blue);
xlabel('Transmissibility T'); ylabel('Mean fraction infected');
title('Epidemic size (dotted: thresholds)');
legend({'Erdős–Rényi', 'scale-free'}, 'Location', 'northwest', ...
    'Box', 'off');
box off
subplot(1, 5, 4:5)
b = bar(categorical({'none', 'random 10%', 'top 10% by degree'}, ...
    {'none', 'random 10%', 'top 10% by degree'}), ...
    [vacc.none vacc.random vacc.hubs], 'FaceColor', 'flat');
b.CData = [0.6 0.6 0.6; 0.62 0.79 0.88; blue];
ylabel('Mean fraction infected'); xlabel('Who is vaccinated');
title(sprintf('Vaccination, scale-free, T = %g', T_vacc));
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-epidemic-networks');

high = T_values >= 0.4;     % well above both thresholds
results.threshold = Tc;
results.T = T_values;
results.size_er = size_er;
results.size_ba = size_ba;
results.size_er_high_T = size_er(high);
results.size_ba_high_T = size_ba(high);
results.vaccination = vacc;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function edges = barabasi_albert(n, m)
% Preferential attachment, as in the networks chapter (nodes from 1).
n_edges = m * (m + 1) / 2 + m * (n - m - 1);
edges = zeros(n_edges, 2);
e = 0;
for i = 1:m + 1                            % start from a clique
    for j = 1:i - 1
        e = e + 1;
        edges(e, :) = [i j];
    end
end
ends = [reshape(edges(1:e, :)', 1, []), zeros(1, 2 * (n_edges - e))];
n_ends = 2 * e;
for new = m + 2:n
    targets = [];
    while numel(targets) < m
        t = ends(randi(n_ends));
        if ~ismember(t, targets)
            targets(end + 1) = t; %#ok<AGROW>
        end
    end
    for t = targets
        e = e + 1;
        edges(e, :) = [new t];
        ends(n_ends + 1:n_ends + 2) = [new t];
        n_ends = n_ends + 2;
    end
end
end

function edges = erdos_renyi(n, mean_k)
[i, j] = find(triu(rand(n) < mean_k / (n - 1), 1));
edges = [i j];
end

function nbrs = neighbour_lists(edges, n)
A = sparse(edges(:, 1), edges(:, 2), 1, n, n);
A = A + A';
nbrs = cell(1, n);
for i = 1:n
    nbrs{i} = find(A(:, i))';
end
end

function total = outbreak(nbrs, T, seed, immune)
% Final number of people infected, starting from one person. Each newly
% infected person gets one chance to infect each contact.
state = false(1, numel(nbrs));            % ever infected (or immune)
state(immune) = true;
state(seed) = true;
current = seed; total = 1;
while ~isempty(current)
    new = [];
    for i = current
        contacts = nbrs{i}(rand(1, numel(nbrs{i})) < T);
        for j = contacts
            if ~state(j)
                state(j) = true;
                new(end + 1) = j; %#ok<AGROW>
            end
        end
    end
    total = total + numel(new);
    current = new;
end
end

function Tc = threshold(nbrs)
% Epidemic threshold T_c = <k> / (<k^2> - <k>).
k = cellfun(@numel, nbrs);
Tc = mean(k) / (mean(k.^2) - mean(k));
end
