%% OptKnock: knock-outs that make E. coli over-produce lactate
% This follows Exercise 9.6 of the book. The E. coli core model grows
% anaerobically on glucose (uptake up to 18 mmol/gDW/h), and we look for up
% to three reaction knock-outs that maximise lactate secretion, while
% keeping growth at no less than 10% of the wild type.
%
% OptKnock (book §9.3.1, Eq. 9.6) is a bi-level problem: the outer problem
% maximises lactate, over the choice of knock-outs, and the inner problem is
% the cell maximising its growth. Replacing the inner problem by its
% optimality conditions (primal feasibility, dual feasibility and strong
% duality) turns it into a single MILP. As the book notes, OptKnock is slow
% for large search spaces, so we let it choose from 14 reactions of central
% carbon metabolism and fermentation. The search space is small enough that
% we can also try every combination of knock-outs directly, and check the
% two answers against each other.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
ids = model.rxns;
model.lb(strcmp(ids, 'EX_glc__D_e')) = -18;   % glucose uptake
model.lb(strcmp(ids, 'EX_o2_e')) = 0;         % anaerobic
net.S = full(model.S); net.lb = model.lb; net.ub = model.ub; net.ids = ids;
net.bio = find(strcmp(ids, 'Biomass_Ecoli_core'));
net.lac = find(strcmp(ids, 'EX_lac__D_e'));

candidates = sort({'ACALD', 'ACKr', 'ALCD2x', 'FRD7', 'FUM', 'ME1', 'ME2', ...
    'PDH', 'PFL', 'PPC', 'PTAr', 'PYK', 'SUCCt3', 'THD2'});

mu_wt = optimise(net, net.bio, net.lb, net.ub, -1);
net.mu_wt = mu_wt;
[~, lac_lo, lac_hi] = evaluate(net, {});
fprintf('Wild type: growth %.4f /h, lactate %.2f to %.2f\n', mu_wt, ...
    lac_lo, lac_hi);

% Every design of up to three knock-outs from the search space
designs = struct('K', {}, 'ko', {}, 'mu', {}, 'low', {}, 'high', {});
for K = 1:3
    combos = nchoosek(1:numel(candidates), K);
    for c = 1:size(combos, 1)
        ko = candidates(combos(c, :));
        [mu, low, high] = evaluate(net, ko);
        if ~isempty(mu)
            designs(end + 1) = struct('K', K, 'ko', {ko}, 'mu', mu, ...
                'low', low, 'high', high); %#ok<SAGROW>
        end
    end
end

results.growth_wild_type = mu_wt;
for K = 1:3
    [value, design] = optknock(net, candidates, K);
    b = best(designs, K, 'high');
    assert(abs(value - b.high) < 1e-5);        % OptKnock finds the true optimum
    g = best(designs, K, 'low');
    fprintf('K = %d: OptKnock picks %s: up to %.2f lactate\n', K, ...
        strjoin(design, ', '), value);
    fprintf('       such a design grows at %.3f /h, and secretes %.2f to %.2f lactate\n', ...
        b.mu, b.low, b.high);
    fprintf('       most lactate guaranteed: %s: growth %.3f /h, lactate %.2f to %.2f\n', ...
        strjoin(g.ko, ', '), g.mu, g.low, g.high);
    results.(sprintf('K%d', K)) = struct('optknock_lactate', value, ...
        'growth', b.mu, 'lactate_min', b.low, 'lactate_max', b.high, ...
        'guaranteed_design', {g.ko}, 'guaranteed_growth', g.mu, ...
        'guaranteed_lactate', g.low);
end

%% Plot: production envelopes
b3 = best(designs, 3, 'high'); g3 = best(designs, 3, 'low');
g1 = best(designs, 1, 'low');
panels = {'Three knock-outs', ...
    {[strjoin(b3.ko, ', ') ' (OptKnock)'], b3.ko, [0.839 0.153 0.157]; ...
     strjoin(g3.ko, ', '), g3.ko, [0.122 0.467 0.706]}; ...
    'One knock-out', {strjoin(g1.ko, ', '), g1.ko, [0.173 0.627 0.173]}};
fig = figure('Position', [100 100 1100 430]);
for p = 1:2
    subplot(1, 2, p); hold on
    shown = [{'wild type', {}, [0.6 0.6 0.6]}; panels{p, 2}];
    h = gobjects(size(shown, 1), 1);
    for d = 1:size(shown, 1)
        [mus, low, high] = envelope(net, shown{d, 2});
        x = [mus, fliplr(mus), mus(1)];
        y = [high, fliplr(low), high(1)];
        fill(x, y, shown{d, 3}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        h(d) = plot(x, y, 'Color', shown{d, 3}, 'LineWidth', 1.5);
    end
    xline(0.1 * mu_wt, ':k');
    xlabel('Growth rate (1/h)'); title(panels{p, 1});
    if p == 1, ylabel('Lactate secretion (mmol/gDW/h)'); end
    legend(h, shown(:, 1), 'Box', 'off', 'Location', 'northeast');
    box off
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-optknock');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function value = optimise(net, j, lb, ub, sense)
% Maximise (sense = -1) or minimise (sense = 1) flux j; [] if infeasible.
c = zeros(numel(lb), 1); c(j) = sense;
opts = optimoptions('linprog', 'Display', 'none');
[~, f, exitflag] = linprog(c, [], [], net.S, zeros(size(net.S, 1), 1), ...
    lb, ub, opts);
if exitflag == 1, value = sense * f; else, value = []; end
end

function [lb, ub] = knock_out(net, reactions)
lb = net.lb; ub = net.ub;
j = ismember(net.ids, reactions);
lb(j) = 0; ub(j) = 0;
end

function [mu, low, high] = evaluate(net, reactions)
% Growth of a design, and the least and most lactate it can secrete while
% growing at that rate.
[lb, ub] = knock_out(net, reactions);
mu = optimise(net, net.bio, lb, ub, -1);
low = []; high = [];
if isempty(mu) || mu < 0.1 * net.mu_wt, mu = []; return; end
lb(net.bio) = mu * (1 - 1e-9);
low = optimise(net, net.lac, lb, ub, 1);
high = optimise(net, net.lac, lb, ub, -1);
end

function [mus, low, high] = envelope(net, reactions)
% Production envelope: least and most lactate at each growth rate.
[lb, ub] = knock_out(net, reactions);
mus = linspace(0, optimise(net, net.bio, lb, ub, -1), 21);
low = zeros(size(mus)); high = low;
for i = 1:numel(mus)
    lb(net.bio) = mus(i); ub(net.bio) = mus(i);
    low(i) = optimise(net, net.lac, lb, ub, 1);
    high(i) = optimise(net, net.lac, lb, ub, -1);
end
end

function d = best(designs, K, field)
% The design with the most lactate ('high': at best, 'low': guaranteed),
% with ties broken alphabetically.
pool = designs([designs.K] <= K);
values = [pool.(field)];
d = pool(find(values > max(values) - 1e-6, 1));
end

function [value, design] = optknock(net, candidates, K)
% OptKnock as one MILP. Variables, in order: fluxes v (n), dual variables
% of S v = 0 (m), of the upper and lower bounds (n each), slacks that free
% the dual constraint of a knocked-out reaction (nc), and y (nc), with
% y = 0 for a knock-out.
M = 1000;
S = net.S; lb = net.lb; ub = net.ub;
[m, n] = size(S);
cand = find(ismember(net.ids, candidates));    % same (sorted) order
[~, order] = sort(net.ids(cand)); cand = cand(order);
nc = numel(cand);
iu = n + m; il = 2 * n + m; isl = 3 * n + m; iy = 3 * n + m + nc;
N = iy + nc;
% Equalities: S v = 0; S'lam + mu_u - mu_l (+ slack) = c; strong duality
E = sparse(cand, 1:nc, 1, n, nc);
c = zeros(n, 1); c(net.bio) = 1;
duality = zeros(1, N);
duality(net.bio) = 1; duality(iu + (1:n)) = -ub'; duality(il + (1:n)) = lb';
Aeq = [sparse(S), sparse(m, N - n);
       sparse(n, n), sparse(S'), speye(n), -speye(n), E, sparse(n, nc);
       sparse(duality)];
beq = [zeros(m, 1); c; 0];
% Inequalities linking each candidate to its y
A = sparse(6 * nc + 1, N); b = zeros(6 * nc + 1, 1);
for k = 1:nc
    j = cand(k); r = 6 * (k - 1);
    A(r + 1, [j, iy + k]) = [1, -ub(j)];          % v_j <= ub_j y_k
    A(r + 2, [j, iy + k]) = [-1, lb(j)];          % v_j >= lb_j y_k
    A(r + 3, [iu + j, iy + k]) = [1, -M];         % bound duals vanish
    A(r + 4, [il + j, iy + k]) = [1, -M];         %   when knocked out
    A(r + 5, [isl + k, iy + k]) = [1, M];  b(r + 5) = M;   % |slack| is 0
    A(r + 6, [isl + k, iy + k]) = [-1, M]; b(r + 6) = M;   %   unless knocked out
end
A(end, iy + (1:nc)) = -1; b(end) = K - nc;        % at most K knock-outs
lower = [lb; -inf(m, 1); zeros(2 * n, 1); -inf(nc, 1); zeros(nc, 1)];
upper = [ub; inf(m + 2 * n + nc, 1); ones(nc, 1)];
lower(net.bio) = 0.1 * net.mu_wt;
f = zeros(N, 1); f(net.lac) = -1;                 % maximise lactate
opts = optimoptions('intlinprog', 'Display', 'none', ...
    'RelativeGapTolerance', 0, 'AbsoluteGapTolerance', 0, ...
    'IntegerTolerance', 1e-6, 'ConstraintTolerance', 1e-9);
[x, fval, exitflag] = intlinprog(f, iy + (1:nc), A, b, Aeq, beq, ...
    lower, upper, opts);
assert(exitflag == 1, 'intlinprog failed (exit flag %d)', exitflag);
value = -fval;
design = net.ids(cand(round(x(iy + (1:nc))) == 0))';
end
