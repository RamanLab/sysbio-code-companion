%% Two E. coli mutants that can only grow together
% Strain A cannot make glutamate (GLUDy and GLUSy deleted); strain B
% cannot replenish its TCA cycle (PPC and ICL deleted). Neither grows
% alone on glucose. In a community model, each strain keeps its own
% network, and both exchange metabolites through a shared medium. With
% the fraction x of strain A fixed, both strains growing at the same rate
% mu (as they must in a stable community), and fluxes per gram of the
% whole community, the problem is a linear programme:
%     maximise mu
%     subject to  S u_A = 0,  S u_B = 0,
%                 u_A(biomass) = x mu,  u_B(biomass) = (1 - x) mu,
%                 x lb_A <= u_A <= x ub_A, (1 - x) lb_B <= u_B <= (1 - x) ub_B,
%                 what the community takes from the medium is available.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = model.S; lb0 = model.lb; ub0 = model.ub; ids = model.rxns;
n = numel(ids); m = size(S, 1);
bio = find(strcmp(ids, 'Biomass_Ecoli_core'));
ex = find(startsWith(ids, 'EX_'))';

% The medium supplies glucose, oxygen, ammonium, phosphate, water and
% protons, and takes up CO2; other metabolites come only from the cells
medium = {'EX_glc__D_e', 'EX_o2_e', 'EX_nh4_e', 'EX_pi_e', 'EX_h2o_e', ...
    'EX_h_e', 'EX_co2_e'};
in_medium = ismember(ids, medium);

A = strain({'GLUDy', 'GLUSy'}, ids, ex, medium, lb0, ub0);
B = strain({'PPC', 'ICL'}, ids, ex, medium, lb0, ub0);
net = struct('S', S, 'bio', bio, 'ex', ex, 'in_medium', in_medium);

fprintf('Alone: strain A grows at %.4f /h, strain B at %.4f /h\n', ...
    alone(net, A.lb, A.ub), alone(net, B.lb, B.ub));

fractions = round(0.01:0.01:0.99, 10);
growth = zeros(size(fractions));
for k = 1:numel(fractions)
    [~, f] = community(net, A, B, fractions(k), false, []);
    growth(k) = max(0, -f);
end
best = max(growth);
plateau = fractions(growth > best - 1e-6);
fprintf(['Community growth up to %.4f /h, for fractions of A from ' ...
    '%.2f to %.2f\n'], best, min(plateau), max(plateau));

% What must be exchanged? At x = 0.5, hold growth at its maximum and
% find the smallest total exchange of metabolites between the strains
x = 0.5;
[~, f] = community(net, A, B, x, false, []);
mu = -f;
[z, least] = community(net, A, B, x, true, [mu * (1 - 1e-7), mu]);
uA = z(1:n); uB = z(n + 1:2 * n);
fprintf(['At x = 0.5 (growth %.4f /h), the least exchange needed, in ' ...
    'mmol per gram of community per hour (+ secreted, - taken up):\n'], mu);
traded = ex(~in_medium(ex) & max(abs(uA(ex)), abs(uB(ex))) > 1e-4);
for i = traded
    fprintf('  %-12s A %+.3f   B %+.3f\n', ids{i}, uA(i), uB(i));
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 5, 1:3)
plot(fractions, growth, 'Color', [0.122 0.467 0.706]);
yline(alone(net, lb0, ub0), ':', 'wild type alone', ...
    'Color', [0.6 0.6 0.6], 'LabelHorizontalAlignment', 'center');
ylim([0 1]);
xlabel('Fraction of strain A in the community');
ylabel('Community growth rate (1/h)');
title('Growth depends on composition');
box off
subplot(1, 5, 4:5)
label = containers.Map({'EX_akg_e', 'EX_glu__L_e', 'EX_lac__D_e', ...
    'EX_succ_e', 'EX_mal__L_e', 'EX_fum_e'}, {'2-oxoglutarate', ...
    'glutamate', 'D-lactate', 'succinate', 'malate', 'fumarate'});
names = ids(traded);
for k = 1:numel(names)
    if isKey(label, names{k}), names{k} = label(names{k}); end
end
h = barh(1:numel(traded), [uA(traded) uB(traded)]);
h(1).FaceColor = [1.000 0.498 0.055]; h(2).FaceColor = [0.173 0.627 0.173];
yticks(1:numel(traded)); yticklabels(names);
xline(0, 'k');
xlabel('Exchange flux (+ secreted, - taken up)');
title('Least cross-feeding, x = 0.5');
legend(h, {'strain A', 'strain B'}, 'Location', 'northoutside', ...
    'Orientation', 'horizontal', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-cross-feeding');

results.growth_alone = [alone(net, A.lb, A.ub) alone(net, B.lb, B.ub)];
results.fractions = fractions;
results.growth = growth;
results.best_growth = best;
results.plateau = [min(plateau) max(plateau)];
results.least_exchange_total = least;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function b = strain(deleted, ids, ex, medium, lb, ub)
% Bounds per gram of a strain. Each cell can take up or secrete at most
% 10 mmol/gDW/h of each metabolite other than the inorganic ones, and
% glucose uptake is limited to 10 as before.
for i = ex
    if ismember(ids{i}, medium(2:end))
        lb(i) = -1000;
    else
        lb(i) = -10; ub(i) = 10;
    end
end
del = ismember(ids, deleted);
lb(del) = 0; ub(del) = 0;
b = struct('lb', lb, 'ub', ub);
end

function [z, f] = community(net, A, B, x, minimise_exchange, mu_fixed)
% Variables: u_A (n), u_B (n), mu, and, to minimise cross-feeding,
% t_A and t_B >= |exchange fluxes| (one per exchange reaction).
[m, n] = size(net.S);
ex = net.ex;
k = numel(ex) * minimise_exchange;
N = 2 * n + 1 + 2 * k;
Aeq = zeros(2 * m + 2, N);
Aeq(1:2 * m, 1:2 * n) = blkdiag(net.S, net.S);
Aeq(2 * m + 1, [net.bio, 2 * n + 1]) = [1, -x];
Aeq(2 * m + 2, [n + net.bio, 2 * n + 1]) = [1, -(1 - x)];
beq = zeros(2 * m + 2, 1);
% Shared medium: net uptake by the community of a metabolite the medium
% does not supply must be zero or negative
shared = ex(~net.in_medium(ex));
Aub = zeros(numel(shared), N);
for j = 1:numel(shared)
    Aub(j, [shared(j), n + shared(j)]) = -1;
end
lb = [x * A.lb; (1 - x) * B.lb; 0; zeros(2 * k, 1)];
ub = [x * A.ub; (1 - x) * B.ub; Inf; Inf(2 * k, 1)];
c = zeros(N, 1);
if ~minimise_exchange
    c(2 * n + 1) = -1;                          % maximise mu
else
    lb(2 * n + 1) = mu_fixed(1); ub(2 * n + 1) = mu_fixed(2);
    c(2 * n + 2:end) = 1;                       % minimise sum t
    extra = zeros(0, N);
    offsets = [0 n];
    for s = 1:2
        for j = 1:numel(ex)
            if net.in_medium(ex(j)), continue; end
            t = 2 * n + 1 + (s - 1) * k + j;
            for sgn = [1 -1]                    % sgn * u <= t
                r = zeros(1, N);
                r([offsets(s) + ex(j), t]) = [sgn, -1];
                extra(end + 1, :) = r; %#ok<AGROW>
            end
        end
    end
    Aub = [Aub; extra];
end
opts = optimoptions('linprog', 'Display', 'none');
[z, f, exitflag] = linprog(c, Aub, zeros(size(Aub, 1), 1), Aeq, beq, ...
    lb, ub, opts);
if exitflag ~= 1
    z = []; f = 0;
end
end

function mu = alone(net, lb, ub)
% Growth of one strain on its own: it can take up only what the medium
% supplies.
for i = net.ex
    if ~net.in_medium(i), lb(i) = max(lb(i), 0); end
end
c = zeros(numel(lb), 1); c(net.bio) = -1;
opts = optimoptions('linprog', 'Display', 'none');
[~, f, exitflag] = linprog(c, [], [], net.S, zeros(size(net.S, 1), 1), ...
    lb, ub, opts);
mu = 0;
if exitflag == 1, mu = max(0, -f); end
end
