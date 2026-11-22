%% Flux balance analysis of a toy metabolic network
% The network is the one in Fig. 8.1 / Table 8.2 of the book: 8 metabolites,
% 7 internal reactions, 2 exchange reactions and a 'biomass' reaction.
% FBA is solved directly as a linear program with linprog.
% Requires: Optimization Toolbox (for linprog).

%% The network
rxns = {'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'b1', 'b2', 'vbio'};
mets = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'ATP'};

% Stoichiometric matrix: one row per metabolite, one column per reaction
%    v1  v2  v3  v4  v5  v6  v7  b1  b2 vbio
S = [-1  -1   0   0   0   0   0   1   0   0     % A
      1   0   1  -1  -1   0   0   0   0   0     % B
      0   1  -1   0   0   0   0   0   0   0     % C
      0   0   0   2   0  -1  -1   0   0   0     % D
      0   0   0   0   0   1   0   0  -1   0     % E
      0   0   0   0   1   0   0   0   0  -1     % F
      0   0   0   0   0   0   1   0   0  -2     % G
      1   0   1   0   0   0  -1   0   0 -10];   % ATP

% Bounds (mmol/gDW/h): reversible reactions v4, v6 and b2 may run backwards
lb = [0 0 0 -1000 0 -1000 0 0 -1000 0]';
ub = 1000 * ones(numel(rxns), 1);

% Objective: maximise the biomass flux vbio
c = zeros(numel(rxns), 1);
c(strcmp(rxns, 'vbio')) = 1;

%% Solve: max c'v  s.t.  Sv = 0,  lb <= v <= ub
% linprog minimises, so we minimise -c'v
opts = optimoptions('linprog', 'Display', 'none');
beq = zeros(numel(mets), 1);
[v, fval, exitflag] = linprog(-c, [], [], S, beq, lb, ub, opts);
growth = -fval;

fprintf('Exit flag: %d (1 means optimal)\n', exitflag);
fprintf('Maximum biomass flux vbio = %g\n', growth);
for k = 1:numel(rxns)
    fprintf('  %4s = %8.1f\n', rxns{k}, v(k));
end
fprintf('Largest mass-balance violation |Sv| = %.1e\n', max(abs(S*v)));

%% Plot the flux distribution
fig = figure;
bar(categorical(rxns, rxns), v, 'FaceColor', [0.000 0.447 0.698]);
xlabel('Reaction');
ylabel('Flux (mmol/gDW/h)');
title(sprintf('An optimal flux distribution (vbio = %g)', growth));
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-toy-fba');

results.growth = growth;
results.fluxes = cell2struct(num2cell(v), rxns, 1);
results.max_mass_balance_error = max(abs(S*v));
sbp.save_results(mfilename('fullpath'), results);
