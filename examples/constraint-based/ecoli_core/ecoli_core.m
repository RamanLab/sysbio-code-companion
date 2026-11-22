%% FBA on the E. coli core metabolic model with the COBRA Toolbox
% 1. Predict growth with and without oxygen.
% 2. Vary the glucose uptake rate and watch growth respond.
% Requires: COBRA Toolbox (run initCobraToolbox once per session first).

% The .mat file is the same model as the SBML file the Python version reads
% (made by tools/make_mat_model.py); reading SBML in MATLAB needs libSBML.
model = readCbModel(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
fprintf('%d reactions, %d metabolites, %d genes\n', ...
    numel(model.rxns), numel(model.mets), numel(model.genes));
fprintf('Objective: %s\n', strjoin(model.rxns(model.c ~= 0), ', '));

%% 1. Aerobic vs anaerobic growth
% Default medium: glucose uptake up to 10 mmol/gDW/h, oxygen unlimited.
% Uptake is a negative flux through an exchange reaction, hence lower bounds.
aerobic = optimizeCbModel(model, 'max');
fprintf('Aerobic growth rate:   %.4f /h\n', aerobic.f);

modelAnaerobic = changeRxnBounds(model, 'EX_o2_e', 0, 'l');  % no oxygen uptake
anaerobic = optimizeCbModel(modelAnaerobic, 'max');
fprintf('Anaerobic growth rate: %.4f /h\n', anaerobic.f);

%% 2. Growth as a function of glucose uptake
uptake = 0:0.5:20;  % mmol/gDW/h
growthAer = growth_curve(model, 1000, uptake);  % O2 effectively unlimited
growthO2_10 = growth_curve(model, 10, uptake);  % O2 uptake at most 10
growthAnaer = growth_curve(model, 0, uptake);   % no O2

%% Plot
blue = [0.000 0.447 0.698];
green = [0.000 0.620 0.451];
orange = [0.902 0.624 0.000];
fig = figure;
plot(uptake, growthAer, 'o-', 'Color', blue, ...
    'MarkerFaceColor', blue, 'MarkerSize', 4);
hold on
plot(uptake, growthO2_10, '^-', 'Color', green, ...
    'MarkerFaceColor', green, 'MarkerSize', 4);
plot(uptake, growthAnaer, 's-', 'Color', orange, ...
    'MarkerFaceColor', orange, 'MarkerSize', 4);
xlabel('Maximum glucose uptake (mmol/gDW/h)');
ylabel('Growth rate (1/h)');
title('E. coli core: growth vs glucose uptake');
legend({'O_2 unlimited', 'O_2 uptake \leq 10', 'no O_2'}, ...
    'Location', 'northwest', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-ecoli-glucose');

results.n_reactions = numel(model.rxns);
results.growth_aerobic = aerobic.f;
results.growth_anaerobic = anaerobic.f;
results.glucose_uptake = uptake;
results.growth_vs_glucose_aerobic = growthAer;
results.growth_vs_glucose_o2_10 = growthO2_10;
results.growth_vs_glucose_anaerobic = growthAnaer;
sbp.save_results(mfilename('fullpath'), results);

%% Local function
function growth = growth_curve(model, o2UptakeLimit, uptake)
% Maximum growth for each glucose uptake limit; NaN where the LP is infeasible.
model = changeRxnBounds(model, 'EX_o2_e', -o2UptakeLimit, 'l');
growth = nan(size(uptake));
for k = 1:numel(uptake)
    m = changeRxnBounds(model, 'EX_glc__D_e', -uptake(k), 'l');
    sol = optimizeCbModel(m, 'max');
    if sol.stat == 1  % 1 means an optimal solution was found
        growth(k) = sol.f;
    end
end
end
