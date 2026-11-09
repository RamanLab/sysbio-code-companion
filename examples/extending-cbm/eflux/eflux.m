%% E-Flux: constraining fluxes with gene expression data
% E-Flux (book §9.4.2) uses the expression of the genes behind a reaction
% as an approximate upper bound on its flux. The expression of a reaction
% follows its GPR rule: genes joined by 'and' (subunits of one complex)
% give the minimum of their expression, and genes joined by 'or'
% (isozymes) give the sum. The bounds of each reaction are then set to
% +/- this value, divided by the largest value over all reactions.
%
% The data are RNA-seq profiles of wild-type E. coli MG1655, grown
% aerobically in minimal medium on six carbon sources, from the PRECISE
% compendium (Sastry et al., 2019), with the growth rates measured in the
% same experiments. For each sample we build E-Flux bounds for the
% genome-scale model iML1515, open the uptake of that sample's carbon
% source, maximise growth, and compare the predictions with the measured
% growth rates. Expression is in log2(TPM + 1); we try both the raw TPM
% values, as in the original E-Flux, and the log values.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'iML1515.mat'));
model = s.iML1515;
log_tpm = readtable(fullfile(sbp.data_dir(), 'precise_log_tpm.csv'), ...
    'ReadRowNames', true, 'VariableNamingRule', 'preserve');
samples = readtable(fullfile(sbp.data_dir(), 'precise_samples.csv'), ...
    'TextType', 'string');
uptake = containers.Map({'glucose', 'xylose', 'glycerol', 'acetate', 'fructose'}, ...
    {'EX_glc__D_e', 'EX_xyl__D_e', 'EX_glyc_e', 'EX_ac_e', 'EX_fru_e'});

carbon = cellstr(extractBefore(samples.carbon_source, '('))';
measured = samples.growth_rate';
genes = log_tpm.Properties.RowNames;
transforms = {'TPM', 'log TPM'};
predicted = zeros(2, numel(carbon));
for t = 1:2
    for k = 1:numel(carbon)
        x = log_tpm.(samples.sample(k));
        if t == 1, x = 2 .^ x - 1; end
        expression = containers.Map(genes, num2cell(x));
        predicted(t, k) = eflux_growth(model, expression, carbon{k}, uptake);
    end
end
r = [corr1(measured, predicted(1, :)), corr1(measured, predicted(2, :))];

fprintf('carbon source    measured   E-Flux (TPM)   E-Flux (log TPM)\n');
for k = 1:numel(carbon)
    fprintf('%-12s  %8.2f   %12.3g   %14.3g\n', carbon{k}, measured(k), ...
        predicted(1, k), predicted(2, k));
end
for t = 1:2
    fprintf('Correlation with measured growth, %s: r = %.2f\n', transforms{t}, r(t));
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
for t = 1:2
    subplot(1, 2, t)
    plot(measured, predicted(t, :), 'o', 'Color', [0.122 0.467 0.706], ...
        'MarkerFaceColor', [0.122 0.467 0.706]);
    text(measured, predicted(t, :), strcat({'  '}, carbon), ...
        'VerticalAlignment', 'bottom', 'FontSize', 9);
    xlabel('Measured growth rate (1/h)');
    ylabel('E-Flux growth (arbitrary units)');
    title(sprintf('Expression as %s: r = %.2f', transforms{t}, r(t)));
    box off
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-eflux');

results.carbon_source = carbon;
results.measured = measured;
results.eflux_tpm = predicted(1, :);
results.eflux_log_tpm = predicted(2, :);
results.r_tpm = r(1);
results.r_log_tpm = r(2);
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function mu = eflux_growth(model, expression, carbon, uptake)
% Maximum growth with E-Flux bounds, on one carbon source.
lb = model.lb; ub = model.ub;
level = nan(numel(model.rxns), 1);
for j = 1:numel(model.rxns)
    if ~isempty(model.grRules{j})
        level(j) = gpr_value(model.grRules{j}, expression);
    end
end
largest = max(level(isfinite(level)));
for j = find(isfinite(level))'
    bound = level(j) / largest;
    if lb(j) < 0, lb(j) = -bound; else, lb(j) = 0; end
    if ub(j) > 0, ub(j) = bound; else, ub(j) = 0; end
end
lb(ismember(model.rxns, values(uptake))) = 0;   % only this carbon source
lb(strcmp(model.rxns, uptake(carbon))) = -1000;
lb(strcmp(model.rxns, 'ATPM')) = 0;              % see the text
opts = optimoptions('linprog', 'Display', 'none');
[~, f, exitflag] = linprog(-model.c, [], [], model.S, ...
    zeros(size(model.S, 1), 1), lb, ub, opts);
assert(exitflag == 1, 'linprog failed');
mu = -f;
end

function value = gpr_value(rule, expression)
% Evaluate a GPR rule such as '(b1 and b2) or b3': 'and' takes the minimum,
% 'or' the sum. A gene without data does not constrain the reaction (its
% value is infinite).
tokens = regexp(rule, '\(|\)|[^\s()]+', 'match');
value = either(tokens, 1, expression);
end

function [v, pos] = either(tokens, pos, expression)
% 'and' groups joined by 'or'
[v, pos] = both(tokens, pos, expression);
while pos <= numel(tokens) && strcmp(tokens{pos}, 'or')
    [w, pos] = both(tokens, pos + 1, expression);
    v = v + w;
end
end

function [v, pos] = both(tokens, pos, expression)
% terms joined by 'and'
[v, pos] = term(tokens, pos, expression);
while pos <= numel(tokens) && strcmp(tokens{pos}, 'and')
    [w, pos] = term(tokens, pos + 1, expression);
    v = min(v, w);
end
end

function [v, pos] = term(tokens, pos, expression)
% a gene, or a bracketed rule
token = tokens{pos};
pos = pos + 1;
if strcmp(token, '(')
    [v, pos] = either(tokens, pos, expression);
    pos = pos + 1;                               % skip ')'
elseif isKey(expression, token)
    v = expression(token);
else
    v = inf;
end
end

function r = corr1(x, y)
% Pearson correlation coefficient.
c = corrcoef(x, y);
r = c(1, 2);
end
