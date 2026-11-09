%% Sloppiness: the eigenvalues of the Hessian of the cost function
% Near its minimum, the weighted SSE behaves like a quadratic form whose
% Hessian is approximated by J'J, where J is the Jacobian of the weighted
% residuals (the Fisher Information Matrix, FIM). Following Gutenkunst et
% al., we differentiate with respect to log(theta), so that the
% eigenvalues measure relative changes in the parameters. A model is
% sloppy if lambda_min / lambda_max < 1e-3 (book §6.5.2).
%
% The model and data are those of fit_timecourse, at the parameters fitted
% there, and we compare two experiments: measuring both mRNA and protein,
% or the protein alone.

names = {'k_m', 'g_m', 'k_p', 'g_p'};
theta = [9.4984, 0.96342, 5.2262, 0.20479];      % fit_timecourse
data = readmatrix(fullfile(sbp.data_dir(), 'gene_expression_timecourse.csv'));
t = data(:, 1)'; m_obs = data(:, 2)'; p_obs = data(:, 3)';
delta = 0.1;

keys = {'mrna_and_protein', 'protein_only'};
labels = {'mRNA and protein', 'protein only'};
results = struct();
for c = 1:2
    use_mrna = c == 1;
    resid = @(log_p) residuals(exp(log_p), t, m_obs, p_obs, delta, use_mrna);
    log_p = log(theta);
    J = zeros(numel(resid(log_p)), numel(theta));
    for j = 1:numel(theta)                        % central differences
        up = log_p; up(j) = up(j) + 1e-5;
        down = log_p; down(j) = down(j) - 1e-5;
        J(:, j) = (resid(up) - resid(down)) / 2e-5;
    end
    [V, D] = eig(J' * J);
    [ev, order] = sort(diag(D), 'descend');
    sloppiest = V(:, order(end));
    first = sloppiest(find(abs(sloppiest) > 1e-3, 1));  % fix the sign: first
    sloppiest = sign(first) * sloppiest;                %   clear entry positive
    results.(keys{c}) = struct('eigenvalues', ev', 'ratio', ev(end) / ev(1), ...
        'sloppiest', sloppiest');
    fprintf('%s: eigenvalues %s;  lambda_min/lambda_max = %.2g\n', keys{c}, ...
        strjoin(compose('%.3g', ev'), ', '), ev(end) / ev(1));
    fprintf('   sloppiest direction (in log parameters): %s\n', ...
        strjoin(compose('%s %+.2f', string(names'), sloppiest), ', '));
end

%% Plot: the eigenvalue spectra
fig = figure('Position', [100 100 500 450]); hold on
for c = 1:2
    ev = max(results.(keys{c}).eigenvalues, 1e-12);   % for a log scale
    for e = ev / ev(1)
        plot([c - 1.3, c - 0.7], [e, e], 'Color', [0.122 0.467 0.706], ...
            'LineWidth', 1.5);
    end
end
set(gca, 'YScale', 'log', 'XTick', [0, 1], 'XTickLabel', labels);
ylim([1e-12, 3]); xlim([-0.6, 1.6]);
yline(1e-3, ':k');
ylabel('Eigenvalue / largest eigenvalue'); title('Eigenvalues of J''J'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-sloppiness');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function r = residuals(p, t, m_obs, p_obs, delta, use_mrna)
rhs = @(~, y) [p(1) - p(2) * y(1); p(3) * y(1) - p(4) * y(2)];
opts = odeset('RelTol', 1e-11, 'AbsTol', 1e-11);
[~, y] = ode45(rhs, [0, t], [0; 0], opts);
y = y(2:end, :)';
r = (y(2, :) - p_obs) ./ (delta * p_obs);
if use_mrna
    r = [(y(1, :) - m_obs) ./ (delta * m_obs), r];
end
r = r';
end
