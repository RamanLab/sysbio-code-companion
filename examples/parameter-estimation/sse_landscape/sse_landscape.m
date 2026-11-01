%% Grid search, and the shape of the error landscape
% The same competitive-inhibition model and data as in fit_inhibition.
% We evaluate the weighted SSE on a grid of whole-number values of KM and
% KI (1 to 60 mM). The rate is proportional to vmax, so for each (KM, KI)
% the best vmax can be computed exactly, as in linear regression, and the
% grid only needs two dimensions.
% Requires: core MATLAB only.

data = readmatrix(fullfile(sbp.data_dir(), 'competitive_inhibition.csv'));
S = data(:, 1); v10 = data(:, 2); v20 = data(:, 3);
delta = 0.1;
names = {'D1', 'D1_D2'};
datasets = {{S, 10 * ones(20, 1), v10}, ...
            {[S; S], [10 * ones(20, 1); 20 * ones(20, 1)], [v10; v20]}};
KM_grid = 1:60;
KI_grid = 1:60;

for d = 1:2
    [s, i, v] = datasets{d}{:};
    grid = zeros(numel(KI_grid), numel(KM_grid));  % rows: KI, columns: KM
    vmax = zeros(size(grid));
    for r = 1:numel(KI_grid)
        for c = 1:numel(KM_grid)
            [vmax(r, c), grid(r, c)] = best_vmax_and_sse( ...
                KM_grid(c), KI_grid(r), s, i, v, delta);
        end
    end
    [best, idx] = min(grid(:));
    [r, c] = ind2sub(size(grid), idx);
    % How many grid points fit almost as well as the best (within 1 unit)?
    n_close = sum(grid(:) < best + 1);
    fprintf(['%s: best on grid KM = %d, KI = %d, vmax = %.4f, ' ...
        'SSE = %.4f; %d grid points within 1 of the best SSE\n'], ...
        names{d}, KM_grid(c), KI_grid(r), vmax(r, c), best, n_close);
    SSE.(names{d}) = grid;
    results.(names{d}) = struct('KM', KM_grid(c), 'KI', KI_grid(r), ...
        'vmax', vmax(r, c), 'SSE', best, 'n_within_1', n_close, ...
        'SSE_at_KI_20', grid(KI_grid == 20, :));
end

%% Plot: log10(SSE - best SSE + 1), so that the best point is at 0
fig = figure('Position', [100 100 1000 420]);
titles = {'D1 only (I = 10 mM)', 'D1 and D2 together'};
for d = 1:2
    subplot(1, 2, d)
    Z = log10(SSE.(names{d}) - min(SSE.(names{d})(:)) + 1);
    contourf(KM_grid, KI_grid, min(Z, 3), linspace(0, 3, 13), ...
        'LineStyle', 'none');
    hold on
    plot(results.(names{d}).KM, results.(names{d}).KI, 'p', ...
        'Color', [0.839 0.153 0.157], ...
        'MarkerFaceColor', [0.839 0.153 0.157], 'MarkerSize', 14);
    colormap(flipud(parula));
    clim([0 3]);
    xlabel('K_M (mM)');
    if d == 1, ylabel('K_I (mM)'); end
    title(titles{d});
end
cb = colorbar;
cb.Label.String = 'log_{10}(SSE - best SSE + 1)';
sbp.save_figure(fig, mfilename('fullpath'), 'fig-sse-landscape');

sbp.save_results(mfilename('fullpath'), results);

%% Local function
function [vmax, sse] = best_vmax_and_sse(KM, KI, S, I, v, delta)
% Weighted residuals are vmax*a - b, so the best vmax is sum(a.*b)/sum(a.^2).
a = S ./ (KM * (1 + I / KI) + S) ./ (delta * v);
b = 1 / delta;
vmax = sum(a .* b) / sum(a .* a);
sse = sum((vmax * a - b) .^ 2);
end
