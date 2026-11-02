%% Noise in gene expression: transcription and translation
%     0 -> M       at rate km           (transcription)
%     M -> 0       at rate gm * M       (mRNA degradation)
%     M -> M + P   at rate kp * M       (translation)
%     P -> 0       at rate gp * P       (protein degradation)
% Two genes make the same mean number of proteins, 50. Gene A is
% transcribed often and each mRNA is translated a few times; gene B is
% transcribed rarely and each mRNA is translated many times, so its
% proteins are made in bursts. Theory (for example, Thattai and van
% Oudenaarden, 2001) gives, at steady state,
%     mean P = km*kp/(gm*gp),   Fano factor = var/mean = 1 + kp/(gm + gp).
% Requires: core MATLAB only.

gm = 1.0; gp = 0.1;                     % per minute
names = {'A', 'B'};
km = [2.0 0.2];                         % B: rare transcription,
kp = [2.5 25.0];                        %    big bursts
t_grid = 0:40000;                       % one long run, sampled each minute
burn_in = 200;                          % discard the approach to steady state
rng(1);

% Changes in (M, P) for each of the four reactions
changes = [1 0; -1 0; 0 1; 0 -1];

traces = zeros(numel(t_grid), 2);
for g = 1:2
    propensities = @(x) [km(g), gm * x(1), kp(g) * x(1), gp * x(2)];
    X = sbp.gillespie([0 0], changes, propensities, t_grid);
    traces(:, g) = X(:, 2);
    P = X(burn_in + 1:end, 2);
    results.(names{g}) = struct('mean', mean(P), ...
        'fano', var(P, 1) / mean(P));
    fprintf(['Gene %s: mean %.1f (theory %g), Fano factor %.2f ' ...
        '(theory %.2f)\n'], names{g}, mean(P), km(g) * kp(g) / (gm * gp), ...
        var(P, 1) / mean(P), 1 + kp(g) / (gm + gp));
end

%% Plot
colors = [0.122 0.467 0.706; 1.000 0.498 0.055];
fig = figure('Position', [100 100 1000 400]);
subplot(1, 3, [1 2])
window = 1001:1301;
for g = 1:2
    plot(t_grid(window) - t_grid(window(1)), traces(window, g), ...
        'Color', colors(g, :), 'LineWidth', 1.2);
    hold on
end
xlabel('Time (min)'); ylabel('Number of protein molecules');
title('Same mean, different noise');
legend({'gene A', 'gene B'}, 'Location', 'northwest', 'Box', 'off');
box off

subplot(1, 3, 3)
edges = 0:5:250;
for g = 1:2
    h = histcounts(traces(burn_in + 1:end, g), edges, ...
        'Normalization', 'pdf');
    stairs(edges, [h h(end)], 'Color', colors(g, :), 'LineWidth', 1.5);
    hold on
end
xlabel('Number of protein molecules'); ylabel('Probability density');
title('Distribution');
legend({'gene A', 'gene B'}, 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-gene-expression');

sbp.save_results(mfilename('fullpath'), results);
