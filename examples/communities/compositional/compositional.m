%% Association networks from relative abundances: beware of compositionality
% Microbial association networks (book §11.1.1.1) are built from the
% co-occurrence of taxa across many samples. Sequencing, however, gives
% relative abundances: the reads of each taxon as a fraction of the total.
% Fractions must add up to one, so if one taxon rises, the others must
% fall, even if they are entirely unaffected. Here, we simulate 10 taxa
% whose absolute abundances vary *independently* across 200 samples
% (log-normally, with one dominant taxon), and compare the correlations
% between taxa computed from
%   * the absolute abundances (no true associations at all),
%   * the relative abundances, and
%   * the centred log-ratio (CLR) transform of the relative abundances,
%     log(x_i) - mean_j log(x_j), used by SparCC-like methods.
% An "edge" is a pair of taxa with |r| > 0.2 (p < 0.005 for 200 samples).
% The Gaussian numbers are made from sbp.uniform (Box-Muller), so they are
% identical in MATLAB and Python.

taxa = 10; samples = 200;
mean_log = [4.0, ones(1, taxa - 1)];             % taxon 1 dominates
sd_log = ones(1, taxa);
u = sbp.uniform(77, 2 * samples * taxa);
z = sqrt(-2 * log(u(1:2:end))) .* cos(2 * pi * u(2:2:end));
z = reshape(z, taxa, samples)';                   % samples x taxa
absolute = exp(mean_log + sd_log .* z);
relative = absolute ./ sum(absolute, 2);
log_rel = log(relative);
clr = log_rel - mean(log_rel, 2);

upper = triu(true(taxa), 1);
names = {'absolute', 'relative', 'clr'};
titles = {'Absolute abundances', 'Relative abundances', 'CLR-transformed'};
data = {absolute, relative, clr};
fig = figure('Position', [100 100 1200 400]);
t = linspace(0, 1, 128)';
cmap = [[t; ones(128, 1)], [t; flipud(t)], [ones(128, 1); flipud(t)]];
for d = 1:3
    R = corrcoef(data{d});
    r = R(upper);
    results.(names{d}) = struct('mean_r', mean(r), 'edges', sum(abs(r) > 0.2), ...
        'dominant_mean_r', mean(R(1, 2:end)));
    fprintf(['%-9s mean r = %+.3f; edges (|r| > 0.2): %2d of %d; mean r ' ...
        'with the dominant taxon = %+.3f\n'], names{d}, mean(r), ...
        sum(abs(r) > 0.2), sum(upper, 'all'), mean(R(1, 2:end)));
    subplot(1, 3, d)
    R(logical(eye(taxa))) = NaN;
    imagesc(R, 'AlphaData', ~isnan(R), [-0.6, 0.6]); colormap(cmap); axis square
    title({titles{d}, sprintf('%d edges', results.(names{d}).edges)});
end
cb = colorbar; cb.Label.String = 'Pearson r';
sbp.save_figure(fig, mfilename('fullpath'), 'fig-compositional');

sbp.save_results(mfilename('fullpath'), results);
