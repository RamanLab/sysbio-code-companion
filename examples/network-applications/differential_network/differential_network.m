%% Differential co-expression: how gene-gene correlations change between
% two conditions.
% The data are RNA-seq profiles from PRECISE-1K (Lamoureux et al., 2023) of
% wild-type E. coli grown on glucose (200 samples) or glycerol (108
% samples), for 44 genes of central carbon and glycerol metabolism. For
% each condition we compute the Pearson correlation of every pair of genes
% (a co-expression network), and test whether the correlation differs
% between the conditions, using Fisher's z-transformation (book §4.3):
%     z = (atanh r1 - atanh r2) / sqrt(1/(n1 - 3) + 1/(n2 - 3)).
% A pair is differentially co-expressed if |z| exceeds the Bonferroni
% threshold for all 946 pairs (overall significance 0.05). The genes with
% the most differential partners are the most "rewired".

expr = readtable(fullfile(sbp.data_dir(), 'precise1k_carbon_log_tpm.csv'), ...
    'ReadRowNames', true, 'VariableNamingRule', 'preserve');
samples = readtable(fullfile(sbp.data_dir(), 'precise1k_carbon_samples.csv'), ...
    'TextType', 'string');
genes = expr.Properties.RowNames';
g = numel(genes);

carbons = {'glucose', 'glycerol'};
for c = 1:2
    columns = cellstr(samples.sample(samples.carbon_source == carbons{c}));
    X = expr{:, columns};                         % genes x samples
    r.(carbons{c}) = corrcoef(X');                % correlations of genes
    n.(carbons{c}) = numel(columns);
end

% Fisher's z (the diagonal, r = 1, is not used, so we set it to 0)
f1 = atanh(r.glucose - diag(diag(r.glucose)));
f2 = atanh(r.glycerol - diag(diag(r.glycerol)));
z = (f1 - f2) / sqrt(1 / (n.glucose - 3) + 1 / (n.glycerol - 3));
pairs = g * (g - 1) / 2;
threshold = sqrt(2) * erfcinv(0.05 / pairs);      % two-sided, Bonferroni
upper = triu(true(g), 1);
differential = abs(z) > threshold & upper;
rewired = sum(differential | differential', 2)';

[i, j] = find(upper);
[~, order] = sort(-abs(z(upper)));                % stable
[~, most] = sortrows(table(-rewired', string(genes')));  % ties: alphabetical
most = most(1:8)';

fprintf('%d gene pairs; |z| > %.2f for %d of them\n', pairs, threshold, ...
    sum(differential, 'all'));
fprintf('Largest changes in correlation (glucose -> glycerol):\n');
for k = order(1:8)'
    fprintf('  %-5s %-5s  %+.2f -> %+.2f\n', genes{i(k)}, genes{j(k)}, ...
        r.glucose(i(k), j(k)), r.glycerol(i(k), j(k)));
end
fprintf('Most rewired genes: %s\n', strjoin(compose('%s (%d)', ...
    string(genes(most)'), rewired(most)'), ', '));

%% Plot: the two correlation matrices
fig = figure('Position', [100 100 1200 550]);
t = linspace(0, 1, 128)';
cmap = [[t; ones(128, 1)], [t; flipud(t)], [ones(128, 1); flipud(t)]];
for c = 1:2
    subplot(1, 2, c)
    imagesc(r.(carbons{c}), [-1, 1]); colormap(cmap); axis square
    set(gca, 'XTick', 1:g, 'XTickLabel', genes, 'YTick', 1:g, ...
        'YTickLabel', genes, 'FontSize', 6);
    xtickangle(90);
    title(sprintf('%s (%d samples)', carbons{c}, n.(carbons{c})), 'FontSize', 11);
end
cb = colorbar; cb.Label.String = 'Pearson r';
sbp.save_figure(fig, mfilename('fullpath'), 'fig-differential-network');

results.genes = genes;
results.n = n;
results.threshold = threshold;
results.r_glucose = r.glucose;
results.r_glycerol = r.glycerol;
results.differential = sum(differential, 'all');
results.rewired = rewired;
results.most_rewired = genes(most);
sbp.save_results(mfilename('fullpath'), results);
