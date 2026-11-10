%% A metabolic network as a graph: the substrate graph of E. coli core
% In a substrate graph (book §2.5.5), metabolites are nodes, and two are
% joined if one is converted into the other by some reaction: a substrate
% and a product of the same reaction. We build it from the stoichiometric
% matrix of the E. coli core model, leaving out the exchange reactions and
% the biomass reaction. Currency metabolites, such as H+, water, ATP and
% NAD(H), take part in so many reactions that they connect almost
% everything to everything else, and shortest paths through them make no
% biochemical sense. We compare the graph with and without them.

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = full(model.S);
mets = model.mets;
internal = find(~startsWith(model.rxns, {'EX_', 'Biomass'}));

M = zeros(numel(mets));
for j = internal'
    substrates = find(S(:, j) < 0);
    products = find(S(:, j) > 0);
    M(substrates, products) = 1;
    M(products, substrates) = 1;
end
G = graph(M, mets);

currency = {'h_c', 'h_e', 'h2o_c', 'h2o_e', 'atp_c', 'adp_c', 'amp_c', ...
    'pi_c', 'pi_e', 'nad_c', 'nadh_c', 'nadp_c', 'nadph_c', ...
    'co2_c', 'co2_e', 'coa_c', 'nh4_c', 'nh4_e', 'o2_c', 'q8_c', 'q8h2_c'};
H = rmnode(G, currency);

results.with_currency = summary(G);
results.without_currency = summary(H);
deg = degree(G);
[~, order] = sortrows(table(-deg, string(mets)));    % by degree, then name
results.top_degree = mets(order(1:10))';
results.top_degree_values = deg(order(1:10))';
results.glc_to_succ.with_currency = distances(G, 'g6p_c', 'succ_c');
results.glc_to_succ.without_currency = distances(H, 'g6p_c', 'succ_c');

fprintf('Most connected metabolites: %s\n', strjoin(compose('%s (%d)', ...
    string(mets(order(1:10))), deg(order(1:10))), ', '));
keys = {'with_currency', 'without_currency'};
for c = 1:2
    r = results.(keys{c});
    fprintf(['%s: %d metabolites, %d edges, %d component(s); in the ' ...
        'largest (%d), mean path %.2f, diameter %d\n'], ...
        strrep(keys{c}, '_', ' '), r.nodes, r.edges, r.components, ...
        r.largest, r.mean_path, r.diameter);
end
fprintf('A shortest path from glucose 6-phosphate to succinate:\n');
fprintf('  with currency metabolites:    %s\n', ...
    strjoin(shortestpath(G, 'g6p_c', 'succ_c'), ' - '));
fprintf('  without currency metabolites: %s\n', ...
    strjoin(shortestpath(H, 'g6p_c', 'succ_c'), ' - '));

%% Plot
fig = figure('Position', [100 100 1100 400]);
subplot(1, 2, 1)
top = order(1:15);
b = bar(deg(top), 'FaceColor', 'flat');
is_currency = ismember(mets(top), currency);
b.CData = repmat([0.122 0.467 0.706], 15, 1);
b.CData(is_currency, :) = repmat([0.839 0.153 0.157], sum(is_currency), 1);
set(gca, 'XTick', 1:15, 'XTickLabel', mets(top), 'TickLabelInterpreter', 'none');
xtickangle(60); ylabel('Degree');
title('Most connected metabolites (red: currency)'); box off
subplot(1, 2, 2); hold on
colours = {[0.839 0.153 0.157], [0.122 0.467 0.706]};
for c = 1:2
    counts = results.(keys{c}).path_counts;
    plot(1:numel(counts), counts / sum(counts), 'o-', 'Color', colours{c});
end
legend({'with currency', 'without'}, 'Box', 'off');
xlabel('Shortest path length'); ylabel('Fraction of pairs');
title('Path lengths in the largest component'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-metabolic-graphs');

sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function r = summary(G)
% Size, and path lengths within the largest connected component.
bins = conncomp(G);
largest = mode(bins);
sub = subgraph(G, find(bins == largest));
d = distances(sub);
lengths = d(triu(true(size(d)), 1));
r.nodes = numnodes(G);
r.edges = numedges(G);
r.components = max(bins);
r.largest = numnodes(sub);
r.mean_path = mean(lengths);
r.diameter = max(lengths);
r.path_counts = accumarray(lengths, 1)';
end
