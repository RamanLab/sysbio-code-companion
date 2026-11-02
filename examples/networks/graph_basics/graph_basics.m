%% Basic network parameters for a small undirected network
% Ten proteins, A to J, in two tightly knit groups joined through E.
% We compute degrees, clustering coefficients, shortest-path lengths and
% two centrality measures, using the definitions in the book (Ch. 3).
% Requires: core MATLAB only (graph objects, R2015b or newer).

nodes = cellstr(('A':'J')');
edges = {'A' 'B'; 'A' 'C'; 'A' 'D'; 'B' 'C'; 'B' 'D'; 'C' 'D'; 'D' 'E';
         'E' 'F'; 'F' 'G'; 'F' 'H'; 'G' 'H'; 'G' 'I'; 'H' 'I'; 'I' 'J'};
G = graph(edges(:, 1), edges(:, 2));
G = reordernodes(G, nodes);   % keep nodes in the order A, B, ..., J
n = numnodes(G);

% Adjacency matrix and degrees (book Eq. 3.1)
A = full(adjacency(G));
deg = sum(A, 2);

% Clustering coefficient (Eq. 3.6): the fraction of pairs of neighbours
% that are themselves connected; (A^3)_ii counts each triangle twice
triangles = diag(A^3) / 2;
pairs = deg .* (deg - 1) / 2;
clustering = zeros(n, 1);
clustering(pairs > 0) = triangles(pairs > 0) ./ pairs(pairs > 0);

% Shortest-path lengths between all pairs of nodes
D = distances(G);
diameter = max(D(:));                                  % Eq. 3.11
char_path_length = mean(D(triu(true(n), 1)));          % Eq. 3.12
closeness = (n - 1) ./ sum(D, 2);                      % Eq. 3.14

% Betweenness centrality (Eq. 3.17), counting each pair of nodes once
betweenness = centrality(G, 'betweenness');

fprintf('%d nodes, %d edges, mean degree %.2f\n', n, numedges(G), ...
    mean(deg));
fprintf(['mean clustering coefficient %.4f, diameter %g, ' ...
    'characteristic path length %.4f\n'], mean(clustering), diameter, ...
    char_path_length);
fprintf('node  degree  clustering  closeness  betweenness\n');
for i = 1:n
    fprintf('  %s   %4g    %8.3f   %8.3f   %9.2f\n', nodes{i}, deg(i), ...
        clustering(i), closeness(i), betweenness(i));
end

%% Plot: node size by degree, colour by betweenness, fixed positions
x = [0 1 0 1 2 3 4 3 4 5];
y = [1 1 0 0 0.5 0.5 1 0 0 0];
fig = figure('Position', [100 100 700 320]);
hold on
for e = 1:size(edges, 1)
    i = find(strcmp(nodes, edges{e, 1}));
    j = find(strcmp(nodes, edges{e, 2}));
    plot([x(i) x(j)], [y(i) y(j)], '-', 'Color', [0.6 0.6 0.6]);
end
scatter(x, y, 150 * deg, betweenness, 'filled', 'MarkerEdgeColor', 'k');
text(x, y + 0.12 + 0.03 * deg', nodes, 'HorizontalAlignment', 'center');
colormap(flipud(parula));
cb = colorbar;
cb.Label.String = 'Betweenness centrality';
title('Node size: degree; colour: betweenness');
axis off
ylim([-0.4 1.6]);
sbp.save_figure(fig, mfilename('fullpath'), 'fig-graph-basics');

results.degree = deg';
results.clustering = clustering';
results.mean_clustering = mean(clustering);
results.diameter = diameter;
results.char_path_length = char_path_length;
results.closeness = closeness';
results.betweenness = betweenness';
sbp.save_results(mfilename('fullpath'), results);
