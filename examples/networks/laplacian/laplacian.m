%% The Laplacian matrix, and what its eigenvectors say about a network
% Zachary's karate club: 34 members, and 78 friendships observed outside
% the club. A dispute between the instructor (member 1, Mr. Hi) and the
% administrator (member 34, the Officer) split the club in two. The
% Laplacian is L = D - A (book §2.4.3), where D holds the degrees on its
% diagonal. Its eigenvalues are all >= 0; the number of zero eigenvalues
% is the number of connected components; and the eigenvector of the
% second-smallest eigenvalue (the Fiedler vector) splits the network into
% two well-separated parts, by the signs of its entries.

edges = readmatrix(fullfile(sbp.data_dir(), 'karate_edges.csv'));
factions = readtable(fullfile(sbp.data_dir(), 'karate_factions.csv'), ...
    'TextType', 'string');
faction = factions.faction';
n = numel(faction);
A = zeros(n);
for e = 1:size(edges, 1)
    A(edges(e, 1), edges(e, 2)) = 1;
    A(edges(e, 2), edges(e, 1)) = 1;
end
L = diag(sum(A, 2)) - A;

[V, D] = eig(L);
[eigenvalues, order] = sort(diag(D));             % ascending order
fiedler = V(:, order(2))';
fiedler = fiedler * sign(fiedler(1));             % member 1 positive
side = repmat("Officer", 1, n);
side(fiedler > 0) = "Mr. Hi";
agree = sum(side == faction);
misplaced = find(side ~= faction);

fprintf('Smallest eigenvalues: %s\n', mat2str(round(eigenvalues(1:4)', 4)));
fprintf('Zero eigenvalues: %d (connected components)\n', sum(eigenvalues < 1e-9));
fprintf('Algebraic connectivity: %.4f\n', eigenvalues(2));
fprintf(['The Fiedler vector puts %d of %d members on the side they ' ...
    'actually joined; it misplaces member(s) %s\n'], agree, n, ...
    strjoin(string(misplaced), ', '));

% Remove the friendships between the two halves found by the Fiedler
% vector: the network falls apart, and a second zero eigenvalue appears
cut = side(edges(:, 1)) ~= side(edges(:, 2));
A_cut = A;
for e = find(cut)'
    A_cut(edges(e, 1), edges(e, 2)) = 0;
    A_cut(edges(e, 2), edges(e, 1)) = 0;
end
ev_cut = eig(diag(sum(A_cut, 2)) - A_cut);
fprintf('Cutting the %d edges between the halves leaves %d zero eigenvalues\n', ...
    sum(cut), sum(ev_cut < 1e-9));

%% Plot: the network, coloured by the sign of the Fiedler vector
fig = figure('Position', [100 100 1100 450]);
subplot(1, 2, 1)
blue = [0.122 0.467 0.706]; red = [0.839 0.153 0.157];
colours = repmat(red, n, 1);
colours(side == "Mr. Hi", :) = repmat(blue, sum(side == "Mr. Hi"), 1);
h = plot(graph(A), 'Layout', 'force', 'NodeColor', colours, ...
    'MarkerSize', 7, 'EdgeAlpha', 0.3);
highlight(h, misplaced, 'MarkerSize', 11);
title('Split by the Fiedler vector'); axis off
subplot(1, 2, 2)
[~, order] = sort(fiedler);
b = bar(fiedler(order), 'FaceColor', 'flat');
b.CData = colours(order, :);
set(gca, 'XTick', 1:n, 'XTickLabel', order, 'FontSize', 6);
xlabel('Member'); ylabel('Fiedler vector entry'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-laplacian');

results.eigenvalues = eigenvalues';
results.fiedler = fiedler;
results.agree = agree;
results.misplaced = misplaced;
results.cut_edges = sum(cut);
results.zero_eigenvalues_after_cut = sum(ev_cut < 1e-9);
sbp.save_results(mfilename('fullpath'), results);
