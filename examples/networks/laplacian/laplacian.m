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
fprintf('Zero eigenvalues: %d (connected components)\n', ...
    sum(eigenvalues < 1e-9));
fprintf('Algebraic connectivity: %.4f\n', eigenvalues(2));
fprintf(['The Fiedler vector puts %d of %d members on the side they ' ...
    'actually joined; it misplaces member(s) %s\n'], agree, n, ...
    strjoin(string(misplaced), ', '));

% Remove the friendships between the two halves found by the Fiedler
% vector: the network falls apart, and a second zero eigenvalue appears
cut = side(edges(:, 1)) ~= side(edges(:, 2));
A_cut = A;
for e = find(cut)
    A_cut(edges(e, 1), edges(e, 2)) = 0;
    A_cut(edges(e, 2), edges(e, 1)) = 0;
end
ev_cut = eig(diag(sum(A_cut, 2)) - A_cut);
fprintf(['Cutting the %d edges between the halves leaves %d zero ' ...
    'eigenvalues\n'], sum(cut), sum(ev_cut < 1e-9));

%% Plot: the network, coloured by the sign of the Fiedler vector;
% a black ring marks the members who joined the other faction
fig = figure('Position', [100 100 1100 450]);
subplot(1, 2, 1)
okabe_ito = sbp.palette();
blue = okabe_ito(1, :); red = okabe_ito(4, :);
colours = repmat(red, n, 1);
colours(side == "Mr. Hi", :) = repmat(blue, sum(side == "Mr. Hi"), 1);
h = plot(graph(A), 'Layout', 'force', 'NodeColor', colours, ...
    'MarkerSize', 7, 'EdgeAlpha', 0.3);
hold on
plot(h.XData(misplaced), h.YData(misplaced), 'ko', 'MarkerSize', 11, ...
    'LineWidth', 1.5);
p1 = plot(NaN, NaN, 'o', 'MarkerFaceColor', blue, 'MarkerEdgeColor', 'none');
p2 = plot(NaN, NaN, 'o', 'MarkerFaceColor', red, 'MarkerEdgeColor', 'none');
p3 = plot(NaN, NaN, 'ko', 'LineWidth', 1.5);
legend([p1 p2 p3], {'Fiedler entry > 0', 'Fiedler entry < 0', ...
    'Joined the other faction'}, 'Location', 'southwest', 'Box', 'off');
title('Split by the Fiedler vector'); axis off
subplot(1, 2, 2)
[~, order] = sort(fiedler);
b = bar(fiedler(order), 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = colours(order, :);
xticks(1:n); xticklabels(string(order)); xtickangle(90);
ax = gca; ax.XAxis.FontSize = 7; ylim padded
xlabel('Member', 'FontSize', 11); ylabel('Fiedler vector entry')
sbp.save_figure(fig, mfilename('fullpath'), 'fig-laplacian');

results.eigenvalues = eigenvalues';
results.fiedler = fiedler;
results.agree = agree;
results.misplaced = misplaced;
results.cut_edges = sum(cut);
results.zero_eigenvalues_after_cut = sum(ev_cut < 1e-9);
sbp.save_results(mfilename('fullpath'), results);
