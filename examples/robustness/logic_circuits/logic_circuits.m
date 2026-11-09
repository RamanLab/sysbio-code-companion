%% Robustness and evolvability in a space of logic circuits
% A circuit is a chain of four NAND gates. Each input of gate g (g = 1..4)
% is wired to one of the two circuit inputs, A and B, or to the output of
% an earlier gate; the output of gate 4 is the output of the circuit. The
% wiring is the genotype, and the logic function the circuit computes, one
% of 16 possible truth tables, is its phenotype. A mutation rewires one
% input of one gate. We enumerate all 14,400 circuits, and ask how robust
% each phenotype is to mutations, and how many new phenotypes lie one
% mutation away (its evolvability).
% Requires: core MATLAB only.

% Truth tables as 4-bit numbers, outputs for (A, B) = 00, 01, 10, 11
A = 3; B = 5;          % 0011 and 0101
NAMES = {'FALSE', 'A AND B', 'A AND NOT B', 'A', 'NOT A AND B', 'B', ...
    'A XOR B', 'A OR B', 'A NOR B', 'A XNOR B', 'NOT B', 'A OR NOT B', ...
    'NOT A', 'NOT A OR B', 'A NAND B', 'TRUE'};

% A genotype is 8 digits, the sources of the two inputs of each gate
% (numbered from 0); gate g can choose among 2 + (g - 1) sources
radix = [2 2 3 3 4 4 5 5];
weight = [fliplr(cumprod(fliplr(radix(2:end)))) 1];   % first digit slowest
n = prod(radix);
index = (0:n - 1)';
digits = mod(floor(index ./ weight), radix);           % n x 8

% Phenotypes: evaluate all circuits at once, with bitwise NAND
signals = [A * ones(n, 1), B * ones(n, 1)];
for g = 1:4
    x = signals(sub2ind(size(signals), index + 1, digits(:, 2 * g - 1) + 1));
    y = signals(sub2ind(size(signals), index + 1, digits(:, 2 * g) + 1));
    signals(:, end + 1) = 15 - bitand(x, y); %#ok<AGROW>
end
phenotype = signals(:, end);

% One-mutation neighbours: change one digit to each of its other values
same = zeros(n, 1);          % neighbours with the same phenotype
seen = zeros(n, 1);          % phenotypes of neighbours, as bits
edges = zeros(0, 2);
for d = 1:8
    for v = 0:radix(d) - 1
        mutant = digits(:, d) ~= v;
        nb = index(mutant) + (v - digits(mutant, d)) * weight(d);
        same_ph = phenotype(nb + 1) == phenotype(mutant);
        same(mutant) = same(mutant) + same_ph;
        seen(mutant) = bitor(seen(mutant), bitshift(1, phenotype(nb + 1)));
        from = index(mutant);
        edges = [edges; from(same_ph) + 1, nb(same_ph) + 1]; %#ok<AGROW>
    end
end
n_neighbours = sum(radix - 1);                         % 20
robustness = same / n_neighbours;
popcount = @(b) sum(dec2bin(b, 16) == '1', 2);
not_own = 65535 - bitshift(1, phenotype);   % all bits but the phenotype's
evolvability = popcount(bitand(seen, not_own));
component = conncomp(graph(edges(:, 1), edges(:, 2), [], n))';

% Summarise for each phenotype that occurs
found = unique(phenotype)';
summary = struct('phenotype', found, 'name', {NAMES(found + 1)}, ...
    'n_genotypes', [], 'robustness', [], 'evolvability', [], ...
    'n_components', []);
for f = found
    g = phenotype == f;
    reach = 0;
    for b = seen(g)'
        reach = bitor(reach, b);
    end
    reach = bitand(reach, 65535 - bitshift(1, f));
    summary.n_genotypes(end + 1) = nnz(g);
    summary.robustness(end + 1) = mean(robustness(g));
    summary.evolvability(end + 1) = popcount(reach);
    summary.n_components(end + 1) = numel(unique(component(g)));
    fprintf(['%-12s %5d circuits, robustness %.3f, %2d new phenotypes ' ...
        'nearby, %d genotype network(s)\n'], NAMES{f + 1}, nnz(g), ...
        mean(robustness(g)), summary.evolvability(end), ...
        summary.n_components(end));
end
fprintf('%d circuits compute %d of the 16 functions\n', n, numel(found));

c = corrcoef(robustness, evolvability);
r_genotype = c(1, 2);
c = corrcoef(summary.robustness, summary.evolvability);
r_phenotype = c(1, 2);
fprintf(['Correlation of robustness and evolvability: %.3f for ' ...
    'genotypes, %.3f for phenotypes\n'], r_genotype, r_phenotype);

%% Plot
blue = [0.122 0.467 0.706]; orange = [1.000 0.498 0.055];
fig = figure('Position', [100 100 1000 420]);
subplot(1, 2, 1)
semilogx(summary.n_genotypes, summary.robustness, 'o', 'Color', blue, ...
    'MarkerFaceColor', blue);
labelled = ismember(summary.name, ...
    {'A XOR B', 'A NOR B', 'FALSE', 'A OR B', 'A NAND B'});
text(summary.n_genotypes(labelled) * 1.15, ...
    summary.robustness(labelled), summary.name(labelled), 'FontSize', 8);
xlim([8 8000]);
xlabel('Number of circuits with this function');
ylabel('Robustness of the function');
title('Common functions are robust');
box off
subplot(1, 2, 2)
levels = unique(robustness);
mean_ev = arrayfun(@(r) mean(evolvability(robustness == r)), levels);
plot(levels, mean_ev, 'o-', 'Color', orange, 'MarkerFaceColor', orange);
hold on
plot(summary.robustness, summary.evolvability, 's', 'Color', blue, ...
    'MarkerFaceColor', blue);
xlabel('Robustness'); ylabel('New functions one mutation away');
title('Robustness and evolvability');
legend({'single circuits', 'functions (all their circuits)'}, ...
    'Location', 'southwest', 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-logic-circuits');

results.n_circuits = n;
results.summary = summary;
results.correlation_genotype = r_genotype;
results.correlation_phenotype = r_phenotype;
sbp.save_results(mfilename('fullpath'), results);
