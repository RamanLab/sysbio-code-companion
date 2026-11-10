%% A probabilistic Boolean network (PBN), analysed exactly
% We start from the three-gene network of §7.2 (book Fig. 7.1):
%     g1* = g2 AND NOT g3,    g2* = NOT g1,    g3* = NOT g3 OR g2,
% whose attractors are the steady state 011 and the cycle 001 -> 010 -> 111.
% In the PBN (book §7.3.1), g3 follows its rule with probability 0.8, and
% an alternative rule, g3* = NOT g1, with probability 0.2; at each step,
% one of the two is picked at random. Also, as proposed by Shmulevich and
% colleagues, each gene may be flipped at random with probability
% p = 0.01, as if by noise. The PBN is then a Markov chain on the 2^3 = 8
% states, and we can compute its transition matrix, and the long-run
% probability of each state (the stationary distribution), exactly. For
% comparison, we do the same for the original network with noise alone.

n = 3;
p_flip = 0.01;
original = @(g1, g2, g3) ~g3 || g2;
alternative = @(g1, g2, g3) ~g1;
P_noise = transition_matrix({original}, 1.0, n, p_flip);
P_pbn = transition_matrix({original, alternative}, [0.8, 0.2], n, p_flip);
pi_noise = stationary(P_noise);
pi_pbn = stationary(P_pbn);

labels = cellstr(dec2bin(0:2 ^ n - 1, n))';
cycle = find(ismember(labels, {'001', '010', '111'}));
steady = find(strcmp(labels, '011'));
fprintf('State   noise only   PBN\n');
for s = 1:2 ^ n
    fprintf('  %s     %.4f    %.4f\n', labels{s}, pi_noise(s), pi_pbn(s));
end
fprintf('Noise only: steady state 011 %.3f; cycle %.3f\n', ...
    pi_noise(steady), sum(pi_noise(cycle)));
fprintf('PBN: steady state 011 %.3f; cycle %.3f\n', ...
    pi_pbn(steady), sum(pi_pbn(cycle)));

%% Plot
fig = figure('Position', [100 100 700 400]);
b = bar(categorical(labels, labels), [pi_noise; pi_pbn]');
b(1).FaceColor = [0.6 0.6 0.6]; b(2).FaceColor = [0.122 0.467 0.706];
xlabel('State (g1 g2 g3)'); ylabel('Long-run probability');
legend({'noise only', 'PBN'}, 'Box', 'off'); box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-probabilistic');

results.states = labels;
results.transition = P_pbn;
results.stationary_noise = pi_noise;
results.stationary_pbn = pi_pbn;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function P = transition_matrix(rules, probs, n, p_flip)
% Markov chain of the PBN, for the rules for g3 and their probabilities.
% States are numbered 0 to 2^n - 1, first gene as the leftmost digit;
% row/column s + 1 of P is state s.
T = zeros(2 ^ n);
for s = 0:2 ^ n - 1
    g = bitget(s, n:-1:1);                        % [g1, g2, g3]
    for r = 1:numel(rules)
        nxt = [g(2) && ~g(3), ~g(1), rules{r}(g(1), g(2), g(3))];
        t = nxt * 2 .^ (n - 1:-1:0)';
        T(s + 1, t + 1) = T(s + 1, t + 1) + probs(r);
    end
end
% With probability (1 - p)^n no gene flips and the network acts;
% otherwise, the genes that flip are flipped instead
P = (1 - p_flip) ^ n * T;
for s = 0:2 ^ n - 1
    for flips = 1:2 ^ n - 1                      % which genes flip (as bits)
        k = sum(bitget(flips, 1:n));
        t = bitxor(s, flips);
        P(s + 1, t + 1) = P(s + 1, t + 1) + p_flip ^ k * (1 - p_flip) ^ (n - k);
    end
end
assert(all(abs(sum(P, 2) - 1) < 1e-12));
end

function p = stationary(P)
% pi P = pi, with the probabilities summing to 1.
A = [P' - eye(size(P)); ones(1, size(P, 1))];
b = [zeros(size(P, 1), 1); 1];
p = (A \ b)';
end
