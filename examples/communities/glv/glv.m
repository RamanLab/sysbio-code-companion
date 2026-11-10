%% A generalised Lotka-Volterra (gLV) model of a three-species community,
% and estimating its parameters from time courses.
%
%     dx_i/dt = x_i (r_i + sum_j a_ij x_j)
%
% r_i is the growth rate of species i on its own, a_ii < 0 limits its
% growth, and a_ij (i ~= j) is the effect of species j on species i:
% negative for competition, positive for cross-feeding. We find the
% steady state where all three species coexist, x* = -A^-1 r, and its
% stability, from the eigenvalues of the Jacobian diag(x*) A.
%
% Then we turn the problem around, as one does with real abundance data
% (book §11.1.2): we simulate three experiments that start from different
% mixtures, "measure" the abundances every half day with 5% noise, and
% estimate r and A by linear regression, since
%     d ln x_i / dt = r_i + sum_j a_ij x_j
% is linear in the parameters. The noise is Gaussian, made from
% sbp.uniform by the Box-Muller method, so that it is identical in MATLAB
% and Python.

species = {'A', 'B', 'C'};
r = [1.0; 0.8; 0.6];
A = [-1.0, -0.4, 0.3;
     -0.5, -1.0, 0.0;
      0.2, -0.3, -0.8];

x_star = -A \ r;
ev = eig(diag(x_star) * A);
[~, order] = sortrows([real(ev), imag(ev)]);
ev = ev(order);
fprintf('Coexistence steady state: %s\n', strjoin(compose('%s = %.3f', ...
    string(species'), x_star), ', '));
fprintf('Eigenvalues of the Jacobian: %s  (all real parts negative: stable)\n', ...
    strjoin(compose('%.3f%+.3fi', real(ev), imag(ev)), ', '));

% Three experiments, sampled every half day for 10 days, with 5% noise
t = 0:0.5:10;
starts = [0.05, 0.05, 0.05; 0.5, 0.02, 0.1; 0.02, 0.6, 0.3];
K = size(starts, 1); T = numel(t);
u = sbp.uniform(5, 2 * K * T * 3);
gauss = sqrt(-2 * log(u(1:2:end))) .* cos(2 * pi * u(2:2:end));
gauss = reshape(gauss, 3, T, K);                  % species x time x experiment
data = cell(1, K);
for k = 1:K
    data{k} = simulate(r, A, starts(k, :), t) .* (1 + 0.05 * gauss(:, :, k)');
end

% Regression: (ln x(t + dt) - ln x(t)) / dt  =  r + A x(t)
X = []; Y = [];
for k = 1:K
    X = [X; ones(T - 1, 1), data{k}(1:end - 1, :)]; %#ok<AGROW>
    Y = [Y; diff(log(data{k})) ./ diff(t)']; %#ok<AGROW>
end
theta = X \ Y;                                    % one column per species
r_hat = theta(1, :);
A_hat = theta(2:end, :)';
fprintf('Estimated growth rates:  %s   (true %s)\n', ...
    strjoin(compose('%.2f', r_hat), ', '), strjoin(compose('%.2f', r'), ', '));
fprintf('Estimated interactions:\n');
for i = 1:3
    fprintf('  %s: %s   (true %s)\n', species{i}, ...
        strjoin(compose('%+.2f', A_hat(i, :)), '  '), ...
        strjoin(compose('%+.2f', A(i, :)), '  '));
end

%% Plot: data, and the fitted model's predictions
fig = figure('Position', [100 100 1200 360]);
tt = linspace(0, 10, 201);
colours = [0.122 0.467 0.706; 1.000 0.498 0.055; 0.173 0.627 0.173];
for k = 1:K
    subplot(1, 3, k); hold on
    fitted = simulate(r_hat', A_hat, data{k}(1, :), tt);
    for i = 1:3
        plot(t, data{k}(:, i), 'o', 'Color', colours(i, :), 'MarkerSize', 3, ...
            'HandleVisibility', 'off');
        plot(tt, fitted(:, i), 'Color', colours(i, :), 'DisplayName', species{i});
    end
    xlabel('Time (days)'); title(sprintf('Experiment %d', k)); box off
    if k == 1, ylabel('Abundance'); legend('Box', 'off'); end
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-glv');

results.steady_state = x_star';
results.eigenvalues_real = real(ev)';
results.eigenvalues_imag = imag(ev)';
results.r_hat = r_hat;
results.A_hat = A_hat;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function x = simulate(r, A, x0, t)
% Rows: times; columns: species.
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
[~, x] = ode45(@(~, x) x .* (r + A * x), t, x0(:), opts);
end
