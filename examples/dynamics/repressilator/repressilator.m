%% Oscillations from a ring of three repressors: the repressilator
% Elowitz and Leibler (2000), in dimensionless form. Gene i makes mRNA m_i
% and protein p_i, and each protein represses the next gene round the ring
% (p3 represses gene 1, p1 gene 2, p2 gene 3):
%     dm_i/dt = -m_i + alpha/(1 + p_j^n) + alpha0
%     dp_i/dt = -beta (p_i - m_i)
% The single steady state is unstable for these parameters, and the system
% settles onto a limit cycle. We find the steady state and its eigenvalues,
% simulate, measure the period, and map where in the (alpha, beta) plane
% the steady state is unstable.
% Requires: core MATLAB only.

alpha = 216; alpha0 = 0.216; beta = 5; n = 2;
repressor = [3 1 2];    % gene i is repressed by protein repressor(i)

s = steady_state(alpha, alpha0, n);
max_real = max(real(eig(jacobian(s, alpha, beta, n, repressor))));
state = 'stable';
if max_real > 0, state = 'unstable'; end
fprintf(['Steady state m = p = %.4f; largest real part of an ' ...
    'eigenvalue = %.4f (%s)\n'], s, max_real, state);

%% Simulate from a slightly asymmetric start, and find the period from
% successive upward crossings of p1 through its steady value
rhs = @(t, y) [-y(1:3) + alpha ./ (1 + y(3 + repressor).^n) + alpha0;
               -beta * (y(4:6) - y(1:3))];
crossing = @(t, y) deal(y(4) - s, 0, 1);   % value, terminal, direction
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'Events', crossing);
y0 = [1 0 0 2 1 3];
sol = ode45(rhs, [0 300], y0, opts);
t_up = sol.xe(sol.xe > 150);               % after the transient
period = mean(diff(t_up));
t = linspace(0, 50, 1001);
Y = deval(sol, t);
late = deval(sol, linspace(150, 300, 3001), 4);
amplitude = [min(late) max(late)];
fprintf('Period = %.4f; p1 oscillates between %.4f and %.4f\n', ...
    period, amplitude(1), amplitude(2));

%% Where is the steady state unstable? A grid in the (alpha, beta) plane
alphas = logspace(0, 4, 41);
betas = logspace(-1, 3, 41);
unstable = false(numel(betas), numel(alphas));   % rows: beta
for r = 1:numel(betas)
    for c = 1:numel(alphas)
        ss = steady_state(alphas(c), alpha0, n);
        J = jacobian(ss, alphas(c), betas(r), n, repressor);
        unstable(r, c) = max(real(eig(J))) > 0;
    end
end
fprintf('Steady state unstable at %d of %d grid points\n', ...
    nnz(unstable), numel(unstable));

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
plot(t, Y(4:6, :));
ylim([0 75]);
xlabel('Time'); ylabel('Protein level'); title('Oscillations');
legend({'p1', 'p2', 'p3'}, 'Location', 'north', 'Orientation', ...
    'horizontal', 'Box', 'off');
box off
subplot(1, 2, 2)
contourf(alphas, betas, double(unstable), [0.5 0.5], 'k');
colormap(gca, [1 1 1; 0.776 0.859 0.937]);
clim([0 1]);
hold on
plot(alpha, beta, 'p', 'Color', [0.839 0.153 0.157], ...
    'MarkerFaceColor', [0.839 0.153 0.157], 'MarkerSize', 12);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('\alpha (maximum transcription)');
ylabel('\beta (mRNA / protein lifetime)');
title('Steady state unstable (shaded)');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-repressilator');

results.steady_state = s;
results.max_real_eigenvalue = max_real;
results.period = period;
results.amplitude = amplitude;
results.n_unstable = nnz(unstable);
results.unstable = double(unstable);
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function s = steady_state(alpha, alpha0, n)
% By symmetry m_i = p_i = s, with s = alpha/(1 + s^n) + alpha0.
s = fzero(@(s) alpha / (1 + s^n) + alpha0 - s, [0 alpha + alpha0]);
end

function J = jacobian(s, alpha, beta, n, repressor)
X = alpha * n * s^(n - 1) / (1 + s^n)^2;   % -d(repression)/dp
J = zeros(6);
for i = 1:3
    J(i, i) = -1;
    J(i, 3 + repressor(i)) = -X;
    J(3 + i, i) = beta;
    J(3 + i, 3 + i) = -beta;
end
end
