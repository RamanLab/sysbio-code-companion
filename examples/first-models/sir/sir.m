%% The SIR model of an epidemic (book Eqs. 1.6 to 1.8)
%     dS/dt = -beta*S*I
%     dI/dt =  beta*S*I - gamma*I
%     dR/dt =  gamma*I
% S, I and R are fractions of the population. With almost everyone
% susceptible at the start, an outbreak grows only if the basic
% reproduction number R0 = beta/gamma exceeds 1. The fraction finally
% infected, R_inf, then satisfies  R_inf = 1 - S0*exp(-R0*R_inf).
% Requires: core MATLAB only.

gamma = 0.1;                       % recovery rate: 10 days to recover
I0 = 1e-4;                         % one person in 10,000 infected at first
S0 = 1 - I0;
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
sir = @(t, y, beta) [-beta * y(1) * y(2);
                     beta * y(1) * y(2) - gamma * y(2);
                     gamma * y(2)];
% Final size: the one root between 0 and 1 of R_inf = 1 - S0*exp(-R0*R_inf)
final_size = @(R0) fzero(@(x) x - (1 - S0 * exp(-R0 * x)), [0 1]);

%% One epidemic, R0 = 3
beta = 0.3;
t = linspace(0, 200, 201);         % days
[~, Y] = ode45(@(t, y) sir(t, y, beta), t, [S0 I0 0], opts);
S = Y(:, 1); I = Y(:, 2); R = Y(:, 3);
[~, peak] = max(I);
fprintf(['R0 = %g: peak of %.4f infected on day %g; after 200 days ' ...
    'R = %.4f (final-size equation: %.4f)\n'], beta / gamma, I(peak), ...
    t(peak), R(end), final_size(beta / gamma));

%% Final size for a range of R0
R0s = linspace(0.5, 4, 36);
simulated = zeros(size(R0s));
theory = zeros(size(R0s));
for k = 1:numel(R0s)
    [~, Yk] = ode45(@(t, y) sir(t, y, R0s(k) * gamma), [0 5000], ...
        [S0 I0 0], opts);
    simulated(k) = Yk(end, 3);
    theory(k) = final_size(R0s(k));
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
plot(t, S, '-', 'Color', [0.122 0.467 0.706]);
hold on
plot(t, I, '-', 'Color', [0.839 0.153 0.157]);
plot(t, R, '-', 'Color', [0.173 0.627 0.173]);
xlabel('Time (days)'); ylabel('Fraction of the population');
title('An epidemic with R_0 = 3');
legend({'S, susceptible', 'I, infected', 'R, recovered'}, ...
    'Location', 'east', 'Box', 'off');
box off

subplot(1, 2, 2)
plot(R0s, theory, '-', 'Color', [0.6 0.6 0.6]);
hold on
plot(R0s, simulated, 'o', 'Color', [0.173 0.627 0.173], ...
    'MarkerFaceColor', [0.173 0.627 0.173], 'MarkerSize', 4);
xline(1, ':', 'Color', [0.6 0.6 0.6]);
xlabel('Basic reproduction number R_0'); ylabel('Fraction ever infected');
title('Final size of the epidemic');
legend({'final-size equation', 'simulation'}, 'Location', 'southeast', ...
    'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-sir');

results.peak_infected = I(peak);
results.peak_day = t(peak);
results.final_R = R(end);
results.final_size_R0_3 = final_size(3.0);
results.R0 = R0s;
results.final_size_simulated = simulated;
sbp.save_results(mfilename('fullpath'), results);
