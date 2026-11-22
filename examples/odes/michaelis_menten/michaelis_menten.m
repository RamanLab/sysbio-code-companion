%% Full mass-action enzyme model vs the Michaelis-Menten rate law
%     E + S <-> ES -> E + P      (rate constants k1, k_1, k2)
% The full model tracks S, ES and P; the enzyme is E = E0 - ES. The
% Michaelis-Menten (Briggs-Haldane) approximation replaces all of this
% with a single rate law, v = vmax*S/(KM + S). How good is it?
% Requires: core MATLAB only.

k1 = 1.0; k_1 = 1.0; k2 = 1.0;     % 1/(mM s), 1/s, 1/s
KM = (k_1 + k2) / k1;              % Michaelis constant: 2 mM
S0 = 10.0;                         % initial substrate (mM)
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);  % tight tolerances

% Two cases: little enzyme (E0 << S0) and a lot of enzyme (E0 ~ S0)
names = {'low_enzyme', 'high_enzyme'};
E0s = [0.1 5.0];
t_ends = [150 6];

fig = figure('Position', [100 100 1000 400]);
for k = 1:2
    E0 = E0s(k);
    t = linspace(0, t_ends(k), 51);

    % Full model: y = [S; ES; P]
    full_model = @(t, y) [-k1*(E0 - y(2))*y(1) + k_1*y(2)
                          k1*(E0 - y(2))*y(1) - (k_1 + k2)*y(2)
                          k2*y(2)];
    [~, y] = ode45(full_model, t, [S0; 0; 0], opts);
    P_full = y(:, 3)';

    % Michaelis-Menten model: a single ODE for S
    vmax = k2 * E0;
    mm_model = @(t, S) -vmax * S / (KM + S);
    [~, S] = ode45(mm_model, t, S0, opts);
    P_mm = S0 - S';                % in the MM model, P = S0 - S

    err = max(abs(P_full - P_mm)) / S0;
    fprintf('E0 = %g mM: max |P_full - P_MM| = %.1f%% of S0\n', E0, 100*err);
    results.(names{k}) = struct('E0', E0, 't', t, 'P_full', P_full, ...
        'P_mm', P_mm, 'max_rel_difference', err);

    subplot(1, 2, k)
    plot(t, P_full, '-', 'Color', [0.000 0.447 0.698], 'LineWidth', 2);
    hold on
    plot(t, P_mm, '--', 'Color', [0.902 0.624 0.000], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Product P (mM)');
    title(sprintf('E_0 = %g mM, S_0 = %g mM', E0, S0));
    legend({'full mass action', 'Michaelis–Menten'}, ...
        'Location', 'southeast', 'Box', 'off');
    box off
end
sbp.save_figure(fig, mfilename('fullpath'), 'fig-michaelis-menten');

sbp.save_results(mfilename('fullpath'), results);
