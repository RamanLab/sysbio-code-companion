%% Dynamic FBA: E. coli growing on glucose in a batch culture
% FBA predicts steady-state rates. Dynamic FBA follows a culture through
% time by solving an FBA problem at each short time step: the current
% concentrations set the maximum uptake rates, FBA gives the growth rate
% and the exchange fluxes, and these update biomass and concentrations
% (Mahadevan, Edwards and Doyle, 2002). Oxygen uptake is limited, as in a
% dense culture, so cells growing fast on glucose overflow acetate, which
% they consume once the glucose runs out.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = model.S; lb0 = model.lb; ub0 = model.ub; ids = model.rxns;
n = numel(ids);
bio = find(strcmp(ids, 'Biomass_Ecoli_core'));
glc = find(strcmp(ids, 'EX_glc__D_e'));
ac = find(strcmp(ids, 'EX_ac_e'));
o2 = find(strcmp(ids, 'EX_o2_e'));

% Uptake kinetics (mmol/gDW/h, mM) and the oxygen limit
vmax_glc = 10; km_glc = 0.015;
vmax_ac = 5; km_ac = 0.015;
o2_max = 15;

dt = 0.01; t_end = 12;                 % h
X = 0.01; G = 20; A = 0;               % biomass (gDW/L), glucose, acetate
times = round(0:dt:t_end, 10);
traj = zeros(numel(times), 4);         % X, G, A, growth rate
c = zeros(n, 1); c(bio) = -1;          % linprog minimises
opts = optimoptions('linprog', 'Display', 'none');
for k = 1:numel(times)
    % Uptake is limited by kinetics, and by what is left in the medium
    lb = lb0; ub = ub0;
    lb(glc) = -min(vmax_glc * G / (km_glc + G), G / (X * dt));
    lb(ac) = -min(vmax_ac * A / (km_ac + A), A / (X * dt));
    lb(o2) = -o2_max;
    [v, f, exitflag] = linprog(c, [], [], S, zeros(size(S, 1), 1), ...
        lb, ub, opts);
    if exitflag == 1
        mu = max(0, -f);
    else                               % cannot even meet maintenance
        mu = 0; v = zeros(n, 1);
    end
    traj(k, :) = [X G A mu];
    % Euler step: dX/dt = mu X, dC/dt = v_exchange X
    G = max(0, G + v(glc) * X * dt);
    A = max(0, A + v(ac) * X * dt);
    X = X + mu * X * dt;
end

X_t = traj(:, 1); G_t = traj(:, 2); A_t = traj(:, 3); mu_t = traj(:, 4);
t_glc = times(find(G_t < 1e-9, 1));            % glucose used up
[A_peak, k_peak] = max(A_t);
t_stop = times(find(mu_t > 0, 1, 'last') + 1);
fprintf(['Glucose runs out at t = %.2f h; acetate peaks at %.3f mM at ' ...
    't = %.2f h\n'], t_glc, A_peak, times(k_peak));
fprintf(['Growth on glucose at %.4f /h, then on acetate at %.4f /h; ' ...
    'growth stops at t = %.2f h with %.4f gDW/L\n'], mu_t(1), ...
    mu_t(find(times > t_glc + 0.1, 1)), t_stop, X_t(end));

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
semilogy(times, X_t, 'Color', [0.173 0.627 0.173]);
xline(t_glc, ':', 'Color', [0.6 0.6 0.6]);
xlabel('Time (h)'); ylabel('Biomass (gDW/L)'); title('Biomass');
box off
subplot(1, 2, 2)
plot(times, G_t, 'Color', [0.122 0.467 0.706]);
hold on
plot(times, A_t, 'Color', [1.000 0.498 0.055]);
xline(t_glc, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
xlabel('Time (h)'); ylabel('Concentration (mM)');
title('Substrates and by-products');
legend({'glucose', 'acetate'}, 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-dynamic-fba');

hourly = abs(times - round(times)) < 1e-9;
results.t_glucose_exhausted = t_glc;
results.acetate_peak = A_peak;
results.t_acetate_peak = times(k_peak);
results.t_growth_stops = t_stop;
results.growth_on_glucose = mu_t(1);
results.final_biomass = X_t(end);
results.hours = times(hourly);
results.biomass_hourly = X_t(hourly)';
results.glucose_hourly = G_t(hourly)';
results.acetate_hourly = A_t(hourly)';
sbp.save_results(mfilename('fullpath'), results);
