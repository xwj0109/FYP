%% eSSVI Calibration & Comparison for HestonIV vs IV_mid
% Calibrate global eSSVI (power-law rho) to both HestonIV and IV_mid, then compare
clear; close all; clc;

%% 1. Load Data
T_matur = readtable('Matlab_df.xlsx').maturity;
K_strike = readtable('Matlab_df.xlsx').strike;
Fwd      = readtable('Matlab_df.xlsx').forward;
riskFree = readtable('Matlab_df.xlsx').interest_rate;
IV_mid   = readtable('Matlab_df.xlsx').implied_volatility;
HestonIV  = readmatrix('HestonIV.xlsx');  % same dimensions/order as IV_mid

q = 0;  % dividend yield
Spot = Fwd .* exp(-riskFree .* T_matur);

%% 2. Build per‐tenor cells
T_vectU = unique(T_matur);
N       = numel(T_vectU);

K_cell     = cell(1,N);
F0_cell    = cell(1,N);
k_arr      = cell(1,N);
marketVols_mid   = cell(1,N);
marketVols_model = cell(1,N);

for i = 1:N
    idx = T_matur == T_vectU(i);
    K_cell{i}    = K_strike(idx);
    F0_cell{i}   = Fwd(find(idx,1));
    k_arr{i}     = log(K_cell{i} ./ F0_cell{i});
    marketVols_mid{i}   = IV_mid(idx);
    marketVols_model{i} = HestonIV(idx);
end

%% 3. Calibration function handle
objFun = @(x, mkt) eSSVI_obj_powerlaw(x, T_vectU, k_arr, mkt);

%% 4. Shared initial guesses & bounds
% ---- rho0, alpha
init_rho0  = 0;
init_alpha = 0.5;
% ---- ATM total‐variance θ_i guesses
theta_guess = zeros(1,N);
for i = 1:N
    [~, atm] = min(abs(k_arr{i}));
    theta_guess(i) = marketVols_mid{i}(atm)^2 * T_vectU(i);
end


for i = 2:N
    theta_guess(i) = max(theta_guess(i), 1.05*theta_guess(i-1));
end
init_theta1 = max(theta_guess(1),1e-12);
init_a      = max(diff(theta_guess),1e-6);
init_c      = 0.5 * ones(1,N);

X0 = [init_rho0, init_alpha, init_theta1, init_a, init_c];
LB = [-0.999, 0, 1e-12, zeros(1,N-1), zeros(1,N)];
UB = [ 0.999, 5, Inf,      Inf(1,N-1), 0.9999*ones(1,N)];

opts = optimoptions('lsqnonlin', ...
    'Display','iter','UseParallel',true, ...
    'MaxFunctionEvaluations',1e6, ...
    'FunctionTolerance',1e-8, ...
    'StepTolerance',1e-6);

%% 5. Calibrate to HestonIV
[param_model,~,~] = lsqnonlin(@(x) objFun(x, marketVols_model), X0, LB, UB, opts);

%% 6. Calibrate to IV_mid
[param_mid,~,~]   = lsqnonlin(@(x) objFun(x, marketVols_mid),   X0, LB, UB, opts);

%% 7. Extract & reconstruct slices for each calibration
compute_slices = @(params) deal_parameters(params, T_vectU, k_arr);

[theta_model, rho_model, psi_model] = compute_slices(param_model);
[theta_mid,   rho_mid,   psi_mid]   = compute_slices(param_mid);

% Compute model vols for each tenor
modelVols_model = cell(1,N);
modelVols_mid   = cell(1,N);
for i = 1:N
    w_model = total_variance(theta_model(i), rho_model(i), psi_model(i), k_arr{i});
    w_mid   = total_variance(theta_mid(i),   rho_mid(i),   psi_mid(i),   k_arr{i});
    modelVols_model{i} = sqrt(max(w_model,0) ./ T_vectU(i));
    modelVols_mid{i}   = sqrt(max(w_mid,0)   ./ T_vectU(i));
end

%% 8. Plot comparison on the first maturity only
i0 = 1;
Kp = exp(k_arr{i0}) * F0_cell{i0};
days = round(T_vectU(i0) * 365);  % convert years→days

figure; hold on;
plot(Kp, modelVols_model{i0},'-r','LineWidth',1.5, 'DisplayName','Heston Implied Volatility');
plot(Kp, modelVols_mid{i0},    '-b','LineWidth',1.5, 'DisplayName','Market Implied Volatility');
xlabel('Strike K'); ylabel('Implied Vol (%)');
title(sprintf('Heston VS market at T = %.2f', days));
legend('Location','Best'); grid on;

%% --- Helper functions ---

function [theta, rho, psi] = deal_parameters(x, T_vectU, k_arr)
    % Extract x
    rho0   = x(1);
    alpha  = x(2);
    theta1 = x(3);
    N      = numel(T_vectU);
    a_vals = x(4:3+N-1);
    c_vals = x(3+N:end);

    % Reconstruct rho and theta
    rho = rho0 .* T_vectU.^(-alpha);
    theta = zeros(1,N); theta(1) = theta1;
    for j = 2:N
        p_fac = max((1+rho(j-1))/(1+rho(j)), (1 - rho(j-1))/(1 - rho(j)));
        theta(j) = theta(j-1)*p_fac + max(a_vals(j-1),1e-12);
    end

    % Build psi globally
    p_vec = ones(1,N);
    for j = 2:N
        p_vec(j) = max((1+rho(j-1))/(1+rho(j)), (1-rho(j-1))/(1-rho(j)));
    end
    f_cap = min(4./(1+abs(rho)), sqrt(4*theta./(1+abs(rho))));
    psi = zeros(1,N);
    Apsi = zeros(1,N); Cpsi = zeros(1,N);

    % i=1
    Apsi(1) = 0;
    cands = f_cap(1);
    for m = 2:N
        cands(end+1) = f_cap(m) / prod(p_vec(2:m));
    end
    Cpsi(1) = min(cands);
    psi(1)  = c_vals(1)*(Cpsi(1) - Apsi(1));

    % i=2:N
    for j = 2:N
        Apsi(j) = psi(j-1) * p_vec(j);
        cands = [Apsi(j), f_cap(j)];
        for m = j+1:N
            cands(end+1) = f_cap(m) / prod(p_vec(j+1:m));
        end
        Cpsi(j) = min(cands);
        psi(j)  = Apsi(j) + c_vals(j)*(Cpsi(j)-Apsi(j));
    end
end

function w = total_variance(th, rh, ps, k)
    % Computes total variance w(k) at a given slice
    w = 0.5*( th + rh*ps.*k + ...
        sqrt((ps.*k + th*rh).^2 + th^2*(1-rh^2)) );
end

function residuals = eSSVI_obj_powerlaw(x, T_vectU, k_arr, marketVols)
    % Build parameters
    [theta, rho, psi] = deal_parameters(x, T_vectU, k_arr);

    % Compute slice‐wise residuals
    residuals = [];
    for i = 1:numel(T_vectU)
        w = total_variance(theta(i), rho(i), psi(i), k_arr{i});
        sigma_model = sqrt(max(w,0) ./ T_vectU(i));
        residuals   = [residuals; sigma_model - marketVols{i}(:)];
    end
end

