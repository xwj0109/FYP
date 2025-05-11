%% Heston-COS Calibration Script
% Fixed implementation following Fang & Oosterlee (2008) and dissertation
clear; close all; clc;

%% 1. Load Data and Preprocess
dataTable = readtable('Matlab_df.xlsx');  % Expect: maturity, strike, implied_volatility, forward, interest_rate
T_matur  = dataTable.maturity(:);
K_strike = dataTable.strike(:);
IV_mid    = dataTable.implied_volatility(:);
Fwd       = dataTable.forward(:);
riskFree  = dataTable.interest_rate(:);
q = 0.0;  % dividend yield (if any)

% Spot from forward
Spot = Fwd .* exp(-riskFree .* T_matur);
nData = numel(T_matur);

%% 2. Compute Market Prices via Black Model
MarketCall = zeros(nData,1);
MarketPut  = zeros(nData,1);
for j = 1:nData
    [MarketCall(j), MarketPut(j)] = BlackPrice(Fwd(j), K_strike(j), T_matur(j), riskFree(j), IV_mid(j));
end

%% 3. Calibration Setup: initial guess and bounds
% params = [kappa, theta, sigma, rho, v0]
param0 = [2, 0.04, 0.5, -0.7, 0.04];
lb     = [0.0001, 0.0001, 0.0001, -0.999, 0.0001];
ub     = [20,      1,      5,      0.999, 1];

% COS parameters
N = 2^12;  L = 10;
options = optimoptions('lsqnonlin','Display','iter','TolFun',1e-6,'TolX',1e-6);

param_opt = lsqnonlin(@(p) computePriceErrors(p, Spot, K_strike, T_matur, riskFree, MarketCall, q, N, L), ...
                      param0, lb, ub, options);

fprintf('Optimized Heston Parameters:\n');
fprintf('kappa: %.4f, theta: %.4f, sigma: %.4f, rho: %.4f, v0: %.4f\n', param_opt);

%% 4. Compare Model vs Market
COS_Call = zeros(nData,1);
COS_Put  = zeros(nData,1);
for i = 1:nData
    S0 = Spot(i);
    K  = K_strike(i);
    T  = T_matur(i);
    r  = riskFree(i);
    if T < 1e-8
        COS_Call(i) = max(S0-K,0);
        COS_Put(i)  = max(K-S0,0);
    else
        [COS_Call(i), COS_Put(i)] = VanillaEuro_COS(S0, K, r, q, T, param_opt, N, L);
    end
end

rmse_call = sqrt(mean((MarketCall - COS_Call).^2));
rmse_put  = sqrt(mean((MarketPut  - COS_Put ).^2));

fprintf('Call RMSE: %.4f\nPut RMSE:  %.4f\n', rmse_call, rmse_put);

%% === FUNCTION DEFINITIONS ===

function [call, put] = BlackPrice(F, K, T, r, vol)
    if T<=0 || vol<=0
        call = exp(-r*T)*max(F-K,0);
        put  = exp(-r*T)*max(K-F,0);
    else
        d1 = (log(F/K)+0.5*vol^2*T)/(vol*sqrt(T));
        d2 = d1 - vol*sqrt(T);
        Nd1 = 0.5*(1+erf(d1/sqrt(2)));
        Nd2 = 0.5*(1+erf(d2/sqrt(2)));
        call = exp(-r*T)*(F*Nd1 - K*Nd2);
        put  = exp(-r*T)*(K*(1-Nd2) - F*(1-Nd1));
    end
end

function err = computePriceErrors(params, SpotArr, Karr, Tarr, rArr, MarketC, q, N, L)
    n = numel(Karr);
    ModelC = zeros(n,1);
    for idx=1:n
        if Tarr(idx)<1e-8
            ModelC(idx) = max(SpotArr(idx)-Karr(idx),0);
        else
            [ModelC(idx), ~] = VanillaEuro_COS(SpotArr(idx), Karr(idx), rArr(idx), q, Tarr(idx), params, N, L);
        end
    end
    err = ModelC - MarketC;
end

function [Call, Put] = VanillaEuro_COS(S0, K, r, q, T, params, N, L)
    % 1. cumulants of log(S_T/K)
    [c1, c2, c4] = HestonCumulants(params, r-q, T);
    a = c1 - L*sqrt(c2 + sqrt(c4));
    b = c1 + L*sqrt(c2 + sqrt(c4));
    k = (0:N-1)';
    u = k*pi/(b-a);

    % 2. CF
    phi = HestonCF(params, u, T);
    CF_RN = exp(1i*u*(log(S0/K)+(r-q)*T)) .* phi;

    % 3. payoff expansion
    Vk_call = 2/(b-a)*(chiFO(0,b,N,a,b) - psiFO(0,b,N,a,b));
    Vk_put  = -2/(b-a)*(chiFO(a,0,N,a,b) - psiFO(a,0,N,a,b));

    % 4. Fourier coefficients
    ExpA = exp(-1i*pi*a*k/(b-a));
    Fk = real(CF_RN .* ExpA);

    % 5. assemble
    DF = exp(-r*T);
    Call = DF*K*(sum(Fk.*Vk_call) - 0.5*Fk(1)*Vk_call(1));
    Put  = DF*K*(sum(Fk.*Vk_put ) - 0.5*Fk(1)*Vk_put (1));
end

function phi = HestonCF(params, u, T)
    kappa = params(1); theta = params(2);
    sigma  = params(3); rho   = params(4);
    v0     = params(5);
    b = kappa - rho*sigma*1i*u;
    d = sqrt(b.^2 + sigma^2*(u.^2 + 1i*u));
    g = (b - d)./(b + d);
    C = kappa*theta/sigma^2*((b-d)*T - 2*log((1 - g.*exp(-d*T))./(1-g)));
    D = (b-d)./sigma^2.*(1 - exp(-d*T))./(1-g.*exp(-d*T));
    phi = exp(C + D*v0);
end

function [c1, c2, c4] = HestonCumulants(params, m, T)
    % m = drift = r - q
    kappa=params(1); theta=params(2); sigma=params(3); rho=params(4); v0=params(5);
    exp_kT = exp(-kappa*T);
    c1 = m*T + (theta - v0)/(2*kappa)*(1-exp_kT) - 0.5*theta*T;
    c2 = 1/(8*kappa^3)* (sigma*T*kappa*exp_kT.*(v0-theta)*(8*kappa*rho - 4*sigma) +...
         2*theta*kappa*T*(-4*kappa*rho*sigma+sigma^2+4*kappa^2) + sigma^2*( (theta-v0)*(1-exp_kT)*8*rho*kappa -5*theta*exp_kT + theta*exp(2*kappa*T) +3*theta ) );
    c4 = (2*sigma^4*exp(-2*kappa*T)/(kappa^4))*(1-exp_kT).^2;
end

function ret = chiFO(c,d,N,a,b)
    k = (0:N-1)';
    alpha = k*pi/(b-a);
    term1 = cos(alpha*(d-a)).*exp(d) - cos(alpha*(c-a)).*exp(c);
    term2 = alpha.*(sin(alpha*(d-a)).*exp(d) - sin(alpha*(c-a)).*exp(c));
    ret = (1./(1+alpha.^2)).*(term1 + term2);
end

function ret = psiFO(c,d,N,a,b)
    k = (0:N-1)';
    ret = zeros(N,1);
    ret(1) = d-c;
    ret(2:end) = (sin(k(2:end)*pi*(d-a)/(b-a)) - sin(k(2:end)*pi*(c-a)/(b-a))).*(b-a)./(k(2:end)*pi);
end

%% 5. Print Market vs Model Prices
fprintf('\nIndex\tMarketCall\tModelCall\tMarketPut\tModelPut\n');
for i = 1:nData
    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', ...
        i, MarketCall(i), COS_Call(i), MarketPut(i), COS_Put(i));
end

%% 6. Residuals and RMSE Plotting
% compute residuals
res_call = MarketCall - COS_Call;
res_put  = MarketPut  - COS_Put;

% a) residuals over strikes
figure;
subplot(2,1,1);
plot(1:nData, res_call, 'bo-', 'LineWidth', 1.5);
xlabel('Option Index'); ylabel('Call Residual');
title('Call Residuals (Market – Model)');
grid on;

subplot(2,1,2);
plot(1:nData, res_put, 'rs-', 'LineWidth', 1.5);
xlabel('Option Index'); ylabel('Put Residual');
title('Put Residuals (Market – Model)');
grid on;

% b) bar chart of overall RMSE
figure;
bar([rmse_call, rmse_put], 'FaceAlpha',0.7);
set(gca, 'XTickLabel', {'Call RMSE','Put RMSE'}, 'FontSize',12);
ylabel('RMSE');
title('Overall RMSE for Heston‐COS Calibration');
grid on;