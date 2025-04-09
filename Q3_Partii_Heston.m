clear 
close all
clc
%%
%% 1. Load Data and Preprocess
dataTable = readtable('Matlab_df.xlsx');  % Load the data table
% Expected columns: maturity, strike, implied_volatility, forward, interest_rate
Maturity  = dataTable.maturity;         % in years
Strike    = dataTable.strike;           % strike price
IV_mid    = dataTable.implied_volatility; % MID implied volatility
Forward   = dataTable.forward;          % forward price
Interest  = dataTable.interest_rate;    % risk-free rate

% Ensure column vectors
Maturity = Maturity(:);
Strike   = Strike(:);
IV_mid   = IV_mid(:);
Forward  = Forward(:);
Interest = Interest(:);

% Compute spot price: S0 = F * exp(-r*T)
Spot = Forward .* exp(-Interest .* Maturity);

% Number of option quotes
nData = length(Maturity);
%%

function ret = CharFunc(params, type, u, dt)

    switch type
        case 10
            ret = HestonCF(params, u, dt);
        otherwise
            error('Unsupported type: %d', type);
    end
end

function output = HestonCF(xxx, u, T)
%HESTONCF Heston characteristic function (without jumps).
%   The function implements the equations from Schoutens et al. (2003) pg 4.

    % Unpack parameters
    mu=xxx(1); sigma2=xxx(2); k=xxx(3); neta=xxx(4); theta=xxx(5);  rho = xxx(6);

    % Compute the complex discriminant d
    d = sqrt((rho * theta .* u * 1i - k).^2 - theta^2 * (-1i .* u - u.^2));
    
    % Compute repeated term rep1 and auxiliary function g
    rep1 = k - rho * theta .* u * 1i - d;
    g = rep1 ./ (rep1 + 2 * d);

    % Calculate the components of the characteristic function
    tmp1 = exp(1i * u * mu .* T);
    tmp2 = exp(neta * k / theta^2 * (rep1 .* T - 2 * log((1 - g .* exp(-d .* T)) ./ (1 - g))));
    tmp3 = exp(sigma2 / theta^2 * rep1 .* (1 - exp(-d .* T)) ./ (1 - g .* exp(-d .* T)));

    % Combine components to obtain the output
    output = tmp1 .* tmp2 .* tmp3;
end

%%

% Density recovery
function [Call_COS, Put_COS]=VanillaEuro_COS(Spot,K,r,q,T,xxx,type,N,L)
% COS option pricing method - Fang and Oosterlee (2008)
 
% Bounds
if type == 10
    a =-7; b = 7;
end
 
% V_k computation for Call and Put 
Vk_call = (2/(b-a))*(chiFO(0,b,N,a,b)-psiFO(0,b,N,a,b)); % call
Vk_put = -(2/(b-a))*(chiFO(a,0,N,a,b)-psiFO(a,0,N,a,b));  % put
 
% Grids
k_in_vect = (0:N-1);
u= k_in_vect *pi/(b-a);
 
% create char fun and risk neutral adjustment
phi=CharFunc([0 xxx],type,u,T);  % calls for the CF of the driftless process
if type == 10
    omega = 0;
end
mu=r-q-omega;  % computes the risk neutral drift
CF_RN=exp(1i.*u.* log(Spot./K)).* exp(1i.*u.*mu.*T).*phi; % log(S./K)) is our 'x'
 
% F_k 
Exp_term=exp(-1i*pi*a*k_in_vect/(b-a));
Fk_x =  real(CF_RN  .* Exp_term);
 
% Option price computation
DF = exp(-r .* T);
Call_COS = DF.*K.*(sum(Fk_x.*Vk_call,2)-0.5.*Fk_x(:,1).*Vk_call(:,1));
Put_COS = DF.*K.*(sum(Fk_x.*Vk_put,2)-0.5.*Fk_x(:,1).*Vk_put(:,1));
end
 
% Fourier-cosine series coefficients of the terminal payoff function

%%

% Cosine expansion 
function ret=chiFO(c,d,N,a,b)
chi = (1./(1+((0:N-1)*pi/(b-a)).^2)).*(cos((0:N-1)*pi*(d-a)/(b-a))*exp(d)-...
    cos((0:N-1)*pi*(c-a)/(b-a))*exp(c)+((0:N-1)*pi/(b-a)).*sin((0:N-1)*pi*(d-a)/(b-a))*exp(d)-...
    ((0:N-1)*pi/(b-a)).*sin((0:N-1)*pi*(c-a)/(b-a))*exp(c));
ret=chi;
end
 
function ret=psiFO(c,d,N,a,b)
psi(1) = d-c;
psi(2:N) = (sin((1:N-1)*pi*(d-a)/(b-a))-sin((1:N-1)*pi*(c-a)/(b-a)))*(b-a)./((1:N-1)*pi);
ret=psi;
end
%% 2. Compute Market Prices using the Black Model
MarketCall = zeros(nData,1);
MarketPut  = zeros(nData,1);
for j = 1:nData
    F = Forward(j);
    K = Strike(j);
    T = Maturity(j);
    r = Interest(j);
    vol = IV_mid(j);
    [MarketCall(j), MarketPut(j)] = BlackPrice(F, K, T, r, vol);
end

%% 3. Set Up Calibration: Initial Guess & Bounds
% Parameter vector: [v0, kappa, sigma, rho, theta]
param0 = [0.0119, 0.2542, 0.8096, -0.0001, 0.9998];
%param0 = [0.04, 2, 0.5, -0.7, 0.04];
%lb     = [0.01, 0.01, 0.1, -0.999, 0.01];
%ub     = [0.05, 10,   2,   -0.01, 0.05];
lb     = [0.0001, 0.0001, 0.0001, -1, 0.0001]; 
ub     = [1, 20, 5, 0, 1];

% The calibration error will compare COS call prices to MarketCall prices.

%% 4. Run Calibration via lsqnonlin
q = 0.0 ;
type = 10;
L = 10;
N = 2^12;
% The anonymous function passes the current parameters along with all required data.
options = optimoptions('lsqnonlin', 'Display', 'iter', 'TolFun', 1e-8, 'TolX', 1e-8);
param_opt = lsqnonlin(@(p) computePriceErrors(p, Spot, Strike, Maturity, Interest, MarketCall, q, type, N, L), ...
                      param0, lb, ub, options);

fprintf('Optimized Heston Parameters:\n');
fprintf('Initial Variance (v0):         %.4f\n', param_opt(1));
fprintf('Mean-Reversion Speed (kappa):  %.4f\n', param_opt(2));
fprintf('Vol of Vol (sigma):            %.4f\n', param_opt(3));
fprintf('Correlation (rho):             %.4f\n', param_opt(4));
fprintf('Long-run Variance (theta):      %.4f\n', param_opt(5));

%% 5. Compare COS Prices with Market Prices Using Calibrated Parameters
COS_Call = zeros(nData,1);
COS_Put  = zeros(nData,1);
for i = 1:nData
    S0 = Spot(i);
    K  = Strike(i);
    T  = Maturity(i);
    r  = Interest(i);
    
    if T < 1e-8
        COS_Call(i) = max(S0 - K, 0);
        COS_Put(i)  = max(K - S0, 0);
    else
        [COS_Call(i), COS_Put(i)] = VanillaEuro_COS(S0, K, r, q, T, param_opt, type, N, L);
    end
end

% Print a comparison table:
fprintf('\nOption Index\tMarket Call\tCOS Call\tMarket Put\tCOS Put\n');
for i = 1:nData
    fprintf('%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', i, MarketCall(i), COS_Call(i), MarketPut(i), COS_Put(i));
end

% Plot comparisons:
figure;
subplot(2,1,1);
plot(MarketCall, 'bo-', 'LineWidth', 2, 'DisplayName', 'Market Call');
hold on;
plot(COS_Call, 'rx-', 'LineWidth', 2, 'DisplayName', 'COS Call');
xlabel('Option Index'); ylabel('Call Price');
legend('show'); title('Market vs COS Call Prices'); grid on;

subplot(2,1,2);
plot(MarketPut, 'bo-', 'LineWidth', 2, 'DisplayName', 'Market Put');
hold on;
plot(COS_Put, 'rx-', 'LineWidth', 2, 'DisplayName', 'COS Put');
xlabel('Option Index'); ylabel('Put Price');
legend('show'); title('Market vs COS Put Prices'); grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Black Price Function (returns both call and put prices)
function [call, put] = BlackPrice(F, K, T, r, vol)
    if T <= 0 || vol <= 0
        call = exp(-r*T)*max(F-K,0);
        put  = exp(-r*T)*max(K-F,0);
    else
        d1 = (log(F/K) + 0.5*vol^2*T) / (vol*sqrt(T));
        d2 = d1 - vol*sqrt(T);
        Nd1 = 0.5*(1+erf(d1/sqrt(2)));
        Nd2 = 0.5*(1+erf(d2/sqrt(2)));
        call = exp(-r*T) * (F*Nd1 - K*Nd2);
        Nmd1 = 1 - Nd1;
        Nmd2 = 1 - Nd2;
        put  = exp(-r*T) * (K*Nmd2 - F*Nmd1);
    end
end

% Compute Price Errors Helper Function for Calibration
% This function computes the difference between model call prices (from COS method)
% and the market call prices for each option in the input vectors.
function err = computePriceErrors(params, SpotArr, StrikeArr, TArr, rArr, MarketPrices, q, type, N, L)
    n = length(StrikeArr);
    ModelPrices = zeros(n,1);
    for idx = 1:n
        S0 = SpotArr(idx);
        K  = StrikeArr(idx);
        T  = TArr(idx);
        r  = rArr(idx);
        if T < 1e-8
            ModelPrices(idx) = max(S0-K, 0);
        else
            [callPrice, ~] = VanillaEuro_COS(S0, K, r, q, T, params, type, N, L);
            ModelPrices(idx) = callPrice;
        end
    end
    err = ModelPrices - MarketPrices;
end

%%
residuals_call = MarketCall - COS_Call;
residuals_put  = MarketPut - COS_Put;


rmse_call = sqrt(mean(residuals_call.^2));
rmse_put  = sqrt(mean(residuals_put.^2));

fprintf('\nCall RMSE: %.4f\n', rmse_call);
fprintf('Put RMSE:  %.4f\n', rmse_put);

figure;
plot(residuals_call, 'bo-', 'LineWidth', 2, 'DisplayName', 'Call Residuals');
hold on;
plot(residuals_put, 'rx-', 'LineWidth', 2, 'DisplayName', 'Put Residuals');
xlabel('Option Index');
ylabel('Residual (Market Price - COS Price)');
title('Residuals Plot for Calls and Puts');
legend('show');
grid on;