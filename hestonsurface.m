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

%% 5. Invert Model Call Prices to Implied Vols
HestonIV = nan(nData,1);

for i = 1:nData
    S0 = Spot(i);
    K  = K_strike(i);
    r  = riskFree(i);
    T  = T_matur(i);
    price = COS_Call(i);
    
    if T > 0 && price > max(S0-K,0)  % only invert realistic calls
        % 'Yield' sets the continuous dividend q;
        % you can also pass 'Class','call' explicitly if you like
        HestonIV(i) = blsimpv(S0, K, r, T, price, 'Yield', q);
    else
        HestonIV(i) = NaN; 
    end
end

% Quick check
fprintf('\nIndex\tModelPrice\tHestonIV (%%)\n');
for i=1:nData
    fprintf('%3d\t%10.4f\t%10.2f\n', i, COS_Call(i), 100*HestonIV(i));
end


%% 3. Organize data by unique maturities
T_vectU = unique(T_matur);
N       = numel(T_vectU);

K_cell     = cell(1,N);
F0_cell    = cell(1,N);
k_arr      = cell(1,N);
marketVols = cell(1,N);

for i = 1:N
    pos          = T_matur==T_vectU(i);
    K_cell{i}    = K_strike(pos);
    F0_cell{i}   = Fwd(find(pos,1,'first'));
    marketVols{i}= HestonIV(pos);
    k_arr{i}     = log(K_cell{i}/F0_cell{i});
end

%% 4. Initial guesses and bounds
% 4.1 rho0, alpha
init_rho0  = 0;
init_alpha = 0.5;

% 4.2 ATM total-variance guesses θ_i
theta_guess = zeros(1,N);
for i = 1:N
    [~,atm]       = min(abs(k_arr{i}));
    theta_guess(i) = marketVols{i}(atm)^2 * T_vectU(i);
end
% enforce monotonicity in T
for i = 2:N
    theta_guess(i) = max(theta_guess(i), 1.05*theta_guess(i-1));
end

init_theta1 = max(theta_guess(1),1e-12);
init_a      = max(diff(theta_guess),1e-6);
init_c      = 0.5*ones(1,N);

X0  = [init_rho0, init_alpha, init_theta1, init_a, init_c];
LB  = [-0.999, 0, 1e-12, zeros(1,N-1), zeros(1,N)];
UB  = [ 0.999, 5, Inf,       Inf(1,N-1), 0.9999*ones(1,N)];

%% 5. Global calibration via lsqnonlin
opts = optimoptions('lsqnonlin', ...
    'Display','iter','UseParallel',true, ...
    'MaxFunctionEvaluations',1e6,'MaxIterations',1e4, ...
    'FunctionTolerance',1e-8,'StepTolerance',1e-6);

objective = @(x) eSSVI_obj_powerlaw(x, T_vectU, k_arr, marketVols);
[param_opt,~,residuals] = lsqnonlin(objective, X0, LB, UB, opts);

%% 6. Extract calibrated parameters
rho0_opt   = param_opt(1);
alpha_opt  = param_opt(2);
theta1_opt = param_opt(3);
a_opt      = param_opt(4:3+N-1);
c_opt      = param_opt(3+N:end);

% Reconstruct ρ_i and θ_i
rho_vec   = rho0_opt * T_vectU.^(-alpha_opt);
theta_cal = zeros(1,N);
theta_cal(1) = theta1_opt;
for i = 2:N
    p_fac        = max((1+rho_vec(i-1))/(1+rho_vec(i)), (1-rho_vec(i-1))/(1-rho_vec(i)));
    theta_cal(i) = theta_cal(i-1)*p_fac + a_opt(i-1);
end

% --- GLOBAL ψ_i CONSTRUCTION ---
% 1) build p-vector
p = ones(1,N);
for i = 2:N
    p(i) = max((1+rho_vec(i-1))/(1+rho_vec(i)), (1-rho_vec(i-1))/(1-rho_vec(i)));
end

% 2) butterfly caps f_i
f_cap = min(4./(1+abs(rho_vec)), sqrt(4*theta_cal./(1+abs(rho_vec))));

% 3) build ψ_i globally
psi_cal = zeros(1,N);
Apsi    = zeros(1,N);
Cpsi    = zeros(1,N);

% i = 1
Apsi(1) = 0;
cands   = f_cap(1);
for j = 2:N
    cands(end+1) = f_cap(j) / prod(p(2:j));
end
Cpsi(1)    = min(cands);
psi_cal(1) = c_opt(1)*(Cpsi(1) - Apsi(1));

% i = 2:N
for i = 2:N
    Apsi(i)    = psi_cal(i-1) * p(i);
    cands      = [Apsi(i), f_cap(i)];
    for j = i+1:N
        cands(end+1) = f_cap(j) / prod(p(i+1:j));
    end
    Cpsi(i)    = min(cands);
    psi_cal(i) = Apsi(i) + c_opt(i)*(Cpsi(i) - Apsi(i));
end

%% 7. Compute model vols per slice
modelVols_fit = cell(1,N);
for i = 1:N
    k = k_arr{i};
    w = 0.5*( theta_cal(i) + rho_vec(i)*psi_cal(i).*k + ...
        sqrt((psi_cal(i).*k + theta_cal(i)*rho_vec(i)).^2 + theta_cal(i)^2*(1-rho_vec(i)^2)) );
    modelVols_fit{i} = sqrt(max(w,0)/T_vectU(i));
end

%% 8. Plot 1D slices
figure; hold on; colors=lines(N);
for i = 1:N
    Kp = exp(k_arr{i}) * F0_cell{i};
    plot(Kp, modelVols_fit{i}*100,'--','Color',colors(i,:));
    plot(Kp, marketVols{i}*100,'o','Color',colors(i,:),'DisplayName',sprintf('T=%.2f',T_vectU(i)));
end
xlabel('K_strike K'); ylabel('Implied Vol (%)');
title('Global eSSVI w/ Power-Law \rho(\tau)'); legend('Model','Market');
grid on; hold off;

%% 9. 3D Surface Plot of Calibrated eSSVI Vol Surface
allT = linspace(min(T_vectU), max(T_vectU), 50);
allK = linspace(min(cellfun(@min, K_cell)), max(cellfun(@max, K_cell)), 50);
[TT, KK] = meshgrid(allT, allK);

surf_vol = zeros(size(TT));
for idx = 1:numel(TT)
    T_cur = TT(idx);
    if T_cur <= T_vectU(1)
        i_low = 1; i_high = 1; lambda = 0;
    elseif T_cur >= T_vectU(end)
        i_low = N; i_high = N; lambda = 0;
    else
        i_high = find(T_vectU >= T_cur,1,'first');
        i_low  = i_high - 1;
        lambda = (T_cur - T_vectU(i_low)) / (T_vectU(i_high)-T_vectU(i_low));
    end

    th = (1-lambda)*theta_cal(i_low) + lambda*theta_cal(i_high);
    rh = (1-lambda)*rho_vec(i_low)   + lambda*rho_vec(i_high);
    ps = (1-lambda)*psi_cal(i_low)   + lambda*psi_cal(i_high);

    k = log(KK(idx) / ((1-lambda)*F0_cell{i_low} + lambda*F0_cell{i_high}));
    w = 0.5*( th + rh*ps*k + sqrt((ps*k + th*rh)^2 + th^2*(1 - rh^2)) );
    surf_vol(idx) = sqrt(max(w,0)/T_cur);
end

figure('Name','eSSVI Vol Surface','Units','normalized','Position',[0.2 0.1 0.6 0.7]);
h = surf(TT, KK, surf_vol, surf_vol, 'FaceColor','flat','EdgeColor',[0 0 0],'LineWidth',0.5);
colormap(parula); colorbar;
xlabel('Maturity'); ylabel('Strike'); zlabel('Implied Volatility');
title('Calibrated Heston implied volatility surface'); view(135,30); grid on;


%% Local Objective: eSSVI_obj_powerlaw
function residuals = eSSVI_obj_powerlaw(x, T_vectU, k_arr, marketVols)
    N      = numel(T_vectU);
    rho0   = x(1);
    alpha  = x(2);
    theta1 = x(3);
    a_vals = x(4:3+N-1);
    c_vals = x(3+N:end);

    % reconstruct rho and theta
    rho   = rho0 * T_vectU.^(-alpha);
    theta = zeros(1,N); theta(1)=theta1;
    for i = 2:N
        p_fac     = max((1+rho(i-1))/(1+rho(i)), (1-rho(i-1))/(1-rho(i)));
        theta(i) = theta(i-1)*p_fac + max(a_vals(i-1),1e-12);
    end

    % --- GLOBAL ψ_i CONSTRUCTION INSIDE OBJ ---
    p      = ones(1,N);
    for i = 2:N
        p(i) = max((1+rho(i-1))/(1+rho(i)), (1-rho(i-1))/(1-rho(i)));
    end
    f_cap = min(4./(1+abs(rho)), sqrt(4*theta./(1+abs(rho))));

    psi  = zeros(1,N);
    Apsi = zeros(1,N);
    Cpsi = zeros(1,N);

    % slice 1
    Apsi(1) = 0;
    cands   = f_cap(1);
    for j = 2:N
        cands(end+1) = f_cap(j) / prod(p(2:j));
    end
    Cpsi(1) = min(cands);
    psi(1)  = c_vals(1)*(Cpsi(1)-Apsi(1));

    % slices 2..N
    for i = 2:N
        Apsi(i)   = psi(i-1)*p(i);
        cands     = [Apsi(i), f_cap(i)];
        for j = i+1:N
            cands(end+1) = f_cap(j) / prod(p(i+1:j));
        end
        Cpsi(i)   = min(cands);
        psi(i)    = Apsi(i) + c_vals(i)*(Cpsi(i)-Apsi(i));
    end

    % build residuals
    residuals = [];
    for i = 1:N
        k = k_arr{i};
        w = 0.5*( theta(i) + rho(i)*psi(i).*k + ...
            sqrt((psi(i).*k + theta(i)*rho(i)).^2 + theta(i)^2*(1-rho(i)^2)) );
        sigma_model = sqrt(max(w,0)/T_vectU(i));
        residuals   = [residuals; sigma_model - marketVols{i}(:)];
    end
end

%%
% assume you already have:
%   allT (1×M vector), allK (1×L vector), and surf_vol (L×M matrix)
% where surf_vol(j,i) = implied vol at (K=allK(j), T=allT(i))

fprintf('%8s |  R_slope  R_ratio   L_slope  L_ratio\n','T');
fprintf('---------------------------------------------\n');

for i = 1:length(allT)
    Tcur = allT(i);
    
    % get total variance w = vol^2 * T
    wcol = surf_vol(:,i).^2 * Tcur;
    
    % corresponding k-grid
    kcol = log(allK(:)/F0_cell{1});  
    % (if F0 varies with tenor, you can build a matrix of F0's similarly)
    
    % pick the two largest k's, and two smallest
    [k_sorted, ix] = sort(kcol);
    w_sorted       = wcol(ix);
    
    kL2 = k_sorted(1);
    kL1 = k_sorted(2);
    kR1 = k_sorted(end-1);
    kR2 = k_sorted(end);
    wL2 = w_sorted(1);
    wL1 = w_sorted(2);
    wR1 = w_sorted(end-1);
    wR2 = w_sorted(end);
    
    % finite‐difference slopes
    slopeR = (wR2 - wR1)/(kR2 - kR1);
    slopeL = (wL1 - wL2)/(kL1 - kL2);
    
    % ratios
    ratioR = wR2/kR2;
    ratioL = wL2/abs(kL2);
    
    fprintf('%8.4f | %8.4f %8.4f %8.4f %8.4f\n', ...
        Tcur, slopeR, ratioR, slopeL, ratioL);
end

