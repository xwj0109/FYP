%% Global eSSVI Calibration Integrated with Existing SVI Code
% This script revises SSVI calibration to use the global eSSVI 
% parametrization as described in essvi_paper.pdf. The goal is to calibrate 
% the implied volatility surface in a way that rules out calendar spread arbitrages.

%% 1. Clear workspace, close figures, and clear command window
clear
close all
clc

%% 2. Load data from Excel
% Assumes the first sheet contains the data with headers.
DataSPX = readtable('Matlab_df.xlsx', 'Sheet', 1);

% (Optional) Remove any rows with missing values
% DataSPX = rmmissing(DataSPX);

% Extract variables from the table
forward      = DataSPX.forward;
maturity     = DataSPX.maturity;
strike       = DataSPX.strike;
implied_vol  = DataSPX.implied_volatility;
log_moneyness= DataSPX.log_moneyness;

%% 3. Organize market data by unique maturities
T_vectU = unique(maturity);
N = length(T_vectU);

% Preallocate cell arrays for strikes, forwards, log-moneyness, and market vols per slice.
K_cell      = cell(1, N);
F0_cell     = cell(1, N);
k_arr       = cell(1, N);
marketVols  = cell(1, N);

for i = 1:N
    pos = (maturity == T_vectU(i));
    % For each maturity, use all available strikes
    K_cell{i} = strike(pos);
    % For forward price, if constant per maturity, take first value
    F0_cell{i} = forward(find(pos, 1, 'first'));
    marketVols{i} = implied_vol(pos);
    % Compute log-forward moneyness: k = log(K / F0)
    k_arr{i} = log( K_cell{i} / F0_cell{i} );
end

%% 4. Set initial guesses for global eSSVI parameters
% In the screenshot/paper excerpt, the suggested initial conditions are:
%   - The 'a' (alpha) parameters come from a guess of total implied variances
%     at the ATM (i.e. from sigma_atm^2 * T).
%   - The rho (sometimes called 'p') are set to 0.
%   - The c parameters are set to 0.5.
%
% Additionally, we can impose a small bounding logic on the 'a' if the
% maximum initial guess is very small (< 0.05) or we might enlarge the
% upper bound if the guess is larger, etc. The exact bounding rule can vary.

% 4.1 Initialize rho = 0 for each slice
init_rho = zeros(1, N);   % each maturity-slice correlation param

% 4.2 Guess total variance (theta_guess) from ATM implied volatility
theta_guess = zeros(1, N);
for i = 1:N
    % Find ATM index (closest to k=0)
    [~, atm_idx] = min(abs(k_arr{i}));
    sigma_atm = marketVols{i}(atm_idx);
    % Approx total variance = sigma_atm^2 * T
    theta_guess(i) = (sigma_atm)^2 * T_vectU(i);
end

% Enforce monotonicity (if desired)
for i = 2:N
    if theta_guess(i) < theta_guess(i-1)
        theta_guess(i) = 1.05 * theta_guess(i-1);
    end
end

% 4.3 The first theta becomes init_theta1
init_theta1 = max(theta_guess(1), 1e-12);

% 4.4 The 'a' parameters (alpha) are the increments in total variance
%     from slice (i-1) to slice i. We keep them strictly positive.
init_a = zeros(1, N-1);
for i = 2:N
    diffVal = theta_guess(i) - theta_guess(i-1);
    init_a(i-1) = max(diffVal, 1e-6);
end

% Optionally, if the maximum of init_a is < 0.05, we might tighten or relax
% the bounds. For example:
maxA = max(init_a);
if maxA < 0.05
    % For small initial 'a', you might:
    %   - artificially increase them
    %   - or keep them small but enforce a strict upper bound
    % Example: double them or set them to 0.05
    init_a = maxA * 2 * (init_a / maxA);  % doubling
    % or: init_a = max(init_a, 0.05);
end

% 4.5 The initial c's are set to 0.5
init_c = 0.5 * ones(1, N);

% 4.6 Construct the initial parameter vector
% paramVec = [ rho1,...,rhoN, theta1, a2...aN, c1...cN ]
X0 = [ init_rho, init_theta1, init_a, init_c ];

% 4.7 Parameter bounds (adapt to taste)
LB_rho    = -0.999 * ones(1, N);
UB_rho    =  0.999 * ones(1, N);

LB_theta1 = 1e-12;   % keep it strictly positive
UB_theta1 = Inf;

LB_a      = zeros(1, N-1);
UB_a      = Inf(1, N-1);
% If desired, refine the upper bound on a, e.g. if max(init_a) < 0.05 then you
% might set a smaller upper bound, or if it's large, you might raise it:

% Example bounding rule:
if maxA < 0.05
    UB_a(:) = 0.05;  % or 0.1, your preference
else
    % or if maxA is large, you could set UB_a to something like 2*maxA
    % UB_a(:) = 2*maxA;
    % For now, keep it infinite if you don't want to impose extra constraints
end

LB_c      = zeros(1, N);
UB_c      = 0.9999 * ones(1, N);

LB = [LB_rho, LB_theta1, LB_a, LB_c];
UB = [UB_rho, UB_theta1, UB_a, UB_c];


%% 6. Run the calibration optimization using lsqnonlin
% Pass the required data to the objective function via an anonymous function.
opts = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...
    'MaxFunctionEvaluations', 1e7, ...  % Increase if necessary
    'MaxIterations', 1e6, ...           % Increase if necessary
    'FunctionTolerance', 1e-5, ...
    'StepTolerance', 1e-5, ...
    'FiniteDifferenceType', 'central', ...   % More accurate finite differences
    'UseParallel', true, ...                 % Utilize multiple cores
    'PlotFcn', {@optimplotx, @optimplotfval});% Monitor progress

% opts = optimoptions(opts, 'SpecifyObjectiveGradient', true);

objective = @(x) eSSVI_obj(x, T_vectU, k_arr, marketVols);
[param_opt, resnorm, residuals] = lsqnonlin(objective, X0, LB, UB, opts);

%% 7. Retrieve calibrated parameters and compute model fits
opt_rho   = param_opt(1:N);
opt_theta1 = param_opt(N+1);
opt_a      = param_opt(N+2 : N+1+(N-1));
opt_c      = param_opt(N+1+(N-1)+1 : end);

% Reconstruct theta and psi from calibrated parameters
theta_cal = zeros(1, N);
psi_cal   = zeros(1, N);
p = ones(1, N);
theta_cal(1) = opt_theta1;
for i = 2:N
    p(i) = max((1 + opt_rho(i-1)) / (1 + opt_rho(i)), (1 - opt_rho(i-1)) / (1 - opt_rho(i)));
    theta_cal(i) = theta_cal(i-1) * p(i) + opt_a(i-1);
end

% Compute f_cap for each maturity using the Fukasawa conditions:
f_cap = zeros(1, N);
for i = 1:N
    f_bound_static = 4 / (1 + abs(opt_rho(i)));
    FG_j = (4 * theta_cal(i)) / (1 + abs(opt_rho(i)));
    f_bound_dyn = sqrt(FG_j);
    %f_bound_dyn = 2 * sqrt(theta_cal(i) / T_vectU(i));  % T_vectU(i) is the maturity T for slice i
    f_cap(i) = min(f_bound_static, f_bound_dyn);
end

% Reconstruct psi for each slice using f_cap:
psi_cal = zeros(1, N);
Apsi = zeros(1, N);
Cpsi = zeros(1, N);
Apsi(1) = 0;
Cpsi(1) = f_cap(1);
if N > 1
    prod_p = 1;
    for j = 2:N
        prod_p = prod_p * p(j); % correct
        Cpsi(1) = min(Cpsi(1), f_cap(j) / prod_p); % correct
    end
end
psi_cal(1) = opt_c(1) * (Cpsi(1) - Apsi(1)) + Apsi(1);
for i = 2:N
    Apsi(i) = psi_cal(i-1) * p(i);
    Cpsi(i) = f_cap(i);
    if i < N
        prod_p = 1;
        for j = i+1:N
            prod_p = prod_p * p(j);
            Cpsi(i) = min(Cpsi(i), f_cap(j) / prod_p); % i think mistake is here 
            %Cpsi(i) = min( ((Apsi(i)/p(i))/theta_cal(i-1)) * theta_cal(i), f_cap(i), f_cap(i+1)/p(i+1)); % testing
        end
    end
    Cpsi(i) = min(Cpsi(i), psi_cal(i-1) * (theta_cal(i-1) / theta_cal(i)));
    psi_cal(i) = opt_c(i) * (Cpsi(i) - Apsi(i)) + Apsi(i);
    psi_cal(i) = max(Apsi(i) + 1e-12, min(psi_cal(i), Cpsi(i) - 1e-12));
end

% Recompute model implied volatilities for plotting using calibrated parameters
modelVols_fit = cell(1, N);
for i = 1:N
    k_vec = k_arr{i};
    w = 0.5 * ( theta_cal(i) + opt_rho(i) * psi_cal(i) * k_vec + ...
        sqrt( (psi_cal(i) * k_vec + theta_cal(i) * opt_rho(i)).^2 + (theta_cal(i)^2) * (1 - opt_rho(i)^2) ) );
    sigma_model = sqrt( max(w,0) ./ T_vectU(i) );
    modelVols_fit{i} = sigma_model;
end


%% 8. Plot the fitted eSSVI volatility smiles versus market data
figure; hold on;
colors = lines(N);
for i = 1:N
    % Recover actual strikes from log-moneyness: K = exp(k)*F0
    K_plot = exp(k_arr{i}) .* F0_cell{i};
    plot(K_plot, modelVols_fit{i}*100, '--', 'Color', colors(i,:), 'LineWidth', 1.5);
    plot(K_plot, marketVols{i}*100, 'o', 'Color', colors(i,:), 'MarkerSize', 4, ...
         'DisplayName', sprintf('T = %.2f', T_vectU(i)));
end
xlabel('Strike K');
ylabel('Implied Volatility (%)');
title('Global eSSVI Calibrated Volatility Smiles');
legend('Model fit','Market data');
grid on;
hold off;

%% 9. (Optional) 3D Surface Plot of Calibrated eSSVI Volatility Surface with Maturity Interpolation
% Create a fine grid for strikes and maturities.
allK = linspace(min(cellfun(@min, K_cell)), max(cellfun(@max, K_cell)), 50);
allT = linspace(min(T_vectU), max(T_vectU), 50);
[TT, KK] = meshgrid(allT, allK);
surf_vol = zeros(size(TT));

% Loop over every grid point.
for ii = 1:numel(TT)
    T_current = TT(ii);
    
    % Find the two calibrated maturities that bracket T_current.
    if T_current <= T_vectU(1)
        % Use the first slice if T_current is below the calibrated range.
        i_low = 1;
        i_high = 1;
        lambda = 0;
    elseif T_current >= T_vectU(end)
        % Use the last slice if T_current exceeds the calibrated range.
        i_low = length(T_vectU);
        i_high = length(T_vectU);
        lambda = 0;
    else
        % Find indices: T_vectU(i_low) <= T_current <= T_vectU(i_high)
        i_high = find(T_vectU >= T_current, 1, 'first');
        i_low = i_high - 1;
        lambda = (T_current - T_vectU(i_low)) / (T_vectU(i_high) - T_vectU(i_low));
    end
    
    % Interpolate the calibrated parameters using convex combination
    theta_interp = (1 - lambda) * theta_cal(i_low) + lambda * theta_cal(i_high);
    rho_interp   = (1 - lambda) * opt_rho(i_low)   + lambda * opt_rho(i_high);
    psi_interp   = (1 - lambda) * psi_cal(i_low)   + lambda * psi_cal(i_high);
    
    % Interpolate the forward price F0 (if available) similarly:
    if i_low == i_high
        F0_interp = F0_cell{i_low};
    else
        F0_interp = (1 - lambda) * F0_cell{i_low} + lambda * F0_cell{i_high};
    end
    
    % Compute log-moneyness using the interpolated forward price.
    k_val = log(KK(ii) / F0_interp);
    
    % Compute the eSSVI total variance at this grid point.
    % The formula: w(k) = 0.5*(theta + rho*psi*k + sqrt((psi*k + theta*rho)^2 + theta^2*(1 - rho^2)))
    w_val = 0.5 * ( theta_interp + rho_interp * psi_interp * k_val + ...
                sqrt( (psi_interp * k_val + theta_interp * rho_interp).^2 + (theta_interp^2) * (1 - rho_interp^2) ) );
    
    % Compute the model-implied volatility: sigma = sqrt(w/T)
    surf_vol(ii) = sqrt(max(w_val, 0) / T_current);
end

% Plot the 3D surface (implied volatility in percent).
figure;
surf(TT, KK, surf_vol, 'EdgeColor', 'none');
xlabel('Maturity T'); ylabel('Strike K'); zlabel('Implied Vol (%)');
title('Calibrated Global eSSVI Volatility Surface (with Interpolation)');
set(gca, 'YDir', 'reverse');
shading interp; view(135, 30); grid on;


%% 10. (Optional) Export calibrated implied volatilities and parameters to CSV
% T_out = table(modelVols_fit(:), 'VariableNames', {'model_implied_volatility'});
% writetable(T_out, 'implied_volatilities_eSSVI.csv');

%% End of Script

%% Local Function: eSSVI Objective Function
function [residuals, modelVols_fit] = eSSVI_obj(paramVec, T_vectU, k_arr, marketVols)
    % This function computes the residuals between model and market volatilities.
    % Inputs:
    %   paramVec   - parameter vector: [rho1,...,rhoN, theta1, a2,...,aN, c1,...,cN]
    %   T_vectU    - vector of unique maturities
    %   k_arr      - cell array of log-forward moneyness vectors
    %   marketVols - cell array of market implied volatilities per maturity
    
    N = length(T_vectU);
    
    % Parse parameters
    rho    = paramVec(1:N);
    theta1 = paramVec(N+1);
    a_vals = paramVec(N+2 : N+1+(N-1));  % a2,...,aN
    c_vals = paramVec(N+1+(N-1)+1 : end);  % c1,...,cN
    
    % Reconstruct theta for each maturity
    theta = zeros(1, N);
    theta(1) = max(theta1, 1e-12);
    p = ones(1, N); % p(1) is set to 1 for convenience.
    for i = 2:N
        p(i) = max((1 + rho(i-1)) / (1 + rho(i)), (1 - rho(i-1)) / (1 - rho(i)));
        theta(i) = theta(i-1) * p(i) + max(a_vals(i-1), 1e-6);
    end
    
    % Compute no-arbitrage bounds for psi for each maturity:
    f_bound_static = zeros(1, N);
    f_bound_dyn    = zeros(1, N);
    FG_j = zeros(1,N);
    f_cap = zeros(1, N);
    for i = 1:N
        f_bound_static(i) = 4 / (1 + (abs(rho(i))));
        FG_j(i) = (4 * theta(i)) /(1 + abs(rho(i)));
        %FG_j(i) = (4 * theta(i) * (1 + (abs(rho(i)))^2)) / ((1 + sqrt(1 - (abs(rho(i)))^2))^2);
        %f_bound_dyn(i) = 2 * sqrt(theta(i) / T_vectU(i)); % dynamic bound: 2*sqrt(theta/T)
        f_bound_dyn(i) = sqrt(FG_j(i));
        f_cap(i) = min(f_bound_static(i), f_bound_dyn(i));
    end
    
    % Reconstruct psi for each maturity:
    psi = zeros(1, N);
    Apsi = zeros(1, N);
    Cpsi = zeros(1, N);
    
    % For maturity 1:
    Apsi(1) = 0;
    Cpsi(1) = f_cap(1);
    if N > 1
        prod_p = 1;
        for j = 2:N
            prod_p = prod_p * p(j);
            Cpsi(1) = min(Cpsi(1), f_cap(j) / prod_p);
        end
    end
    psi(1) = Apsi(1) + c_vals(1) * (Cpsi(1) - Apsi(1));
    
    % For maturities 2 to N:
    for i = 2:N
        Apsi(i) = psi(i-1) * p(i);
        Cpsi(i) = f_cap(i);
        if i < N
            prod_p = 1;
            for j = i+1:N
                prod_p = prod_p * p(j);
                Cpsi(i) = min(Cpsi(i), f_cap(j) / prod_p);
            end
        end
        Cpsi(i) = min(Cpsi(i), psi(i-1) * (theta(i-1) / theta(i)));
        psi(i) = Apsi(i) + c_vals(i) * (Cpsi(i) - Apsi(i));
        psi(i) = max(Apsi(i) + 1e-12, min(psi(i), Cpsi(i) - 1e-12));
    end
    
    % Compute model implied volatilities and residuals:
    residuals = [];
    modelVols_fit = cell(1, N);
    for i = 1:N
        k_vec = k_arr{i};
        % eSSVI total variance:
        % w(k) = 0.5 * ( theta + rho*psi*k + sqrt((psi*k + theta*rho)^2 + (theta^2)*(1 - rho^2)) )
        w = 0.5 * ( theta(i) + rho(i) * psi(i) * k_vec + ...
            sqrt( (psi(i) * k_vec + theta(i) * rho(i)).^2 + (theta(i)^2) * (1 - rho(i)^2) ) );
        sigma_model = sqrt( max(w, 0) ./ T_vectU(i) ); 
        modelVols_fit{i} = sigma_model;
        resid_slice = sigma_model - marketVols{i};
        residuals = [residuals; resid_slice(:)];  %#ok<AGROW>
    end
end

