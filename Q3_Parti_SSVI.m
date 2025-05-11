%% Clear workspace, close figures, and clear command window
clear
close all
clc

%% Load data from Excel
% Loading data from an Excel file. It assumes the first sheet contains the data and the first row has header names.
DataSPX = readtable('Matlab_df.xlsx', 'Sheet', 1);

% Extract variables from the table
ask                     = DataSPX.ask;
bid                     = DataSPX.bid;
forward                 = DataSPX.forward;
interest_rate           = DataSPX.interest_rate;
maturity                = DataSPX.maturity;
strike                  = DataSPX.strike;
CP_vect                 = DataSPX.CP_vect;
mid                     = DataSPX.mid;
implied_volatility      = DataSPX.implied_volatility;
implied_volatility_bid  = DataSPX.implied_volatility_bid;
implied_volatility_ask  = DataSPX.implied_volatility_ask;
log_moneyness           = DataSPX.log_moneyness;
moneyness               = DataSPX.moneyness;

% Unique maturities in the dataset
T_vectU = unique(maturity);

%% Calibrate SSVI using the SVI library functions
phifun = 'power_law';

% Calibrate for mid, bid, and ask volatilities using the SVI calibration function
parameters      = fit_svi_surface(implied_volatility, maturity, log_moneyness, phifun);
parameters_bid  = fit_svi_surface(implied_volatility_bid, maturity, log_moneyness, phifun);
parameters_ask  = fit_svi_surface(implied_volatility_ask, maturity, log_moneyness, phifun);

%% Compute fitted total implied variances and implied volatilities using svi_jumpwing
% Preallocating matrices for computed values
model_total_implied_variance      = zeros(size(implied_volatility));
model_implied_volatility          = zeros(size(implied_volatility));
model_total_implied_variance_bid  = zeros(size(implied_volatility));
model_implied_volatility_bid      = zeros(size(implied_volatility));
model_total_implied_variance_ask  = zeros(size(implied_volatility));
model_implied_volatility_ask      = zeros(size(implied_volatility));

for t = 1:length(T_vectU)
    pos = (maturity == T_vectU(t));
    [model_total_implied_variance(pos), model_implied_volatility(pos)] = ...
        svi_jumpwing(log_moneyness(pos), parameters(:,t), T_vectU(t));
    [model_total_implied_variance_bid(pos), model_implied_volatility_bid(pos)] = ...
        svi_jumpwing(log_moneyness(pos), parameters_bid(:,t), T_vectU(t));
    [model_total_implied_variance_ask(pos), model_implied_volatility_ask(pos)] = ...
        svi_jumpwing(log_moneyness(pos), parameters_ask(:,t), T_vectU(t));
end

%% Plot calibrated implied volatilities for mid, bid, and ask (slice plots)
nCol = 5;
nRow = ceil(length(T_vectU) / nCol);

figure;
for t = 1:length(T_vectU)
    pos = (maturity == T_vectU(t));
    [x, idx] = sort(log_moneyness(pos));
    
    IV       = model_implied_volatility(pos);       IV = IV(idx);
    IV_a     = model_implied_volatility_ask(pos);     IV_a = IV_a(idx);
    IV_b     = model_implied_volatility_bid(pos);     IV_b = IV_b(idx);
    
    subplot(nRow, nCol, t);
    plot(x, IV_a, 'vb', 'LineWidth', 1.5, 'MarkerSize', 0.5);
    hold on;
    plot(x, IV, '-r', 'LineWidth', 1.5);
    plot(x, IV_b, '^b', 'LineWidth', 1.5, 'MarkerSize', 0.5);
    xlabel('$\ln\frac{K}{F}$', 'Interpreter', 'latex');
    ylim([0.1 0.6]);
    grid on;
    title(sprintf('Maturity = %.2f', T_vectU(t)));
    hold off;
end

%% Export calibrated implied volatilities to CSV
T_out = table(model_implied_volatility_ask, model_implied_volatility_bid, model_implied_volatility, ...
    'VariableNames', {'model_implied_volatility_ask', 'model_implied_volatility_bid', 'model_implied_volatility'});
writetable(T_out, 'implied_volatilities.csv');

%% Export SVI parameters to CSV
maturities = T_vectU;         % Column vector of unique maturities
svi_parameters = parameters';  % Transpose so that each row corresponds to a maturity
params_table = array2table([maturities, svi_parameters], ...
    'VariableNames', {'Maturity', 'v', 'psi', 'p', 'c', 'vt'});
writetable(params_table, 'svi_parameters.csv');

%% 3D Surface of SVI Implied Volatility
% Define a grid for log-moneyness
k_range = linspace(min(log_moneyness)-0.5, max(log_moneyness)+0.5, 100);

% Preallocating a matrix for the implied volatility surface.
% Rows correspond to log-moneyness (k_range) and columns to maturities (T_vectU).
ImpVolSurface = zeros(length(k_range), length(T_vectU));

for t = 1:length(T_vectU)
    % Identify data for the current maturity
    pos = (maturity == T_vectU(t));
    
    % Extract calibrated parameters for this maturity
    params = parameters(:, t);
    
    % Compute a representative theta as the average fitted total variance at this maturity
    theta_val = mean(model_total_implied_variance(pos));
    
    % Map the parameters as required by svi_surface.
    % Here, we assume:
    %   - The third parameter is rho.
    %   - The phi parameter is expected as a two-element vector. If calibration returns a scalar,
    %     we supply a default second element.
    rho_val   = params(3);
    phi_param = [params(2), 1];  % Adjust the default value if needed.
    
    % Compute the SVI surface for the current maturity over the k_range grid.
    [~, impliedvol] = svi_surface(k_range, theta_val, rho_val, phifun, phi_param, T_vectU(t));
    % Also add a line to store the SVI parameters used above
    % and export that to a .csv file called ssvi_parameters_check.csv
    % Store the computed implied volatility into the surface matrix.
    ImpVolSurface(:, t) = impliedvol(:);
end

% Create a meshgrid for plotting: X = log-moneyness, Y = maturity.
[K, T] = meshgrid(k_range, T_vectU);

% Plot the 3D surface.
figure;
surf(K, T, ImpVolSurface', 'EdgeColor', 'none');
xlabel('$\ln\frac{K}{F}$','Interpreter','latex');
ylabel('Maturity');
zlabel('Implied Volatility');
title('3D SVI Implied Vol Surface');
colorbar;
shading interp;
view(135, 30);

%% Revised function with BOTH calendar-spread AND butterfly-spread checks
function ArbitrageChecks(params_table)
    % Extract columns from the params_table
    maturities = params_table.Maturity;
    v_vals     = params_table.v;     % interpreted as theta(t)
    rho_vals   = params_table.p;     % interpreted as rho(t)
    psi_vals   = params_table.psi;   % power-law coefficient
    c_vals     = params_table.c;     % power-law exponent
    
    %  CALENDAR-SPREAD CHECK
    %  ----------------------
    %  Condition 1: theta(t) should be nondecreasing
    [~, sortIdx] = sort(maturities);
    v_vals   = v_vals(sortIdx);
    rho_vals = rho_vals(sortIdx);
    psi_vals = psi_vals(sortIdx);
    c_vals   = c_vals(sortIdx);
    
    diffs = diff(v_vals);
    if any(diffs < 0)
        warning('Calendar-spread condition 1 violated: theta(t) is not nondecreasing.');
    else
        disp('Calendar-spread condition 1 passed: theta(t) is nondecreasing in t.');
    end
    
    %  Condition 2: 0 <= d/dtheta [theta*phi(theta)] <= upper bound
    for i = 1:length(v_vals)
        theta = v_vals(i);
        rho   = rho_vals(i);
        psi   = psi_vals(i);
        c     = c_vals(i);
        
        if theta <= 0
            warning('theta(t) <= 0 at index %d. Skipping derivative check.', i);
            continue;
        end
        
        % Derivative of [theta * varphi(theta)] = psi*(c+1)*theta^c
        deriv = psi * (c + 1) * theta^c;
        
        % Lower bound check
        if deriv < 0
            warning('Calendar-spread condition 2 (lower bound) violated at index %d.', i);
        end
        
        % If rho is nearly zero, skip the upper bound check
        if abs(rho) < 1e-12
            continue;
        end
        
        % varphi(theta) = psi*theta^c
        phi_theta = psi * theta^c;
        upperBound = (1 / rho^2) * (1 + sqrt(1 - rho^2)) * phi_theta;
        
        if deriv > upperBound
            warning('Calendar-spread condition 2 (upper bound) violated at index %d.', i);
        end
    end
    disp('Calendar-spread checks complete.');
    
    %  BUTTERFLY-SPREAD CHECK
    %  ----------------------
    %  According to Gatheral & Jacquier (Theorem 4.2):
    %   1) theta*varphi(theta)*(1+|rho|) < 4   (strict)
    %   2) theta*varphi(theta)^2*(1+|rho|) <= 4
    disp('Starting butterfly-spread checks...');
    for i = 1:length(v_vals)
        theta = v_vals(i);
        rho   = rho_vals(i);
        psi   = psi_vals(i);
        c     = c_vals(i);
        
        if theta <= 0
            warning('theta(t) <= 0 at index %d. Skipping butterfly check.', i);
            continue;
        end
        
        % varphi(theta) = psi*theta^c
        phi_theta = psi * theta^c;
        
        % Condition 1
        check1 = theta * phi_theta * (1 + abs(rho));
        if check1 >= 4
            warning('Butterfly condition 1 violated at index %d: %.4f >= 4', i, check1);
        end
        
        % Condition 2
        check2 = theta * (phi_theta^2) * (1 + abs(rho));
        if check2 > 4
            warning('Butterfly condition 2 violated at index %d: %.4f > 4', i, check2);
        end
    end
    disp('Butterfly-spread checks complete.');
end

%%
ArbitrageChecks(params_table)
