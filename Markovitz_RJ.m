% Ryan Jaipersaud
% ECE-478 Financials Signal Processing
% 10/3/2018
% The code below generates four plots
% The first is a plot of the 5 markovitz bullets consisting of two randomly
% picked securities. The second is a plot of the markovitz bullet for the
% all 48 securites. The third is a plot of the Beta values of the MVP, 3
% portfolios and the naive portfolio. The fourth is a plot of a $1 dollar
% investment at the beginning of the year and its value over time.
% All interest rate data was taken from below.
% The years looked at include 2002, 2007 and 2014
% https://www.global-rates.com/interest-rates/libor/american-dollar/usd-libor-interest-rate-12-months.aspx

clc;
clear all;

% annual rates for 2000 - 2016 obtained from link above
YearNumbers = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016];
AnnualR = [6.866, 3.832, 2.206, 1.356, 2.121, 4.033, 5.325, 5.124, 3.089, 1.559, 0.923,0.830, 1.013, 0.683, 0.561, 0.794, 1.376]; 
Filename = 'Processed_Files/2002.csv';
K = csvread(Filename, 0 ,1);
T = transpose(K); % 48 by 251 roughly

Year = str2num(Filename(17:20));
YearIndex = (Year - 2000) + 1;

fprintf('Market Analysis for %d\n', Year)
Annual_USD_LIBOR = AnnualR(YearIndex);


%------------------------Part A -------------------------------------------
% mean value of each security over N observations
N = size(T,2);
Mean_48 = mean(T,2); % daily mean rate vector
Covr = (1/(N-1))*(T*K - Mean_48*transpose(Mean_48)); % daily covariance matrix

MeanR = 100*((1+(Mean_48./100)).^N -1); % annual mean rate vector
CovR = N*Covr; % annual covariance matrix


% -----------------------Part B -------------------------------------------
%Naive porfolio
naive_w = (1/48)*ones(48,1);
naive_mu = transpose(Mean_48)*naive_w; 
naive_sigma = sqrt(transpose(naive_w)*Covr*naive_w); 

%Minimum variance porfolio
OneVector = ones(48,1);
MVP_sigma = 1 / sqrt(transpose(OneVector)*inv(Covr)*OneVector);
MVP_w = (MVP_sigma^2)*inv(Covr)*OneVector;
MVP_mu = (transpose(Mean_48)*MVP_w); 

%Market Portfolio

daily_USD_LIBOR  = 100*( ((1+(Annual_USD_LIBOR/100))^(1/N)) -1);
Mean_ex = Mean_48 - daily_USD_LIBOR*OneVector;
Market_w = (1/ (transpose(OneVector)*inv(Covr)*Mean_ex)) * inv(Covr)*Mean_ex;
Market_mu = transpose(Mean_48)*Market_w; 
Market_sigma = sqrt(transpose(Market_w)*Covr*Market_w);



for f = 1:5
security1 = randi([1 48],1,1); % random selection of a security
security2 = randi([1 48],1,1); % % random selection of a security

% Recompute the Mean and covariance matrix for the two securities
X_security_2 = [T(security1,:); T(security2,:)]; 
Mean_security_2 = [Mean_48(security1,1);Mean_48(security2,1)];
C_security_2 = (1/(N-1))*( (X_security_2 * transpose(X_security_2)) - (Mean_security_2 * transpose(Mean_security_2)));

% Recompute the MVP and market porfolio of the 2 securities
sigma_MVP_2 = 1/sqrt(transpose([1;1])*C_security_2*[1;1]);
w_MVP_2 = (sigma_MVP_2^2)*inv(C_security_2)*[1;1];
mu_MVP_2 = transpose(Mean_security_2)*w_MVP_2;

% the vectors below will store the MVP of all 5 random picks
sigma_MVP_vector(f,1) = sigma_MVP_2;
mu_MVP_vector(f,1) = mu_MVP_2;


daily_USD_LIBOR  = 100*( ((1+(Annual_USD_LIBOR/100))^(1/N)) -1);
Mean_ex_2 = Mean_security_2 - daily_USD_LIBOR*[1;1];
Market_w_2 = (1/ (transpose([1;1])*inv(C_security_2)*Mean_ex_2)) * inv(C_security_2)*Mean_ex_2;
Market_mu_2(f,1) = transpose(Mean_security_2)*Market_w_2; % i dont like that this is negative
Market_sigma_2(f,1) = sqrt(transpose(Market_w_2)*C_security_2*Market_w_2);




% Calculate all values for the Markovitz bullet
rho_12 = corrcoef(T(security1,:),T(security2,:));
rho_12 = rho_12(1,2);
rho_12_vector(f,1) = rho_12; % This will store all correlation coefficents across securites
sigma1 = std(T(security1,:));
sigma2 = std(T(security2,:));
A = (sigma1^2 + sigma2^2 - 2*rho_12*sigma1*sigma2^2)/(( Mean_security_2(1,1) - Mean_security_2(2,1))^2);

lowerbound = min([Mean_48(security1,1), Mean_48(security2,1)]);
upperbound = max([Mean_48(security1,1), Mean_48(security2,1)]);

mu_v = transpose(linspace(lowerbound,upperbound,30)); % vector of possible mu values
sigma_v_2 = (sigma_MVP_2^2 + A^2*((mu_v - mu_MVP_2).^2)).^0.5; % vector of portfolio sigmas on markovitz bullet
    
Color = ['b','g','r','k','m'];
figure1 = plot( sigma_v_2, mu_v,Color(1,f), 'LineWidth',2);
hold on

legend1(f,1) = security1;
legend1(f,2) = security2;

end
legend1 = num2str(legend1);
legend(legend1(1,:), legend1(2,:), legend1(3,:), legend1(4,:), legend1(5,:))

scatter(sigma_MVP_vector,mu_MVP_vector); % Plots the MVPs for each curve
title(strcat('Figure 1-Markovitz Bullet for 2 securities in: ',Filename(17:20) ))
xlabel('sigma')
ylabel('mu')
saveas(figure1,'figure1.jpg')

%combo = [sigma_MVP_vector mu_MVP_vector rho_12_vector] ;
%----------------------Part C-----------------------------------------------
max(Mean_48);
m_tilde = [Mean_48, ones(48,1)]; % combination of constrainst
B = transpose(m_tilde)*inv(Covr)*m_tilde;
G = inv(B)*transpose(m_tilde)*inv(Covr)*m_tilde*inv(B);
a = G(1,1);
d = G(1,2);
b = G(2,2);

mu_v_48 = transpose(linspace(min(Mean_48),max(Mean_48),30));
calc_sigma_48 = @(x) sqrt(a*(x.^2) + 2*d*(x)+ b);
sigmav_48 = arrayfun(calc_sigma_48,mu_v_48); % return array of calc_sigma_48(mu_v_48)

figure(2)
plot(sigmav_48, mu_v_48)
title(strcat('Figure 2-Markovitz bullet for 48 securities in:', Filename(17:20)))
xlabel('sigma')
ylabel('mu')
hold on 
scatter(MVP_sigma, MVP_mu)
hold on
figure2 = plot([0 Market_sigma] , [daily_USD_LIBOR Market_mu], 'LineWidth', 1); % Plots line connecting interest rate to market porfolio
saveas(figure2,'figure2.jpg')
%-------------------------------- Part D ----------------------------------

step = (max(Mean_48) - MVP_mu)/4;
mu_48_1 = MVP_mu + step;
mu_48_2 = MVP_mu + 2*step;
mu_48_3 = MVP_mu + 3*step;

sigma_48_1 = calc_sigma_48(mu_48_1);
sigma_48_2 = calc_sigma_48(mu_48_2);
sigma_48_3 = calc_sigma_48(mu_48_3);

w_48_1 = inv(Mean_48*transpose(Mean_48))*Mean_48*mu_48_1;
w_48_2 = inv(Mean_48*transpose(Mean_48))*Mean_48*mu_48_2;
w_48_3 = inv(Mean_48*transpose(Mean_48))*Mean_48*mu_48_3;

w = warning('query','last');
id = w.identifier;
warning('off',id);

% becasue each element in alpha is a constant w_48_2 must be a combination of w_48_1 and w_48_3
alpha = (w_48_3+w_48_1)./w_48_2 

fprintf('Alpha is the summation of the weight vector for the first security plus the third divided element wise by the second.\n')
fprintf('Becasue each element in alpha is a constant w_48_2 must be a combination of w_48_1 and w_48_3.\n');
sigma_verification = [sigma_48_1;sigma_48_2;sigma_48_3];
mu_chosen = [ mu_48_1; mu_48_2;mu_48_3];
verificiationTable = table(mu_chosen,sigma_verification)
fprintf('As mu increases so does sigma.\n');


%--------------------------------Part E -----------------------------------
daily_USD_LIBOR  = 100*( ((1+(Annual_USD_LIBOR/100))^(1/N)) -1);
OneVector = ones(48,1);
MVP_sigma = 1 / sqrt(transpose(OneVector)*inv(Covr)*OneVector);
MVP_w = (MVP_sigma^2)*inv(Covr)*OneVector;
MVP_mu = (transpose(Mean_48)*MVP_w); % daily value

if daily_USD_LIBOR < MVP_mu
fprintf('The interest rate value %d is less than the MVP mu value %d\n', daily_USD_LIBOR, MVP_mu);
elseif daily_USD_LIBOR > MVP_mu
fprintf('The interest rate value %d is greater than the MVP mu value %d\n', daily_USD_LIBOR, MVP_mu);
end


%--------------------------------Part F -----------------------------------

daily_USD_LIBOR  = 100*( ((1+(Annual_USD_LIBOR/100))^(1/N)) -1);
Mean_ex = Mean_48 - daily_USD_LIBOR*OneVector;
Market_w = (1/ (transpose(OneVector)*inv(Covr)*Mean_ex)) * inv(Covr)*Mean_ex;
Market_mu = transpose(Mean_48)*Market_w;
Market_sigma = sqrt(transpose(Market_w)*Covr*Market_w);

mu = @(sigma) daily_USD_LIBOR + ((Market_mu - daily_USD_LIBOR)/Market_sigma)*sigma; % CML equation
slope = ((Market_mu - daily_USD_LIBOR)/Market_sigma);
fprintf('The equation of the CML is y = %d x + %d. \n',slope, daily_USD_LIBOR);

%-------------------------------Part G ------------------------------------
% This section calculates the Beta values 
B_MVP = (MVP_mu - daily_USD_LIBOR)/(Market_mu - daily_USD_LIBOR);
B_1 = (mu_48_1 - daily_USD_LIBOR)/(Market_mu - daily_USD_LIBOR);
B_2 = (mu_48_2 - daily_USD_LIBOR)/(Market_mu - daily_USD_LIBOR);
B_3 = (mu_48_3 - daily_USD_LIBOR)/(Market_mu - daily_USD_LIBOR);
B_naive = (naive_mu - daily_USD_LIBOR)/(Market_mu - daily_USD_LIBOR);


fprintf('The MVP    porfolio has a mu of %d and and a beta of %d. \n',MVP_mu,B_MVP );
fprintf('The first  porfolio has a mu of %d and and a beta of %d. \n',mu_48_1,B_1 );
fprintf('The second porfolio has a mu of %d and and a beta of %d. \n',mu_48_2,B_2 );
fprintf('The third  porfolio has a mu of %d and and a beta of %d. \n',mu_48_3,B_3 );
fprintf('The fourth porfolio has a mu of %d and and a beta of %d. \n',naive_mu,B_naive );

mu = [MVP_mu;mu_48_1;mu_48_2;mu_48_3;naive_mu];
betas = [B_MVP;B_1;B_2;B_3;B_naive];
beta_mu = table(mu,betas);

figure(3)
figure3 = plot(betas,mu,'-o');
title(strcat('Figure 3 -Mu versus Beta in:',Filename(17:20)))
xlabel('beta')
ylabel('mu')
saveas(figure3,'figure3.jpg')

%-------------------------------------Part H-------------------------------

dimensions = size(T);
time = dimensions(1,2);

%set initial values
Vo = 1; % initial capital is 1 dollar
V_market(1,1) = Vo;
V_MVP(1,1) = Vo;
V_naive(1,1) = Vo;
V_1(1,1) = Vo;
V_2(1,1) = Vo;
V_3(1,1) = Vo;

for t = 2:time+1 % iterate over all time points in your return matrix
    
% Mutiply each time column by the weight vector for the respective porfolio
Kv_market = transpose(T(:,t-1))* Market_w;
V_market(t,1) = (Kv_market + 1)*Vo;

Kv_MVP = transpose(T(:,t-1))* MVP_w;
V_MVP(t,1) = (Kv_MVP + 1)*Vo;

Kv_naive = transpose(T(:,t-1))* naive_w;
V_naive(t,1) = (Kv_naive + 1)*Vo;

Kv_1 = transpose(T(:,t-1))* w_48_1 ;
V_1(t,1) = (Kv_1 + 1)*Vo;
Kv_2 = transpose(T(:,t-1))* w_48_2 ;
V_2(t,1) = (Kv_2 + 1)*Vo;
Kv_3 = transpose(T(:,t-1))* w_48_3 ;
V_3(t,1) = (Kv_3 + 1)*Vo;
%Vo = V(t);
end

time = transpose(linspace(0,dimensions(1,2)+1,dimensions(1,2)+1));
%V_market(:,1);
figure(4)
figure4 = plot(time,V_market, time, V_MVP, time,V_naive, time, V_1,time, V_2,time, V_3);
legend('market','MVP','Naive','V_1','V_2','V_3')
title(strcat('Figure 4-Portfolio over time in:',Filename(17:20)))
xlabel('time (days)')
ylabel('dollar amount ($)')
%saveas(figure4,'figure4.jpg')

