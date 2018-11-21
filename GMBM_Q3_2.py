# -*- coding: utf-8 -*-
"""
Ryan Jaipersaud
ECE-478: Financial Signal Processing
11/18/2018
Project: Correlated Brownian Motion and Portfolio Creation
The purpose of the code below is to generate stock paths for 2 stocks with 2 underlying
brownian motion paths that are correlated to one another. A 2 by 2 covariance matrix
is provided in the code. 10 plots of the correlated brownian motion and stock price
are generated. The formula for the optimal weight for each stock in the porfolio 
is computed by setting up by setting up an optimization  problem of the form:
risk^2 = min w.T*C'w
s.t.  sum(w_i) = 1  (1 dimensional hyperplane)
and solving the langrangian form to find w*.
If w* was found to involve short selling where one of the weights equaled a negative
value the point was discarded. Points on the boundary [1,0] and [0,1] were checked.
The portfolios corresponding to the weights that gave the largest and lowest risk 
were found and computed for the 10 stocks generated above. 
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


# This function generates m (2 by 1) gaussian vectors that are independent of 
# each other which have a mean of zero and a standard devation of delta. 
# From there is calculates a brownian motion path W of 2 stocks that are correlated
# according to a covariance matrix 
def generate_dW(m,delta):
    C = np.array([[1,0.75],[0.75,0.9]]) # Covarance matrix of W
    C_prime = scipy.linalg.sqrtm(C) #This is the square root of C s.t. C_prime*C_prime = C
    
    dX_paths = np.zeros((2,1)) # This holds all randomly generated vectors
    
    # this generate m random gaussian columns
    for i in range(m):
        dX = np.random.normal(0,np.sqrt(delta),2) # This picks a 1 by 2 vector from a gaussian distribution
        dX = np.array([dX]).T # Turns row into a vector
        dX_paths = np.hstack((dX_paths,dX)) # stack the randomly generated vector onto dX_paths
        
    dX_paths = np.delete(dX_paths, (0), axis=1) # removes initial column 
    dW = np.sqrt(delta)*np.matmul(C_prime,dX_paths) # This computes dW = (delta^1/2)*(C^1/2)*dX
    
    W = np.array([[1],[1]]) # initializes brownian path
    W = np.hstack((W,dW)) # appends initialzation to beginning of dW
    W = np.cumsum(W,axis = 1) # computes W[i+1] = W[i] + dW
    
    # The covariance of W is roughly C
    #print(np.cov(dW))
    #print(delta*C)
    return W,dW

# The pupose of the function below is to take in the differential vector of a 
# brownian motion path from generate_dW() and compute the stock price for 2 stocks.
def generate_S(dW):
    global t,alphas,sigma
    
    S_1 = np.array([1]) # Holds the stock price of the first stock
    S_2 = np.array([1]) # Holds the stock price of the second stock
    
    # The for loop below computes the stock based on the correlated SDEs
    for i in range(dW.shape[1]): 
        dS_1 = alphas[0]*S_1[i] + sigma[0,0]*S_1[i]*dW[0,i]+sigma[0,1]*S_1[i]*dW[1,i] 
        dS_2 = alphas[1]*S_2[i] + sigma[1,0]*S_2[i]*dW[1,i]+sigma[1,1]*S_2[i]*dW[1,i] 
        S_1 = np.append(S_1,S_1[i] + dS_1)
        S_2 = np.append(S_2,S_2[i] + dS_2)
    
    S = np.vstack((S_1,S_2)) # stacks the paths that were just generated
    return S

# The below function returns the risk of a portfolio given the weights and the 
# M by M sigma matrix of risks and the covariance matrix
def sigma_prime(w,sigma):
    C = np.array([[1,0.75],[0.75,0.9]]) # Covarance matrix of W
    sigma_prime = w.T.dot(sigma).dot(C).dot(sigma.T).dot(w)
    return sigma_prime

# The below function returns the adapted covariance matrix given the
# M by M sigma matrix of risks and the covariance matrix
def c_prime(sigma):
    C = np.array([[1,0.75],[0.75,0.9]]) # Covarance matrix of W
    c_prime = sigma.dot(C).dot(sigma.T)
    return c_prime
        
# Paramaters
N = 250 # Length of path generated, for N = 10 the plot of S and V become more linear
m = N   # number of 2 by 1 gaussian vectors that need to be generated
T = 1   # end time
delta = T/N # time step
t = (T*np.arange(N+1)/N) # time vector

alpha = 0.1/N
alphas = np.array([[alpha],[alpha]])
sigma = np.array([[0.3,0.5],[0.4,2*alpha]]) # this can greatly affect how plots of S and V turn out
eigen_value, vector = np.linalg.eig(sigma)
determinant = np.linalg.det(sigma)


#------------------------ Question 3 & 4 ------------------------------------------
print('------------------------ Questions 3 & 4 ------------------------------------')
print('For Q3 part A please look at code for function generate_dW which generates m gaussian vectors.')
print('The sigma matrix is as follows:')
print(sigma)
print('The determinant for sigma is',determinant)
print('Therefore sigma is invertible with all positive entries.\n')

C_prime = c_prime(sigma) # this determines the adapted covariance matrix C' = s*C*(s^T)

# This computes the weight of the each stock using the adapted covariance
weight_1 = (C_prime[1,1]-C_prime[0,1])/ ( C_prime[0,0] - C_prime[0,1] - C_prime[1,0] + C_prime[1,1] ) 
weight_2 = 1 - weight_1

print('Optimal Weights')
print('weight: ',weight_1)
print('weight: ',weight_2)

# This sets the boundary points
b_1 = np.reshape(np.array([1,0]),(2,1))
w_soln = np.array([[weight_1],[weight_2]])
b_2 = np.reshape(np.array([0,1]),(2,1))

# calculate the adapted sigma values s' = sqrt( (w^T)*C'*w )
sigma_prime_1 = np.sqrt(b_1.T.dot(C_prime).dot(b_1))
sigma_prime_soln = np.sqrt(w_soln.T.dot(C_prime).dot(w_soln))
sigma_prime_2 = np.sqrt(b_2.T.dot(C_prime).dot(b_2))

print('\nSigmas')
print('sigma_prime_1: ',sigma_prime_1)
print('sigma_prime_soln: ',sigma_prime_soln)
print('sigma_prime_2: ',sigma_prime_2,'\n')

if weight_1 < 0 or weight_2 < 0:
    print('Since one of the weights calculated above corresponded to short selling (i.e. less than zero) the point was rejected.')
    print('Please note that becasue the point was rejected the arguements that give the max and min are on the boundary.')
    print('Therefore one of the weights in the portfolio is zero such that Vmin and Vmax will each model a single stock.')
    print('Thus the plots of V and S will look similar.')
    weights = [b_1,b_2] # stores weights in weight vector
    calc_sigmas = [sigma_prime_1,sigma_prime_2] # stores the calculated sigmas in a vector
else:
    weights = [b_1,w_soln,b_2] # stores weights in weight vector
    calc_sigmas = [sigma_prime_1,sigma_prime_soln,sigma_prime_2] # stores the calculated sigmas in a vector

# Determine the weight with the lowest and greatest risk 
w_min = weights[calc_sigmas.index(min(calc_sigmas))]
w_max = weights[calc_sigmas.index(max(calc_sigmas))]
sigma_min = min(calc_sigmas)
sigma_max = max(calc_sigmas)


f1,axes1 = plt.subplots(nrows = 2,ncols = 5)
f2,axes2 = plt.subplots(nrows = 2,ncols = 5)
f3,axes3 = plt.subplots(nrows = 2,ncols = 5)

f1.suptitle('Q3 Part B: Correlated Brownian motion paths W(t)')
f2.suptitle('Q3 Part C: Correlated Stock Paths S(t)')
f3.suptitle('Q4 Correlated Portfolios V(t)')

f1.text(0.5, 0.04, 'time', ha='center')
f1.text(0.04, 0.5, 'W(t)', va='center', rotation='vertical')
f2.text(0.5, 0.04, 'time', ha='center')
f2.text(0.04, 0.5, 'S(t)', va='center', rotation='vertical')
f3.text(0.5, 0.04, 'time', ha='center')
f3.text(0.04, 0.5, 'V(t)', va='center', rotation='vertical')

for i in range(10):
    W,dW = generate_dW(m,delta) # generate correlated brownian path 
    S = generate_S(dW) # calculate the stock price using the brownian path
    V_min = w_min.T.dot(S)# generate minimum portfolio
    V_max = w_max.T.dot(S)# generate maximum portfolio
    
    if i < 5:
        axes1[0,i].plot(t,W[0,:],label = 'W1')
        axes1[0,i].plot(t,W[1,:],label = 'W2')
        axes2[0,i].plot(t,S[0,:],label = 'S1')
        axes2[0,i].plot(t,S[1,:],label = 'S2')
        axes3[0,i].plot(t,V_max[0,:],label = 'Vmax')
        axes3[0,i].plot(t,V_min[0,:],label = 'Vmin')
        
    else:
        axes1[1,i-5].plot(t,W[0,:],label = 'W1')
        axes1[1,i-5].plot(t,W[1,:],label = 'W2')
        axes2[1,i-5].plot(t,S[0,:],label = 'S1')
        axes2[1,i-5].plot(t,S[1,:],label = 'S2')
        axes3[1,i-5].plot(t,V_max[0,:],label = 'Vmax')
        axes3[1,i-5].plot(t,V_min[0,:],label = 'Vmin')
    
f1.legend()
f2.legend()
f3.legend()

