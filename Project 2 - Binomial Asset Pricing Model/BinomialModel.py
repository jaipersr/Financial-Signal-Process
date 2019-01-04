# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:44:24 2018
Ryan Jaipersaud
Binomial Asset Pricing model
The code below uses the binomial asset pricing model to determine the replicating 
portfolio by both using the martingale approach and the replicating portfolio strategy.
Tables of the expected payoff (V) at time N as well as at time 0 are generated for 
various probabilities including the risk neutral probability. Random walks were generated 
by Monte Carlo methods. The delta hedging factors and stock prices are plotted 
for the last 3 steps of a specific walk.
"""
import random
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from copy import deepcopy

# This function calculates the strike price which is a function of the 
# average return for a binomial model
def strike_price(n,u,d,p):
    S0 = 1
    expectation = (numpy.log(u/d))*n*p + n*numpy.log(d)
    K = S0*numpy.exp(expectation)
    return K

# This calculates the risk neutral probability based on the interest 
# rate and the up and down factors
def risk_neutral(u,d,r):
    p_tilde = ((1+r)- d)/(u-d)
    return p_tilde

# This function will create L random walks that are n steps long and will find the
# payoff V at the end of each walk. It will store the expected average payoff at N
# in V_N. It will then find the discounted payoff at N = 0 and store it in X0 (replicated portfolio).
# This function will also store all L paths inside of S_path for plotting purposes.
def martingale_approach(n,u,d,p,r,option,K = 0,plot = 'off'):
    L = 1000 # number of walks/ simulationg generated
    V0 = 0
    V_N = 0 
    S_path = numpy.zeros((n+1,1))
    for i in range(L): # this determines the number of random walks generated
        w_path = random_walk(n,p) # Generate the random walk
        # Based on the option passed to the function the function will go into one 
        # of the if conditions
        if option == 'lookback':
            V, s_path = lookback(w_path,u,d) # this calculates the payoff for the specific function
        elif option == 'call':
            V, s_path = European_call(w_path,u,d,K)
        elif option == 'put':
            V, s_path = European_call(w_path,u,d,K)
        else:
            print('Error: You have chosen an option that does not exist')
        V_N = V_N + V # this will hold the cumulative pay off of each walk
        S_path = numpy.hstack((S_path,s_path)) # S_path holds all paths
        
    n_path = numpy.arange(n+1) # path of iteration lives in n + 1
    S_path = S_path[:,1:n] # this removes the initial column of zeros
    if plot == 'on':
        plt.figure(1)
        for i in range(S_path.shape[1]):
            plt.plot(n_path, S_path[:,i])
        plt.xlabel =('Flips/time')
        plt.ylabel =('Stock Price')
        Title = 'Stock Price for option = ' + str(option) + ',strike price = ' + str(round(K,2))
        plt.title(Title)
        
    S_path = numpy.mean(S_path,axis = 1) # calculated the average path by averaging over rows
    S_path = numpy.array([S_path]).T
    V_N = V_N/L # Average pay off after n steps
    V0 = V_N/((1+r)**n) # discounted payoff
    X0 = V0 # this is the replicating portfolio
    return X0,V_N, S_path, n_path

# This function will add a head to a path and a tail to a path and return both paths
# This is to make a binomial tree
def binomial_tree(w_path):
    w_path_A = numpy.vstack((deepcopy(w_path),[1]))
    w_path_B = numpy.vstack((deepcopy(w_path),[0]))
    return w_path_A, w_path_B
    

# This function calculate the hedge factor associated with the previous flip.
# This assume that w_path_A and w_path_b are the same length n and thus the 
# delta that is returned is the hedging strategy for n-1
def delta_hedge(w_path_A,w_path_B,u,d,p,r,option,K):
    
    if option == 'lookback':
        V_A, s_path_A = lookback(w_path_A,u,d)
        V_B, s_path_B = lookback(w_path_B,u,d)
    elif option == 'call':
        V_A, s_path_A = European_call(w_path_A,u,d,K)
        V_B, s_path_B = European_call(w_path_B,u,d,K)
    elif option == 'put':
        V_A, s_path_A = European_put(w_path_A,u,d,K)
        V_B, s_path_B = European_put(w_path_B,u,d,K)
        
    delta = (V_A - V_B)/(s_path_A[-1] - s_path_B[-1] )
    Vn = (1/(1+r))*( (p*V_A) + ((1-p)*V_B))
    return delta, Vn
    

# This is a monte carlo function that creates a random walk making n steps.
# It first generates a feasible set of heads and tails that reflects the probability passed to it.
# It then randomly selects from the set to generate a path w_path 
# This path is then passed to an option for calculation of the replicated portfolio and 
# payoff at time N
def random_walk(n,p):
    w_path = numpy.array([]) # This will contain the path of the random walk
    
    H_set = ['H']
    H_set_copies = 100 * round(p, 2) # if p1 = 0.4 then H_set_ copies equals 40
    H = numpy.tile(H_set,int(H_set_copies)) # This will create a row vector of 'H's ( if p1 = 0.4 it will create a row vector of 40 H's)
    
    T_set = ['T']
    T_set_copies = 100 * round(1-p, 2)
    T = numpy.tile(T_set,int(T_set_copies)) # This will create a row vector of 'T's ( if p1 = 0.4 it will create a row vector of 60 T's)
    
    feasible_set = numpy.hstack((H,T)) # This set reflects the probability of choosing heads and tails

    for i in range(n):
        step = random.choice(feasible_set)
        if step == 'H':
            w_path = numpy.append(w_path,1) # This appends a 1 to the path if heads was chosen
        else:
            w_path = numpy.append(w_path,0) # This appends a 0 to the path if tails was chosen
    w_path = numpy.array([w_path]).T # This turn w from being (n,) to (n,1) to prevent problems later on
    return  w_path 

# This is a look back option. Depending on the path generated by the Monte Carlo technique
# it will move through the path and will multiple the initial stock value by up or down factors.
# It will then calculate the payoff of the path
def lookback(w_path,u,d):
    S = 1 # initial stock value
    s_path = S *numpy.cumprod((u-d)*w_path + d) # this calculate the stock path as a function of the w_path
    s_path = numpy.insert(s_path,0,S) # insert the initial stock value at the beginning of the list
    s_path = numpy.array([s_path]).T # turn s_path into vector
    V = max(s_path - s_path[-1]) # finds maximum difference between an element of s_path and the last entry
    return V,s_path

def European_call(w_path,u,d,K):
    S = 1 # initial stock value
    s_path = S *numpy.cumprod((u-d)*w_path + d)
    s_path = numpy.insert(s_path,0,S) # insert the initial stock value at the beginning of the list
    s_path = numpy.array([s_path]).T
    
    if (s_path[-1] - K) > 0: # european call function
        V = s_path[-1] - K
    else:
        V = 0
    return V,s_path

def European_put(w_path,u,d,K):
    S = 1 # initial stock value
    s_path = S *numpy.cumprod((u-d)*w_path + d)
    s_path = numpy.insert(s_path,0,S) # insert the initial stock value at the beginning of the list
    s_path = numpy.array([s_path]).T
    
    if ( K - s_path[-1] ) > 0:# european put function
        V =  K - s_path[-1] 
    else:
        V = 0
    return V,s_path

    
    


n = 100
u = 1.005
d = 1.002
r = 0.003
p1 = 0.4
p2 = 0.6
ptilde = risk_neutral(u,d,r)

K1 = strike_price(n,u,d,p1)
K2 = strike_price(n,u,d,p2)
K3 = strike_price(n,u,d,ptilde)
print('Strike Prices: ','K1 = ',K1,'K2 = ', K2,'K3 = ', K3)


print('-----------------------------------------Part E------------------------')
# This part generates the expected payoffs for using probabilities p1 and p2 with all three stike prices
# for a Euopean call
X_p1_K1_C,V_p1_K1, S_path, n_path = martingale_approach(n,u,d,p1,r,'call',K1,plot = 'off')
X_p1_K2_C,V_p1_K2, S_path, n_path = martingale_approach(n,u,d,p1,r,'call',K2,plot = 'off')
X_p1_K3_C,V_p1_K3, S_path, n_path = martingale_approach(n,u,d,p1,r,'call',K3,plot = 'off')

V_p1 = numpy.vstack((V_p1_K1,V_p1_K2,V_p1_K3))

X_p2_K1_C,V_p2_K1, S_path, n_path = martingale_approach(n,u,d,p2,r,'call',K1,plot = 'off')
X_p2_K2_C,V_p2_K2, S_path, n_path = martingale_approach(n,u,d,p2,r,'call',K2,plot = 'off')
X_p2_K3_C,V_p2_K3, S_path, n_path = martingale_approach(n,u,d,p2,r,'call',K3,plot = 'off')

V_p2 = numpy.vstack((V_p2_K1,V_p2_K2,V_p2_K3))

V_N = numpy.hstack((V_p1,V_p2)) # This horizontally stacks the payoff for each probability

df = DataFrame(data = V_N, columns = ['p1 = '+str(p1),'p2 = ' +str(p2)]) # Creates a summary table
print('For a European Call the Expected payoff at time N = 100 for L = 1000 is summarized in the below table.' )
print('The row index 0 is K1, index 1 is K2, index 2 is K3.')
print(df,'\n')
# ---------------------------------------------------------------------------------------------
# This part generates the expected payoffs for using probabilities p1 and p2 with all three stike prices
# for a Euopean put
X_p1_K1_P,V_p1_K1, S_path, n_path = martingale_approach(n,u,d,p1,r,'put',K1,plot = 'off')
X_p1_K2_P,V_p1_K2, S_path, n_path = martingale_approach(n,u,d,p1,r,'put',K2,plot = 'off')
X_p1_K3_P,V_p1_K3, S_path, n_path = martingale_approach(n,u,d,p1,r,'put',K3,plot = 'off')

V_p1 = numpy.vstack((V_p1_K1,V_p1_K2,V_p1_K3))

X_p2_K1_P,V_p2_K1, S_path, n_path = martingale_approach(n,u,d,p2,r,'put',K1,plot = 'off')
X_p2_K2_P,V_p2_K2, S_path, n_path = martingale_approach(n,u,d,p2,r,'put',K2,plot = 'off')
X_p2_K3_P,V_p2_K3, S_path, n_path = martingale_approach(n,u,d,p2,r,'put',K3,plot = 'off')

V_p2 = numpy.vstack((V_p2_K1,V_p2_K2,V_p2_K3))

V_N = numpy.hstack((V_p1,V_p2))

df = DataFrame(data = V_N, columns = ['p1 = '+str(p1),'p2 = ' +str(p2)])
print('For a European Put the Expected payoff at time N = 100 for L = 1000 is summarized in the below table.' )
print('The row index 0 is K1, index 1 is K2, index 2 is K3.')
print(df,'\n')

#-----------------------------------------------------------------------------
# This part will calculate the replicated portfolio for all probabilities for a European Call
print('-----------------------------------------Part F and H------------------------')
X_ptilde_K1_C,V_ptilde_K1, S_path, n_path = martingale_approach(n,u,d,ptilde,r,'call',K1,plot = 'off')
X_ptilde_K2_C,V_ptilde_K2, S_path, n_path = martingale_approach(n,u,d,ptilde,r,'call',K2,plot = 'off')
X_ptilde_K3_C,V_ptilde_K3, S_path, n_path = martingale_approach(n,u,d,ptilde,r,'call',K3,plot = 'off')

X_p1_C = numpy.vstack((X_p1_K1_C,X_p1_K2_C,X_p1_K3_C)) # These portflios were already calculated above
X_p2_C = numpy.vstack((X_p2_K1_C,X_p2_K2_C,X_p2_K3_C)) # These portflios were already calculated above
X_ptilde_C = numpy.vstack((X_ptilde_K1_C,X_ptilde_K2_C,X_ptilde_K3_C))

X0 = numpy.hstack((X_p1_C,X_p2_C,X_ptilde_C)) # This horizontally stacks the replicated portfolio for each probability

df = DataFrame(data = X0, columns = ['p1 = '+str(p1),'p2 = ' +str(p2),'ptilde = '+ str(round(ptilde,2))])
print('For a European Call the replicated portfolio (X0) at time N = 0 for L = 1000 is summarized in the below table.' )
print('The row index 0 is K1, index 1 is K2, index 2 is K3.')
print(df,'\n')
#-----------------------------------------------------------------------------
# This part will calculate the replicated portfolio for all probabilities for a European Put
X_ptilde_K1_P,V_ptilde_K1, S_path, n_path = martingale_approach(n,u,d,ptilde,r,'put',K1,plot = 'off')
X_ptilde_K2_P,V_ptilde_K2, S_path, n_path = martingale_approach(n,u,d,ptilde,r,'put',K2,plot = 'off')
X_ptilde_K3_P,V_ptilde_K3, S_path, n_path = martingale_approach(n,u,d,ptilde,r,'put',K3,plot = 'off')

X_p1_P = numpy.vstack((X_p1_K1_P,X_p1_K2_P,X_p1_K3_P)) # These portflios were already calculated above
X_p2_P = numpy.vstack((X_p2_K1_P,X_p2_K2_P,X_p2_K3_P)) # These portflios were already calculated above
X_ptilde_P = numpy.vstack((X_ptilde_K1_P,X_ptilde_K2_P,X_ptilde_K3_P))

X0 = numpy.hstack((X_p1_P,X_p2_P,X_ptilde_P))

df = DataFrame(data = X0, columns = ['p1 = '+str(p1),'p2 = ' +str(p2),'ptilde = '+ str(round(ptilde,2))])
print('For a European Put the replicated portfolio (X0) at time N = 0 for L = 1000 is summarized in the below table.' )
print('The row index 0 is K1, index 1 is K2, index 2 is K3.')
print(df,'\n')

#----------------------------Part H -------------------------------------------
# This is the replicating portfolio strategy
# This will create a random walk of 100 - 3 = 97 steps and then it will produce 
# a binomial path for the last three steps ( 8 paths total)
# This will calculate the deltas and pull back Vn for V_100, V_99, V_98, delta_99, delta_98, delta_97
print('-----------------------------------------Part G-----------------------------')
k = 3
n = 6 # deltas will be different if n is large n =100 usually implies deltas are all 1's or all zeros 
u = 1.005
d = 1.002
r = 0.003
p = 0.5
K = strike_price(n,u,d,p)

w_path_start = random_walk(n-k,p) # create a starting vector
V,s_path = European_call(w_path_start,u,d,K)

# First branching of tree paths
w_path_H, w_path_T = binomial_tree(w_path_start)

#Second Branching of tree paths
w_path_HH, w_path_HT = binomial_tree(w_path_H)
w_path_TH, w_path_TT = binomial_tree(w_path_T)

# Third branching of tree paths
w_path_HHH, w_path_HHT = binomial_tree(w_path_HH)
w_path_HTH, w_path_HTT = binomial_tree(w_path_HT)
w_path_THH, w_path_THT = binomial_tree(w_path_TH)
w_path_TTH, w_path_TTT = binomial_tree(w_path_TT)


# You need to calculate the last set of deltas by computing the last set of paths
# This is to get the payoff  V of these paths which you pull backwards using the probabilities
# This is for delta N - 1
delta_HH,V_HH = delta_hedge(w_path_HHH,w_path_HHT,u,d,p,r,'call',K)
delta_HT,V_HT = delta_hedge(w_path_HTH,w_path_HTT,u,d,p,r,'call',K)
delta_TH,V_TH = delta_hedge(w_path_THH,w_path_THT,u,d,p,r,'call',K)
delta_TT,V_TT = delta_hedge(w_path_TTH,w_path_TTT,u,d,p,r,'call',K)

# To find the deltas for paths N-2 to N - 3 you need to use the V_n that were just calculated.
# But now you need to generate the stock prices for these paths for the next delta calculation
V,s_path_HH = European_call(w_path_HH,u,d,K)
V,s_path_HT = European_call(w_path_HT,u,d,K)
V,s_path_TH = European_call(w_path_TH,u,d,K)
V,s_path_TT = European_call(w_path_TT,u,d,K)

# This is delta N-2
delta_H = (V_HH - V_HT)/ (s_path_HH[-1] - s_path_HT[-1]) # equation 1.2.17
delta_T = (V_TH - V_TT)/ (s_path_TH[-1] - s_path_TT[-1])

V_H = (1/(1+r)) * (p*V_HH +((1-p)*V_HT)) # equation 1.2.16
V_T = (1/(1+r)) * (p*V_TH +((1-p)*V_TT))
V,s_path_H = European_call(w_path_H,u,d,K)
V,s_path_T = European_call(w_path_T,u,d,K)

# This is delta N - 3
delta_0 = (V_H - V_T) / (s_path_H[-1] - s_path_T[-1])

# This is V_97
V_0 = (1/(1+r)) * (p*V_H +((1-p)*V_T)) 

print('\nThe replicated portfolio and hedging strategy are as follows for a European call')
print('For p = 0.5 the follow deltas were calculated for n = N - 3 to n = N - 1 with n = ', n)
# The deltas are usualy all zeros or all ones if n is large but sometimes they change
print('Part_1: delta N-1', delta_HH, delta_HT, delta_TH, delta_TT)
print('Part_2: delta N-2', delta_H, delta_T)
print('Part_3: delta N-3', delta_0)
print('Part_4: V0', V_0)

# The code below create a delta path from 0 to 2 or in this case from N-3 to N-1
plt.figure(1)
n = [0,1,2]
delta_1 = [delta_0,delta_H,delta_HH]
delta_2 = [delta_0,delta_H,delta_HT]
delta_3= [delta_0,delta_T,delta_TH]
delta_4 = [delta_0,delta_T,delta_TT]
plt.plot(n,delta_1,'r',n,delta_2,'b',n,delta_3,'g',n,delta_4,'k')
plt.xlabel('Flips/time')
plt.ylabel('delta')
plt.title('Deltas for last three tosses')

# This plots all 8 branches for the initial path w_path_start
V,s_path_HHH = European_call(w_path_HHH,u,d,K)
V,s_path_HHT = European_call(w_path_HHT,u,d,K)
V,s_path_HTH = European_call(w_path_HTH,u,d,K)
V,s_path_HTT = European_call(w_path_HTT,u,d,K)

V,s_path_THH = European_call(w_path_THH,u,d,K)
V,s_path_THT = European_call(w_path_THT,u,d,K)
V,s_path_TTH = European_call(w_path_TTH,u,d,K)
V,s_path_TTT = European_call(w_path_TTT,u,d,K)

s_path = numpy.hstack((s_path_HHH,s_path_HHT,s_path_HTH,s_path_HTT, s_path_THH,s_path_THT,s_path_TTH,s_path_TTT))
n_path = numpy.arange(s_path_HHH.shape[0])
plt.figure(2)
for i in range(8):
    plt.plot(n_path,s_path[:,i])
plt.xlabel('Flips/time')
plt.ylabel('Stock Price')
plt.title('Stock Price for last three tosses')
