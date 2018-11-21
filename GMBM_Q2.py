# -*- coding: utf-8 -*-
"""
Ryan Jaipersaud
ECE-478: Financial Signal Processing
11/18/2018
Project: Interest Rate Models
The purpose of the code below is understand how to transform SDEs to PDEs using 
Feynman Kac theorem. A discrete stochastic approach is still used to solve the 
SDEs. Based on parameters {a(t),b(t),s(t),Ka,Kb,Ks} L paths of length N are generated 
for the Hull White and Cox Ingersoll Ross models which take on the forms
dR(t) = (a(t)-b(t)*R(t))*dt + s(t)dW(t)~ and dR(t) = (a(t)-b(t)*R(t))*dt + s(t)(R(t)^0.5)dW(t)~,
repsectively. 10 random paths for HW and CIR are plotted. A symbolic form for 
the Hull White Model is also created using sympy. A Monte Carlo approach is used to
find the expectation and variance of each model for 0<t<T.
"""

    
    
    
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# The purpose of this function is to generate L random paths of length N 
# for the Hull White interest rate model. This returns the interest rate paths
# and record the number of times the interest falls below epsilon
def Hull_White(N,L):
    global a,b,s,t,epsilon,delta
    Ro = 1 # initial condition
    R_paths = np.zeros((1,N+1)) # This holds all rate paths 
    record_paths = np.zeros((1,N+1)) # This is a log of times where the rate went below a minimum epsilon
    exception = 0 # This tracks the total number of exceptions over all paths
    
    for j in range(L):# The outer for loop generates the number of paths specified
        
        R = np.array([Ro]) # R is a specific path being generated
        record = np.zeros((N+1)) # This hold a record of all exceptions for the specified path
        for i in range(N): # This for loop determines the length of each path
             
            dW_tilde = np.random.normal(0,np.sqrt(delta),1) # At each iteration this generates 1 number from a gausian distribution with mean 0 and st dev delta
            next_rate = R[i] + (a[i]-b[i])*R[i]*delta + s[i]*dW_tilde # computes the next rate
            
            if next_rate < epsilon: # if the next rate goes below epsilon then set the next_rate to epsilon
                exception = exception + 1 # increments the total number of exception
                next_rate = epsilon
                record[i] = t[i] # this will document the time the exception occurs
            R = np.append(R,next_rate ) # append the next rate to the path    
        R_paths = np.vstack((R_paths,R)) # appends the rate path to R_paths
        record_paths = np.vstack((record_paths,record)) # appends the record path to record_paths
    
    # removes initial rows 
    R_paths = np.delete(R_paths, (0), axis=0) 
    record_paths = np.delete(record_paths, (0), axis=0) 
    
    return R_paths, record_paths, exception

# The purpose of this function is to generate L random paths of length N 
# for the Cox-Ingersoll interest rate model. This returns the interest rate paths
# and record the number of times the interest falls below epsilon
def CIR(N,L):
    global a,b,s,t,epsilon,delta
    Ro = 1
    R_paths = np.zeros((1,N+1)) # this holds all rate paths 
    record_paths = np.zeros((1,N+1))  # This is a log of times where the rate went below a minimum epsilon
    exception = 0 # This tracks the total number of exceptions over all paths
    
    for j in range(L):# The outer for loop generates the number of paths specified
        
        R = np.array([Ro]) # R is a specific path being generated
        record = np.zeros((N+1)) # This hold a record of all exceptions for the specified path
        for i in range(N):  # This for loop determines the length of each path
            dW_tilde = np.random.normal(0,np.sqrt(delta),1) # at each iteration generate 1 number from a gausian distribution with mean 0 and st dev delta t
            
            next_rate = R[i] + (a[i]-b[i])*R[i]*delta + s[i]*np.sqrt(R[i])*dW_tilde # computes the next rate
            
            if next_rate < epsilon: # if the next rate goes below epsilon then set the next_rate to epsilon
                exception = exception +1 
                next_rate = epsilon
                record[i] = t[i] # this will replace the value of the record where the exception occurs
                
            R = np.append(R,next_rate )  # append the next rate to the path      
        R_paths = np.vstack((R_paths,R)) # appends the rate path to R_paths
        record_paths = np.vstack((record_paths,record)) # appends the record path to record_paths
    
    # removes initial rows 
    R_paths = np.delete(R_paths, (0), axis=0) 
    record_paths = np.delete(record_paths, (0), axis=0)
    return R_paths,record_paths, exception
    
    
    
    
# Parameters
N =250 # length of walk
L = 1000 # number of paths 
alpha = 0.1/N
sigma = alpha
r = alpha/3
T = 1 # end time
t = (T*np.arange(N+1)/N) # Creates a time vector from 0 to T = 1
delta = T/N # delta is constant

epsilon = 0.01
Kb = 1
Ks = 1.5
Ka = 1.2
pi = np.pi
b = Kb*(1.1 + np.sin(t*pi/T))
s = Ks*(1.1 + np.cos(4*t*pi/T))
a = 0.5*(s**2) + Ka*(1.1 + np.cos(t*pi/T))

print('The Hull White model: dR = (a-b*R)dt + s*dW~ is not a martingale since it contains a drift dt term')
print('------------------------------------Part A ----------------------------')
print('The following PDEs were generated from the interest rate model SDEs for HW and CIR using Feynmann-Kac theorem.')
print('Hull-White')
print('g(t,R) = E(h(R)) = E(R)')
print('g_t + (a-b*R)g_R + 0.5*(s^2)*g_RR = 0')

print('Cox-Ingersoll-Ross')
print('g(t,R) = E(h(R)) = E(R)')
print('g_t + (a-b*R)g_R + 0.5*(s^2)*R*g_RR = 0\n')

#------------------------------------Part B -----------------------------------
plt.figure(1)
plt.plot(t,b, label = 'b')
plt.plot(t,a, label = 'a')
plt.plot(t,s, label = 's')
plt.plot(t,0.5*s**2,label='0.5*s^2')
plt.xlabel('t')
plt.title('Parameters')
plt.legend()

plt.figure(2)
R_HW,record_HW,exception = Hull_White(N,10)
plt.plot(t.T,R_HW.T)
plt.xlabel('t')
plt.ylabel('R(t)')
plt.title('Question2 Part b: Superimposed Rates for Hull White Model')

plt.figure(3)
R_CIR,record_CIR,exception = CIR(N,10)
plt.plot(t.T,R_CIR.T)
plt.xlabel('t')
plt.ylabel('R(t)')
plt.title('Question2 Part b: Superimposed Rates for CIR Model')
print('Both models have the same general path which can be seen easier in the expectation values.')
print('Each graphs displays two humps')

#------------------------------------Part C -----------------------------------


R_HW,record_HW,exception_HW = Hull_White(N,L)
R_CIR,record_CIR,exception_CIR = CIR(N,L)

R_HW_E = np.mean(R_HW,axis =0) # This is the average path from N/2 to N 
R_CIR_E = np.mean(R_CIR,axis =0) # This is the average path from N/2 to N 

R_HW_Var = np.var(R_HW,axis =0)
R_CIR_Var = np.var(R_CIR,axis =0)

plt.figure(4)
plt.plot(t,R_HW_E, label = 'HW')
plt.plot(t,R_CIR_E, label = 'CIR')
plt.xlabel('t')
plt.ylabel('R(t)')
plt.title('Question 2 Part C: Expectations')
plt.legend()

plt.figure(5)
plt.plot(t,R_HW_Var, label = 'HW')
plt.plot(t,R_CIR_Var, label = 'CIR')
plt.xlabel('t')
plt.ylabel('R(t)')
plt.title('Question 2 Part C: Variance')
plt.legend()

#------------------------------------Part D -----------------------------------
print('For epsilon = ',epsilon)
print('For Hull White ',exception_HW,' exceptions occured')
print('For Cox-Ingersoll-Ross ',exception_CIR,'excetions occured')

#------------------------------------Part B -----------------------------------

# this part is to find the symbolic formula for R(T) for the Hull White model
r,t,T,Kb,Ks,Ka = sp.symbols('r,t,T,Kb,Ks,Ka')

# Define parameters in terms of symbolic variables
pi = sp.pi
b = Kb*(1.1 + sp.sin(pi*t/T))
c = sp.integrate(b,(t,0,t)) # integrate b w.r.t to t from 0 to t
s = Ks*(1.1 + sp.sin(4*pi*t/T))
a = Ka*(0.5*(s**2) + 1.1*sp.cos(pi*t/T))

# Evaluate and compute Ro and Gamma
c_T = c.subs(t,T).evalf(2) # evaluate c by substiuting in T= 1 for t and rounding to 2 decimals
Ro = r*sp.exp(-c_T) + sp.integrate(sp.exp(-c_T)*a,(t,0,T)) 
Gamma = sp.exp(-2*c_T - c)*s

print('\nThe Symbolic values for Ro and Gamma are as follows:')
print('Ro =',Ro)
print('-----------')
print('Gamma = ',Gamma)
print('-----------')
print('R_HW = Ro + the of integral(Gamma) w.r.t to dW~ from 0 to T ')

