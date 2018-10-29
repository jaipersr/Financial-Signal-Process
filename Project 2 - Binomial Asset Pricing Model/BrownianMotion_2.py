# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:23:18 2018
Ryan Jaipersaud
Brownian Motion
ECE 478 Financial Signal Processing
The following code uses an asymmetrix walk to generate L paths of length N.
This code looks at two the cases of one-boundary levels and two-boundary level.
In the case of one boundary reflection only occurs once and several plots of
m = 0.5, 1, 2 are shown. In the case of two boundary relfection occurs each time
the path tries to cross one of the boundaries. 5 plot at a = -1 and b = 1 are 
superimposedo n one plot. Expectation of reaching the boundary for the first time 
and the probability of the path being within the boundary in either 
case are calculated at N = 1000 and L = 1000. The expected number of reflection is 
determined for the two boundary case.
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from pandas import DataFrame
from copy import deepcopy



# This is a symmetric random walk where the probability of choosing heads or tails
# is 1/2 and thus probability doesn not need to be a parameter passed to the function
def random_walk(N,L):
    
    W_path = np.zeros((N+1,1)) # initializes the matrix that holds all paths

    feasible_set = ['H','T'] # This set reflects the probability of choosing heads and tails p = 0.5
    
    for i in range(L): # for each path
        x_path = np.array([]) # initialize the path of the random walk
        for i in range(N): # for each step
            step = random.choice(feasible_set) # randomly choose heads or tails
            if step == 'H':
                x_path = np.append(x_path,1) # Append 1 to the path if heads was chosen
            else:
                x_path = np.append(x_path,-1) # append -1 to the path if tails was chosen
                
        x_path = np.array([x_path]).T # This turn x from a row to column vecotr to prevent problems later on
        M_path = np.cumsum(x_path) # M is the cumulative sum of X 
        M_path = np.insert(M_path,0,0) # appends zero to beginning of list
        
        w_path = (1/np.sqrt(N))*M_path # formula for the specific path
        w_path = np.array([w_path]).T  # creates column vector
        W_path = np.hstack((W_path,w_path)) # add the brownian motion to the matrix containing all brownian motions where ech column represent a specific path
        
    W_path = np.delete(W_path,0,axis = 1) # removes initial column

    return  W_path # return all paths (N + 1) by L

# This takes in a single path w_path and a level m and calculates a sinlge reflection
# when the path cross a level m and retrun both the original path, the modified path
# the time at which the reflection occured and the time vector
def one_boundary(w_path,m):

    N = w_path.shape[0] - 1 # The - 1 is to remove the space the 0 index takes up
    k = np.arange(N+1) # createa a k vector from 0 to N+1
    T = 1
    t = (T*k/N) # Creates a time vector from 0 to T = 1

    w_path = np.array([w_path]).T # turn row into column
    w_original = deepcopy(w_path) # creates deepcopy such that changes to w_path don't change w_original
    stop_time = 0
    for i in range(w_path.shape[0]): # for each step in the path
        # this case is for when the stop time is reached
        if w_path[i,0] > m:
            stop_time = t[i] # sets the time index equal to the stop time when the w_path crosses m
            epsilon = w_path[i,0] - m # calculate difference from m
            w_path[i,0] = w_path[i,0] - 2*epsilon # reflect over m
            break # after the first reflection break out of for loop
            
    if stop_time == 0: # if by the end of looping through the path no reflection has occurred
        stop_time = math.inf # set the stop time to infinity
            
    return stop_time, w_path,w_original,t

# This takes in a single path w_path and a level m and calculates multiple reflections
# when the path cross a level m and retrun both the original path, the modified path
# the time at which the first reflection occurs and the time vector
def two_boundary(w_path,a,b):

    N = w_path.shape[0] - 1 # The - 1 is to remove the space the 0 index takes up
    k = np.arange(N+1) # createa a k vector from 0 to N
    T = 1
    t = (T*k/N) # Creates a time vector from 0 to T = 1
 
    w_path = np.array([w_path]).T # turn row into column
    w_original = deepcopy(w_path) # creates deepcopy such that changes to w_path don't change w_original
    stop_time = 0 # this will store the stop time of the first reflection only
    reflections = 0 # this will hold the number of reflections when the path crosses a and b
    
    for i in range(w_path.shape[0]): # for each step in the path
       
        if w_path[i,0] > b: # check for a reflection over b
            if stop_time == 0: # if the stop_time has not been changed
                stop_time = t[i] # set the stop_time equal to the first relflection time
            reflections = reflections + 1 # increment count because a reflection over b will occur
            epsilon = w_path[i,0] - b # calculate difference from b
            w_path[i,0] = w_path[i,0] - 2*epsilon # reflect over b

        if w_path[i,0] < a: # check for a reflection over a
            if stop_time == 0:  # if the stop_time has not been changed
                stop_time = t[i] # set the stop_time equal to the first relflection time
            reflections = reflections + 1 # increment count because a reflection over b will occur
            epsilon = a - w_path[i,0]  # calculate difference from a
            w_path[i,0] = w_path[i,0] + 2*epsilon # reflect over a
            
    if stop_time == 0: # if by the end of looping through the path no reflection has occurred
        stop_time = math.inf # set the stop_time to inf
       
    return stop_time, w_path,w_original,t, reflections

# This function is for testing various numbers of path and steps as well as boundaries
# Plots of the paths can also be generated as functions of time
def testing(m,N,L,plot = 'off',figure_number = 0,option = 'one_boundary',a =0,b =0):
    
    total_time = 0 # hold the cumulative time build up for expectation
    count = 0 # iterator to count the number of times the stop time is  not reached ( equal inf)
    total_reflections = 0 # counts the total number of reflection from the two-boundary function
    
    W = random_walk(N,L) # randomly generates the paths where each column is a path

    for i in range(W.shape[1]): # for each path in W
        if option == 'one_boundary':
            # find the first time it takes to hit m, the modified path, the original path, and the time vector
            first_time,w_path,w_original,t = one_boundary(W[:,i],m) 
        if option == 'two_boundary':
            # find the first time it takes to hit either a or b, the modified path, the original path, the time vector, and the number of times reflection occurred
            first_time,w_path,w_original,t,reflections = two_boundary(W[:,i],a,b)
            total_reflections = total_reflections + reflections
        
        # counts the number of times the stop time is at infinity which happens 
        # if the path doesn't cross m for one_boundary case or a & b for two boundary case
        if first_time == math.inf: 
            count = count + 1
        else:
            total_time = total_time + first_time  # stores the cumulative first time if the path does cross m or a & b
    
    # This will plot the last path modified and original path in W
    # This is used to plot sample path so in the case where L = 1 the last path
    # is the only path.
    if plot == 'on':
        plt.figure(figure_number) # plots on the specified figure number parameter
        plt.plot(t,w_original,'r',label = 'original')
        plt.plot(t, w_path,'b', label = 'reflected')
        
        if option == 'one_boundary':
            m = m* np.ones((t.shape[0],1)) # creates a vector of the level
            plt.title('Part D: 3 Sample Brownian Paths N = 100')
            plt.plot(t, m,'k', label = 'm')
            #plt.legend()
           
        elif option == 'two_boundary':
            a = a* np.ones((t.shape[0],1)) # creates a vector of the lower level
            b = b* np.ones((t.shape[0],1)) # creates a vector of the upper level
            plt.title('Part E Part 1: Superimposed Brownian Paths N = 100')
            plt.plot(t, a,'k')
            plt.plot(t, b,'k')
            
        plt.ylabel('W(t)')
        plt.xlabel('t')
              
    p = count/L  # probability of the stop time being at infinity, meaning the path stayed within bounds for either one or two boundaries
    expected_time = total_time/L # expectation value of the first time to a reflection
    
    if option == 'two_boundary':
        expected_reflections = total_reflections/L # calculates average total reflections
        return p, expected_time,expected_reflections
    
    return p, expected_time

# This part calculates expected probabilites and expected first passage times
N = 1000 # Number of steps/ flips in a single path
L = 1000 # Number of paths
m1 = 0.5
p1, expected_time_1 = testing(m1,N,L)
m2 = 1
p2, expected_time_2 = testing(m2,N,L)
m3 = 2
p3, expected_time_3 = testing(m3,N,L)

m = np.vstack((m1,m2,m3))
p = np.vstack((p1,p2,p3))
expectation = np.vstack((expected_time_1,expected_time_2,expected_time_3))

X = np.hstack((m,p,expectation))
df = DataFrame(data = X, columns = ['m','probabilities', 'expectations'])
print('-----------------------Part C----------------------------')
print('For N = ', N, 'and L =', L )
print(df)



print('-----------------------Part D----------------------------')
N = 100
L = 1 # Sample path
print('For all graphs used the black line refers to the boundary.')
print('The red line represents the original path')
print('The blue line represents the reflected path')
print('All Graphs were generated using N = ', N,' and L = ', L)
# This makes plots of brownian motion paths for only one reflection
for i in range(3):
    p1, expected_time_1 = testing(m1,N,L,'on',1)
    p2, expected_time_2 = testing(m2,N,L,'on',2)
    p3, expected_time_3 = testing(m3,N,L,'on',3)

print('-----------------------Part E Part 2 & 3----------------------------')
# This makes plots of brownian motion paths for only multiple reflections
for i in range(5):
    p1, expected_time_1,expected_reflections = testing(m1,N,L,'on',4,'two_boundary',-1,1)


# This calculates average values for the case with two strict boundaries
N = 1000
L = 1000
p, expected_time,expected_reflections = testing(m1,N,L,'off',5,'two_boundary',-1,1)

print('For N = ', N ,' and L = ', L, ':')
print('The probability that the path stays within the range {-1 < W(t) < 1} is p = ', p)
print('When the path did not stay in the range, the expected_time to reach the first reflection is t = ', expected_time)
print('When the path did not stay in the range, the expected number of reflection is ', expected_reflections)



