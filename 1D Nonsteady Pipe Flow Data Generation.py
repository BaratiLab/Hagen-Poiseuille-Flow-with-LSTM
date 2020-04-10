#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Nina Prakash
# CMU Mechanical and AI Lab
# Professor Farimani
# June 2019

# Generate dataset of 1D non-steady Hagen-Poiseuille flow in a circular pipe.
# Assume constant pressure gradient and fluid starting from rest.

# references: "On Exact Solutions of the Navier-Stokes Equations for Uni-Directional Flows" (F. Lam, 2015)
#             Batchelor, George Keith. An introduction to fluid dynamics. Cambridge university press, 2000.
#             https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os



# global variables

numPoints = 16
dt = .25 # s
ti = 0 # s
tf = 5 # s
# rho = 888      # of oil at 20 degrees C [kg/m^3], 0.001 of water
# mu = 0.8       # of oil at 20 degrees C [kg/m*s], 998 of water
# nu = mu/rho
# dpdx = -10    # axial pressure gradient [Pa/m = kg/m^2*s^2], should be negative

Ds = np.linspace(0.5,1,num=15)
dpdxs = np.linspace(-11,-9,num=15)
mus = np.linspace(0.7,0.9,num=15)
nus = np.linspace(0.0008,0.0010,num=15)


# function to generate series of velocity profiles given boundary condition (diameter)

def generateProfiles(D,ti,tf,dt,dpdx,mu,nu,numPoints=16):
    # inputs: diameter, number of velocity points measured per profile
    # output: 2D list of velocity profiles (one for each time step) and 1D position vector
    
    R = D/2
    
    numProfiles = int((tf - ti)/dt)
    
    # set up an empty 2D array to populate
    profiles = np.zeros((numProfiles,numPoints+1))
    
    i = 0
    
    for t in np.arange(ti,tf,dt):
        # generate a velocity profile for each time step
        
        g = -dpdx
        
        r = []
        interval = D/numPoints
        x = int(numPoints/2)
        for j in range(-x,x+1):
            r.append(interval*j)

        u = np.zeros(numPoints + 1)
        for k in range(len(r)):
            S = 0
            for n in range(1,101):
                S += 1/(lambdan(n))**3 * J0(lambdan(n)*r[k]/R) / J1(lambdan(n)) * np.exp( -(lambdan(n))**2*nu*t/R**2 )
            v = g/(4*mu) * (R**2 - (r[k])**2) - 2*g*R**2/mu * S
            u[k] = v
        
        profiles[i] = u
        
        i += 1
    
    return profiles, r



# main data generation function

def generateData(Ds,ti,tf,dt,dpdxs,mus,nus,numPoints):
    print('generating data')
    # inputs:
    #    Ds: list of possible diameters
    #    ti: initial time (should be 0)
    #    tf: end time
    #    dt: time step
    #    dpdx: axial pressure gradient, should be negative
    #    numPoints: number of data points per velocity profile
    
    numTimeSteps = int((tf-ti)/dt)
    
    BCs = itertools.product(Ds,dpdxs,mus,nus)
    
    BCs = list(BCs)
    numBCs = len(BCs)
    
    inputs = BCs
    
    # initialize X and target arrays
    
    # print('size of inputs: ', np.shape(inputs))
    target = np.zeros((numBCs,numTimeSteps,numPoints+1))
    # print('size of target: ', np.shape(target))

    # calculate velocity profile for each boundary condition
    i = 0
    for bc in BCs:
        if i%10 == 0:
            print(i)
        D = bc[0]
        dpdx = bc[1]
        mu = bc[2]
        nu = bc[3]
        profiles,r = generateProfiles(D,ti,tf,dt,dpdx,mu,nu,numPoints)
        target[i] = profiles
        i += 1
    
    return inputs,target, numTimeSteps, numBCs
        

# helper functions to generate data

def lambdan(n):
    B = beta(n)
    return B + 1/(8*B)

def beta(n):
    return (n - 1/4)*np.pi
        
def J0(x):
    return special.jv(0,x)

def J1(x):
    return special.jv(1,x)



# function call and save

inputs, target, numTimeSteps, numBCs = generateData(Ds,ti,tf,dt,dpdxs,mus,nus,numPoints)
data_dir = 'data/'

data_session_name = 'all_params_%s' %(numBCs)

dataset_filename = os.path.join(data_dir,'dataset_'+data_session_name+'.npy')
bc_filename = os.path.join(data_dir,'bc_'+data_session_name+'.npy')

np.save(dataset_filename, target)
np.save(bc_filename, inputs)

print('dataset generated of size: ', np.shape(target))


# plot
'''
import numpy as np
import matplotlib.pyplot as plt

bc = np.load('data/bc_500_0.50 to 1.00_20 time steps.npy')
dataset = np.load('data/dataset_500_0.50 to 1.00_20 time steps.npy')
print(np.shape(dataset))

test_i = 200

numPoints = 16

D = bc[test_i]
r = []
interval = D/numPoints
x = int(numPoints/2)
for j in range(-x,x+1):
    r.append(interval*j)

plt.figure(1,dpi=600)
for i in range(len(dataset[test_i])):
    if i != 0:
        plt.plot(dataset[test_i][i],r,label='t = %s' %(i))
plt.title('Velocity Profile Development for Oil at 20 C')
plt.xlabel('velocity [m/s]')
plt.ylabel('r [m]')
plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
plt.savefig('dataset for 0.7m for paper.png')
'''
