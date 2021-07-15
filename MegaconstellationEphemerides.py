#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to create a dataset of positions for a mega-constellation of satellites in orbit.
    
This code uses the commuity-developed AstroPy and Poliastro packages:

https://www.poliastro.space/
https://www.astropy.org/

INPUT: Constellation design, timestep, duration
OUTPUT:  Panda dataframe with orbital elements and cartesian positions and velocities for a constellation of satellites over the specified timeframe


Notes:
>Code originally inspired by the STK simulation in A. U. Chaudhry and H. Yanikomeroglu, ‘Laser Intersatellite Links in a Starlink Constellation: A Classification and Analysis’, IEEE Vehicular Technology Magazine, vol. 16, no. 2, pp. 48–56, Jun. 2021, doi: 10.1109/MVT.2021.3063706.
>The Starlink constellation parameters were updated based on an FCC filing dated April 7th 2020


LICENSE INFORMATION
Copyright 2021 Calum Turner

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


#----------------------------------------------#
#--------------------SETUP---------------------#
#----------------------------------------------#

#Import necessary packages

#Basics
import numpy as np
import math
import pandas as pd
import itertools as it

#Astrodynamics
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody import propagation


#----------------------------------------------#
#----------------SET PARAMETERS----------------#
#----------------------------------------------#

"""
The parameters of the simulation are set up, including timestep, 
total time to run, and the parameters of the constellation
"""

#Megaconstellation parameters — for a single shell. Currently produces Starlink Phase 1.

altitude = 550 #km
i = 53 #inclination in degrees
number_planes = 72
satellites_per_plane = 22
e = 0.0 #eccentricity


number_agents = int(number_planes * satellites_per_plane) #Number of satellites
plane_spacing = 360/24 #spacing between orbital planes in degrees
satellite_spacing = 360/satellites_per_plane #spacings between satellties in a plane in degrees
omega = 0 #argument of periapsis in degrees (irrelevant in case of circular orbit)


#Simple calculations assuming circular orbits, used to detrmine simulation duration
r_E = 6378 #Earth radius in km
mu = 3.986e5 #Gravitational parameter for the Earth (km^3 per s^2)
a = altitude + r_E #Semimajor axis = altitude + r_E for simple circular orbit (km)
#n = math.degrees((math.sqrt(mu/a**3))) #mean motion (degrees per second)
T = (2*math.pi/math.sqrt(mu))*a**(3/2) #orbital period in seconds

#Set up simulation parameters
time = 1.0*T #Choose how many orbital periods to run simulation for.
dt = 10 #timestep in seconds
number_Timesteps = int(time/dt)+1 #number of timesteps
print("The simulation will run for " + str(number_Timesteps) + " timesteps.")


#----------------------------------------------#
#-----------------CREATE DATA------------------#
#----------------------------------------------#

"""
The orbits are created and propagated with Poliastro, 
and stored in a multidimensional Numpy array
"""

#Create Numpy data cube with format: plane_number, satellite_number, time, a, e, i, RAAN, argP, nu
satellite_Data_cube = np.zeros((number_Timesteps,number_agents,15),dtype=object) #Set up data cube for swarm positions and data

#Declare arrays of plane and satellite numbers
PlaneNos = np.array(list(it.chain.from_iterable(it.repeat(x, satellites_per_plane) for x in range(1,number_planes+1))))
SatNos = np.array(list(it.chain.from_iterable(it.repeat(range(1,satellites_per_plane+1),number_planes))))
        
#add fixed values — plane numbers, satellite numbers, constant orbital parameters, and timesteps— to the data cube
for j in range(0,number_Timesteps):
    satellite_Data_cube[j,:,0] = PlaneNos #Add plane numbers to all rows in the data cube
    satellite_Data_cube[j,:,1] = SatNos #Add satellite numbers to all rows in the data cube
    satellite_Data_cube[j,:,2] = j*dt #Add timestamp to all rows in the data cube
    satellite_Data_cube[j,:,3] = a #Add semi-major axis
    satellite_Data_cube[j,:,4] = e #Add eccentricity
    satellite_Data_cube[j,:,5] = i #Add inclination

#Add initial orbital parameters
for j in range(0,number_agents):
    satellite_Data_cube[0,j,6] = (satellite_Data_cube[0,j,0]-1)*plane_spacing #Right ascension of ascending node
    satellite_Data_cube[0,j,8] = ((satellite_Data_cube[0,j,1]-1)*satellite_spacing) #Initial true anomolies


#Quick internal function to return a Poliastro two body orbit from a slice of values with format a,e,i,raan,argp,nu
def OrbitCreator(dataslice):
    semimajor = dataslice[0]* u.km
    ecc = dataslice[1] * u.one
    inc = dataslice[2] * u.deg
    raan = dataslice[3] * u.deg
    argp = dataslice[4] * u.deg
    nu = dataslice[5] * u.deg
    orb = Orbit.from_classical(Earth, semimajor, ecc, inc, raan, argp, nu)
    return orb


#Create a list of orbit objects for each satellite
Orbits = [OrbitCreator(satellite_Data_cube[0,j,3:9]) for j in range(0,number_agents)]

#Create array of times with units
Timestamps = np.arange(0,time,dt) * u.second

#Propagate each Orbit using Poliastro and write data to the array
for i in range(0,number_agents):
    [Position,Velocity] = propagation.vallado(Earth[1], Orbits[i].r, Orbits[i].v, Timestamps)
    satellite_Data_cube[:,i,9:12] = Position.value
    satellite_Data_cube[:,i,12:15] = Velocity.value
    satellite_Data_cube[:,i,8] = [math.degrees(Orbits[i].n.value)*satellite_Data_cube[j,i,2]+satellite_Data_cube[0,i,8] for j in range(0,number_Timesteps)]
    print("Propagated orbit for satellite " + str(i) + " of " + str(number_agents))
    
    
#----------------------------------------------#
#------------------WRAP DATA-------------------#
#----------------------------------------------#

"""
With the positional data created and placed in a multidimensional 
Numpy arrary, it is wrapped in a nice human-readable panda dataframe.
The dataframe contains extra information to label the satellties 
and make later analysis easier.
"""

#Reshape date cube to a 2d array in order to add to a pandas dataframe
satellite_Data_array = satellite_Data_cube.reshape((number_Timesteps*number_agents,15),order='F')

#Set up unique identifiers for each satellite, in the format sXXYYY where XX is plane number and YYY is satellite number.
planes = ['s' + str(x).zfill(2) for x in list(range(1,number_planes+1))]
satellites = [str(x).zfill(3) for x in list(range(1,satellites_per_plane+1))]
SatelliteIDs = [''.join(pair) for pair in list(it.product(planes, satellites))]


#Declare a list of timesteps
Timesteps = np.arange(0,number_Timesteps,1)

#Declare a list of timestamps
Timestamps_array = np.arange(0,time,dt)

#Declare a multilevel index for the dataframe 
SatelliteMultiIndex = pd.MultiIndex.from_product([SatelliteIDs,Timestamps_array],names=['Satellite ID','Time'])

#Declare a list of labels for the columns of the dataframe
labels = ['Plane No.', 'Satellite No.', 'Time', 'Semimajor axis', 'Eccentricity', 'Inclination', 'Right Ascension of Ascending Node','Argument of Perigee','True Anomoly','X-position','Y-position','Z-position','X-velocity','Y-velocity','Z-velocity']

#Declare a Dataframe
ConstellationEphemerides = pd.DataFrame(satellite_Data_array, columns = labels,index = SatelliteMultiIndex)

#Export Dataframe to a CSV file in a .zip archive
print("Final Dataframe created. Exporting.")
compression_opts = dict(method='zip', archive_name='ConstellationEphemerides.csv')  
ConstellationEphemerides.to_csv('ConstellationEphemerides.zip', index=True, header=True, compression=compression_opts)  
 
















