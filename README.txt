# ConstellationEpheremides
Python code to create a epheremides (a dataset of positions in orbit) for a mega-constellation of satellites in Low Earth Orbit.

This code was used to create the database hosted at https://ieee-dataport.org/open-access/simulated-megaconstellation-ephemerides

The link above included detailed documentation, but essentially this code creates a dataset providing the time-varying positions of 1584 satellites in a simulated megaconstellation modelled on Phase 1 of SpaceX's Starlink.

INPUT: Design of the constellation and the number of timesteps and temporal resolution required.
OUTPUT: .zip archive containing a .csv dataset with positions, velocities, and orbital elements of each satelltie at each timestep.

This code requires the AstroPy and Poliastro packages to run:

https://www.astropy.org/
https://www.poliastro.space/
