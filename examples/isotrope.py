import numpy as np
from numpy import random
import sys
import NPGrowth
import NPGrowth.functions

parameters_filename = sys.argv[1]

system = NPGrowth.System(parameters_filename) # Load system with parameters in parameters_filename

system.run(5000) # Run some steps to termalize system

# Nedded for later use
seed_center_of_mass = system.get_center_of_mass() 

cutoff = system.get_cutoff_from_log()
for _ in range(50000):
    radius = system.get_further_atom(seed_center_of_mass) + cutoff
    position = [random.normal(), random.normal(), random.normal()]
    norm = np.linalg.norm(position)
    position = position/norm
    position = radius * position
    position += seed_center_of_mass
    system.depo([position], [seed_center_of_mass])
    system.run(200)