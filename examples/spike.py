import copy
import numpy as np
import sys
import NPGrowth
import NPGrowth.functions

parameters_filename = sys.argv[1]

system = NPGrowth.System(parameters_filename) # Load system with parameters in parameters_filename
parameters = NPGrowth.SimulationParameters(parameters_filename)

system.run(parameters.termalize_steps) # Run some steps to termalize system

# Nedded for later use
seed_center_of_mass = system.get_center_of_mass() 

cutoff = system.get_cutoff_from_log()
for _ in range(parameters.n_depo_repeat):
    radius = system.get_further_atom(seed_center_of_mass) + cutoff
    positions = [[radius, 0, 0], [-radius, 0, 0], [0, radius, 0], [0, -radius, 0], [0, 0, radius], [0, 0, -radius]]
    positions = [p + seed_center_of_mass for p in positions]
    system.depo(positions, [seed_center_of_mass for _ in range(6)])
    system.run(parameters.steps_to_next_depo)

system.run(2000)