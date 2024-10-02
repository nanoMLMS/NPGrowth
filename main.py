import numpy as np
import utils
import sys
import dynamics

system = utils.System('CuNP.xyz')
parameters = utils.SimulationParameters('parameters.toml')

subsystem_indices = list(range(len(system.atoms)))

for i in range(50):
    radius = system.max_distance_from(system.center_of_mass(subsystem_indices))
    position = np.array([0, 0, radius + parameters.radius_offset]) + system.center_of_mass(subsystem_indices)
    system.add_atom(parameters.temperature, system.center_of_mass(subsystem_indices), position)

    count = 0
    while system.max_atom_separation() > 3.7:
        dynamics.langevin(system, parameters)
        count += 1
        if count > 10:
            system.view_trajectory()
            print('probably detached atom')
            sys.exit()
    print(i)

system.view_trajectory()