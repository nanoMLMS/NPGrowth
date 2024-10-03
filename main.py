import numpy as np
import utils
import sys
import dynamics

system = utils.System('CuNP.xyz')
parameters = utils.SimulationParameters('parameters.toml')

subsystem_indices = list(range(len(system.atoms)))

tips = 7
for i in range(parameters.n_atoms):
    radius = system.max_distance_from(system.center_of_mass(subsystem_indices))
    
    for j in range(tips):
            phi = (2 * np.pi / tips) * j
            position = utils.spherical_to_cartesian(radius + parameters.radius_offset, np.pi / 2, phi) + system.center_of_mass(subsystem_indices)
            system.add_atom(1.5 * parameters.temperature, system.center_of_mass(subsystem_indices), position)

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
system.save_trajectory('traj.xyz')