# import numpy as np
# import utils
# import sys
# import dynamics

# system = utils.System('Cu675.xyz')
# parameters = utils.SimulationParameters('parameters.toml')

# subsystem_indices = list(range(len(system.atoms)))

# parameters.steps = 1000
# dynamics.langevin(system, parameters)

# parameters.steps = 300
# radius = system.max_distance_from(system.center_of_mass(subsystem_indices))

# position = utils.spherical_to_cartesian(radius + parameters.radius_offset, np.pi / 2, 0) + system.center_of_mass(subsystem_indices)
# system.add_atom(1.5 * parameters.temperature, system.center_of_mass(subsystem_indices), position)

# dynamics.velocityVerlet(system, parameters)
# system.view_trajectory(viewer='ovito')
# system.save_trajectory('traj.xyz')

# from lamm

import utils

system = utils.System('Cu675.xyz')
parameters = utils.SimulationParameters('parameters.toml')

# Assign algorithms to different parts of the system
system.set_algorithm(range(675), utils.Algorithm.langevin)

# Evolve the system
system.evolve(parameters)