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
import dynamics
import numpy as np

system = utils.System('Cu675.xyz')
seed_center_of_mass = system.get_center_of_mass()

for i in range(50):
    angular_positions = [[np.pi / 2, np.pi/2 * i] for i in range(4)] # [theta, phi]
    directions = [seed_center_of_mass for i in range(len(angular_positions))]
    system.depo(angular_positions, directions)
    system.run(1000)