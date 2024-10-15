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

from lammps import PyLammps

# create LAMMPS instance
L = PyLammps()

# PART A - ENERGY MINIMIZATION
# 1) Initialization
L.units('lj')
L.dimension(3)
L.atom_style('atomic')
L.pair_style('lj/cut', 2.5)
L.boundary('p p p')
# 2) System definition
L.region('simulation_box', 'block', -20, 20, -20, 20, -20, 20)
L.create_box(2, 'simulation_box')
L.create_atoms(1, 'random', 1500, 341341, 'simulation_box')
L.create_atoms(2, 'random', 100, 127569, 'simulation_box')
# 3) Simulation settings
L.mass(1, 1)
L.mass(2, 1)
L.pair_coeff(1, 1, 1.0, 1.0)
L.pair_coeff(2, 2, 0.5, 3.0)
# 4) Visualization
L.thermo(10)
# 5) Run
L.minimize(10**-4, 10**-6, 1000, 10000)

# PART B - MOLECULAR DYNAMICS
# 4) Visualization
L.thermo(50)
# 5) Run
L.fix('mynve', 'all', 'nve')
L.fix('mylgv', 'all', 'langevin', 1.0, 1.0, 0.1, 1530917)
L.timestep(0.005)
L.run(10000)

thermo_data = L.lmp.get_thermo("temp")  # Get the temperature
pressure_data = L.lmp.get_thermo("press")  # Get the pressure

# Output the retrieved thermo data
print("Final temperature:", thermo_data)
print("Final pressure:", pressure_data)

# explicitly close and delete LAMMPS instance (optional)
L.close()