from ase import Atoms
import numpy as np
import utils
import dynamics

system = utils.System('CuNP.xyz')
parameters = utils.SimulationParameters('parameters.toml')

subsystem_indices = list(range(len(system.atoms)))

for i in range(50):
    position = system.center_of_mass(subsystem_indices) + utils.get_position(20)
    system.add_atom(parameters.temperature, system.center_of_mass(subsystem_indices), position)


    while system.max_atom_separation() > 3.7:
        dynamics.langevin(system, parameters, showProgress=False)
    print(i)

system.view_trajectory()