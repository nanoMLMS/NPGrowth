import utils

import dynamics
from ase.visualize import view

from asap3.io.trajectory import Trajectory # Used trajectory from asap3 module

system = utils.System('CuNP.xyz')

parameters = utils.SimulationParameters('parameters.toml')

trajectory = Trajectory("output.traj", 'w', system.atoms)

dynamics.langevin(system, parameters, trajectory, showProgress=True)

trajectory.close()
trajectory = Trajectory('output.traj', 'r')
view(trajectory)
trajectory.close()