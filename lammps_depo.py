from lammps import PyLammps

# Initialize PyLammps
lmp = PyLammps()

# 1) Initialization
lmp.units('metal')
lmp.atom_style('atomic')
lmp.boundary('s s s') # fixed boundary

import tempfile
from ase.io import read, write
# 2) System definition
atoms = read('Cu675.xyz')

from scipy.spatial.distance import pdist
import numpy as np
positions = atoms.get_positions()  # Get atomic positions
distances = pdist(positions)  # Calculate all pairwise distances
max_diameter = np.max(distances)  # Find the max radius

atoms.set_cell([max_diameter, max_diameter, max_diameter])
atoms.center()

with tempfile.NamedTemporaryFile(delete=False) as tmp:
    write(tmp.name, atoms, format='lammps-data') # write atoms in LAMMPS data format

lmp.read_data(tmp.name) # load the LAMMPS data file into PyLammps

# Simulation settings
lmp.mass(1, 63.546)
lmp.pair_style('eam')
lmp.pair_coeff('* * Cu_u3.eam')

lmp.neighbor(2.0, 'bin')  # Define neighbor skin distance. Useful for efficiency, need to check
lmp.neigh_modify('every', 1, 'delay', 0, 'check', 'yes')  # Control neighbor list updating

# 4) Visualization
lmp.thermo(10)
lmp.thermo_style('custom', 'step', 'temp', 'pe', 'ke', 'etotal', 'press')
lmp.dump('dump1', 'all', 'atom', 10, 'dump.lammpstrj')

# Define LAMMPS variables for the thermodynamic quantities
lmp.variable("step_var", "equal", "step")
lmp.variable("temp_var", "equal", "temp")
lmp.variable("pe_var", "equal", "pe")
lmp.variable("ke_var", "equal", "ke")
lmp.variable("etotal_var", "equal", "etotal")
lmp.variable("press_var", "equal", "press")

# Use fix print to output thermo data to a file
lmp.fix('thermo_output', 'all', 'print', 10, '"${step_var} ${temp_var} ${pe_var} ${ke_var} ${etotal_var} ${press_var}"',
       'file', 'thermo_output.txt', 'screen', 'no', 'title', '"Step Temp PE KE Total_Energy Pressure"')

import random
# 5) Run
# A minimize
lmp.minimize(1.0e-4, 1.0e-6, 1000, 10000)

# B langevin
lmp.fix('mynve', 'all', 'nve') # updates positions and velocities of the atoms at every step
lmp.fix('mylgv', 'all', 'langevin', 0, 300.0, 0.1, random.randint(1, 999999)) # langevin thermostat
lmp.timestep(0.001)
lmp.run(10000)