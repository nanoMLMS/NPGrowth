from lammps import PyLammps

# Initialize PyLammps
lmp = PyLammps()

# 1) Initialization
lmp.units('metal')
lmp.atom_style('atomic')
lmp.boundary('s s s')

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
lmp.group('langevin_atoms', 'id', '1', str(len(lmp.atoms)))

# Simulation settings
lmp.mass(1, 63.546)
lmp.pair_style('eam')
lmp.pair_coeff('* * Cu_u3.eam')

lmp.neighbor(2.0, 'bin')  # Define neighbor skin distance. Useful for efficiency, need to check
lmp.neigh_modify('every', 1, 'delay', 0, 'check', 'yes')  # Control neighbor list updating

# Step 4: Remove initial linear momentum from the entire system if present
lmp.velocity('langevin_atoms', 'zero', 'linear')

# Step 5: Remove rotational momentum to avoid system rotation
lmp.fix('momentum_fix', 'langevin_atoms', 'momentum', '1', 'linear', '1 1 1', 'angular', 'rescale')

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
lmp.minimize(1.0e-4, 1.0e-6, 1000, 10000) # minimize

lmp.fix('mynve', 'all', 'nve') # updates positions and velocities of the atoms at every step
lmp.fix('mylgv', 'langevin_atoms', 'langevin', 3000.0, 3000.0, 0.1, random.randint(1, 999999)) # langevin thermostat
lmp.timestep(0.001)
# lmp.run(10000)

# ------------------------------------------------------------------------------------
# add atom
# ------------------------------------------------------------------------------------

new_atom_position = [0, 0, 0]

# Combine the x, y, and z coordinates into a single array
positions = np.array([lmp.atoms[i].position for i in range(len(lmp.atoms))])
positions = np.append(positions, [new_atom_position], axis=0)
min_coords = positions.min(axis=0)
max_coords = positions.max(axis=0)

# Expand the simulation box so to include newly added atom
lmp.change_box('all', 
             'x', 'final', min_coords[0], max_coords[0], 
             'y', 'final', min_coords[1], max_coords[1], 
             'z', 'final', min_coords[2], max_coords[2],
             'boundary', 'f f f')

# Add a new atom outside the original box
lmp.create_atoms(1, 'single', new_atom_position[0], new_atom_position[1], new_atom_position[2])

# Create a group for the newly added atom
lmp.group('new_atom', 'id', str(len(lmp.atoms)))  # The 'next' keyword assigns the last added atom to 'new_atom'

# Assign an initial velocity to the newly added atom
initial_velocity = [0.0, 0.5, 0.5]  # Example velocity in x, y, z
lmp.velocity('new_atom', 'set', *initial_velocity)

lmp.dump("force_dump", "new_atom", "custom", 10, "forces_output.txt", "id", "fx", "fy", "fz")

lmp.change_box('all', 'boundary', 's s s')

# Continue the simulation after adding the atom
lmp.run(50000)