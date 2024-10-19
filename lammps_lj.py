from lammps import PyLammps

lmp = PyLammps()

# 1) Initialization
lmp.units('lj') # all quantities are unitless
lmp.dimension(3) # 3 dimensions
lmp.atom_style('atomic') # atoms are dot with mass
lmp.pair_style('lj/cut', 2.5) # cut off of lj potential (unitless)
lmp.boundary('p p p') # periodic boundary conditions in along three directions

# 2) System definition
import random
simulation_region_name = 'simulation_box'
different_type_of_atoms = 2
atom1_type = 1 # each atom has a numeric atom type associated
atom2_type = 2 # we have two different types of atoms

lmp.region(simulation_region_name, 'block', -20, 20, -20, 20, -20, 20)
lmp.create_box(different_type_of_atoms, 'simulation_box')
lmp.create_atoms(atom1_type, 'random', 5000, random.randint(1, 999999), simulation_region_name)
lmp.create_atoms(atom2_type, 'random', 5000, random.randint(1, 999999), simulation_region_name)

# 3) Simulation settings
lmp.mass(atom1_type, 1)
lmp.mass(atom2_type, 1)
lmp.pair_coeff(atom1_type, atom1_type, 10.0, 1.0) # sets lj coefficients for interactions between atoms of type 1. (energy param 11, distance param 11)
lmp.pair_coeff(atom2_type, atom2_type, 10.0, 1.0) # same for atoms of type 2.
lmp.pair_coeff(atom1_type, atom2_type, 0.05, 1.0)

# 4) Visualization
dump_name = 'minimizationdmb'
dump_file = 'minimization.lammpstrj'
write_interval = 10
print_interval = 10
lmp.thermo(print_interval) 
lmp.thermo_style('custom', 'step', 'temp', 'pe', 'ke', 'etotal', 'press')
lmp.dump(dump_name, 'all', 'atom', write_interval, dump_file)

# 5) Run
max_number_iterations = 1000
lmp.minimize(10**-4, 10**-6, max_number_iterations, 10000) # energy minimization

# MOLECULAR DYNAMICS
# 4) Visualization
dump_name = 'langevindmp'
dump_file = 'dump.lammpstrj'
write_interval = 100
print_interval = 50
lmp.thermo(print_interval)
lmp.dump(dump_name, 'all', 'atom', write_interval, dump_file)

# 5) Run
nph_fix_name = 'mynph'
langevin_fix_name = 'mylgv'
timestep = 0.005 # unitless because of the lj units
steps = 100000
lmp.fix(nph_fix_name, 'all', 'nph', 'iso 1.0 1.0 1.0') # updates positions and velocities of the atoms at every step
lmp.fix(langevin_fix_name, 'all', 'langevin', 1.0, 1.0, 0.1, random.randint(1, 999999)) # langevin thermostat
lmp.timestep(timestep)
lmp.run(steps)

