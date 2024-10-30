from lammps import PyLammps
from ase.io import read, write
import tempfile
import numpy as np
from scipy.spatial.distance import pdist, cdist
import random
import utils.functions
import sys

force_treshold = 0.1
temperature = 100.0
initial_temperature = final_temperature = temperature
lattice_constant = 3.6150

class System:
    def __init__(self, filename):
        self.L = PyLammps()

        # 1) Initialization
        self.L.units('metal')
        self.L.atom_style('atomic') # atoms like points with position
        self.L.boundary('s s s') # the simulation box shrink in all three directions

        # 2) System definition
        atoms = read(filename)

        positions = atoms.get_positions()  # Get atomic positions
        distances = pdist(positions)  # Calculate all pairwise distances
        max_diameter = np.max(distances)  # Find the max radius

        atoms.set_cell([max_diameter, max_diameter, max_diameter])
        atoms.center()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            write(tmp.name, atoms, format='lammps-data') # write atoms in LAMMPS data format

        self.L.read_data(tmp.name) # load the LAMMPS data file into PyLammps
        self.L.group('initial_atoms', 'id', f'1:{str(self.L.atoms.natoms)}')

        # 3) Simulation settings
        self.L.pair_style('eam')
        self.L.pair_coeff('* * ./potentials/Cu_u3.eam')
        self.L.mass(1, 63.55)

        self.L.neighbor(2.0, 'bin')  # Define neighbor skin distance. Useful for efficiency, need to check what does exactly
        self.L.neigh_modify('every', 1, 'delay', 0, 'check', 'yes')  # Control neighbor list updating

        self.L.velocity('initial_atoms', 'zero', 'linear') # Set initial velocity to zero if present
        self.L.fix('momentum_fix', 'initial_atoms', 'momentum', '1', 'linear', '1 1 1', 'angular', 'rescale') # No rotation. Without rescale the system shouldn't rotate but it does. Need to recheck

        # 4) Visualization
        self.L.thermo(10)
        self.L.thermo_style('custom', 'step', 'temp', 'pe', 'ke', 'etotal', 'press')
        self.L.dump('dump1', 'all', 'atom', 20, 'dump.lammpstrj')
        
        # Define thermo variables
        self.L.variable("step_var", "equal", "step")
        self.L.variable("temp_var", "equal", "temp")
        self.L.variable("pe_var", "equal", "pe")
        self.L.variable("ke_var", "equal", "ke")
        self.L.variable("etotal_var", "equal", "etotal")
        self.L.variable("press_var", "equal", "press")

        # Set up fix print to log only thermo data to a file without command logs
        self.L.fix("thermo_output", "all", "print", "100",  # Print every 100 steps
                '"${step_var} ${temp_var} ${pe_var} ${ke_var} ${etotal_var} ${press_var}"',
                "file", "thermo_data.data", "screen", "no", "title",
                '"Step Temp PE KE Etotal Press"')

        # 5) Run algorithms
        self.L.minimize(1.0e-4, 1.0e-6, 1000, 10000)

        self.L.fix('mynve', 'all', 'nve') # updates positions and velocities of the atoms at every step
        
        self.L.fix('mylgv', 'initial_atoms', 'langevin', initial_temperature, final_temperature, 1, random.randint(1, 999999)) # langevin thermostat

        self.L.timestep(0.001) # 1 femtosecond. In metal units time is in picoseconds

    def depo(self, atom_positions, targets):
        if len(atom_positions) != len(targets):
            sys.exit(f'Positions number: {len(atom_positions)} must be equal to number of targets: {len(targets)}')

        positions = self.get_positions()
        positions = np.vstack((positions, atom_positions)) # Add the new atom positions to the already present atom's positions

        min_coords = positions.min(axis=0) # Min coordinate of the positions
        max_coords = positions.max(axis=0) # Max coordinate of the positions

        # Expand the simulation box so to include newly added atom and change the boundary to fixed (neccesary when creating new atoms)
        self.L.change_box('all', 
                    'x', 'final', min_coords[0], max_coords[0], 
                    'y', 'final', min_coords[1], max_coords[1],
                    'z', 'final', min_coords[2], max_coords[2],
                    'boundary', 'f f f')

        
        ids = [] # array that will contain the id of the newly added atoms
        for i, position in enumerate(atom_positions):
            self.L.create_atoms(1, 'single', position[0], position[1], position[2]) # Create atom
            velocity_direction = targets[i] - position
            velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            self.L.atoms[self.L.atoms.natoms - 1].velocity = velocity_direction * 50 # Assign initial velocity to the newly created atom with direction the center of mass
            ids.append(self.L.atoms[self.L.atoms.natoms - 1].id)

        self.L.change_box('all', 'boundary', 's s s') # Change to shrink boundary for the simulation box again

        # Position all the atoms exactly where they start to feel a force
        while any((np.linalg.norm(self.get_atom_from_id(id).force) < force_treshold) for id in ids): # While any of the atoms feels a force weaker than treshold
            self.L.run(1, 'pre no post no') # Evolve
            for id in ids: # Stop atom if it starts to feel force bigger than treshold
                atom = self.get_atom_from_id(id)
                if (np.linalg.norm(atom.force) > force_treshold):
                    atom.velocity = [0, 0, 0]


        two_radius = utils.functions.atomic_radius(lattice_constant) * 2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!1hardcoded lattice constant!!!!!!!!!!!!!!!!!!!!!
        max_dist = two_radius + (two_radius * 5 / 100)
        distances = {id: self.distance_from_system(self.get_atom_from_id(id).position) for id in ids}
        while any(distances[id] > max_dist for id in ids): # While any of the added atoms distance from system is bigger than treshold
            self.L.run(50) # !!!!!!!!!!!!!!!!!!!!!!!!hardcoded steps!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! evolve
            distances = {id: self.distance_from_system(self.get_atom_from_id(id).position) for id in ids}
            for id in ids:
                if distances[id] < max_dist: # Close enough to be considered deposited
                    ids = np.delete(ids, np.where(ids == id))
                    del distances[id]
                    self.L.group('initial_atoms', 'id', id) # Add atom to de initial atoms so that it will evolve with langevin algorithm

    def get_center_of_mass(self):
        positions = self.get_positions()
        masses = np.array([self.L.atoms[i].mass for i in range(self.L.atoms.natoms)])
        total_mass = np.sum(masses) # Calculate the total mass
        weighted_positions = np.sum(positions.T * masses, axis=1)  # Weighted sum of positions
        center_of_mass = weighted_positions / total_mass # Compute the center of mass
        return center_of_mass
    
    def get_atom_from_id(self, id):
        atom_ids = self.L.lmp.numpy.extract_atom('id', 0)
        index = np.where(atom_ids == id)[0][0]
        return self.L.atoms[index]
    
    def get_max_diameter(self):
        positions = self.get_positions()
        distances = pdist(positions)  # Calculate all pairwise distances
        max_diameter = np.max(distances)  # Find the max diameter
        return max_diameter
    
    def get_max_distance_between_atoms(self):
        positions = self.get_positions()
        pairwise_distances = cdist(positions, positions) # Compute pairwise distances
        np.fill_diagonal(pairwise_distances, np.inf) # Set diagonal (self-distances) to infinity so they don't interfere with finding the minimum
        min_distances = np.min(pairwise_distances, axis=1) # Find the minimum distance for each atom, excluding the self-distance
        return max(min_distances)
    
    def distance_from_system(self, position):
        distances = np.linalg.norm(self.get_positions() - position, axis=1) # Compute distances between position and all positions
        distances = distances[distances > 0] # Exclude the distance from itself if position atom of the system
        shortest_distance = np.min(distances)
        return shortest_distance
    
    def get_positions(self):
        positions = self.L.lmp.numpy.extract_atom("x", 3)  # "x" for position, 3 for the dimensionality (x, y, z)
        return positions
    
    def get_cutoff_from_log(self):
        self.L.run(0)
        cutoff = None
        with open('log.lammps', 'r') as log_file:
            for line in log_file:
                if "master list distance cutoff" in line.lower():
                    # Extract the cutoff value from the line
                    words = line.split()
                    cutoff = float(words[-1])  # The last word should be the cutoff value
                    break
        if not cutoff:
            sys.exit('no cutoff found')
        
        return cutoff

    def run(self, steps):
        self.L.run(steps)

    def get_thermo(self):
        step = []
        temp = []
        pe = []
        ke = []
        etotal = []
        press = []

        for run in self.L.runs:
            if run != self.L.runs[-1]:
                run.thermo.Step.pop()
                run.thermo.Temp.pop()
                run.thermo.PotEng.pop()
                run.thermo.KinEng.pop()
                run.thermo.TotEng.pop()
                run.thermo.Press.pop()

            step += run.thermo.Step
            temp += run.thermo.Temp
            pe += run.thermo.PotEng
            ke += run.thermo.KinEng
            etotal += run.thermo.TotEng
            press += run.thermo.Press

        return {
            'steps': step,
            'temp': temp,
            'potE': pe,
            'kinE': ke,
            'totE': etotal,
            'press': press
        }
    

# import timeit

# system = System('./seeds/Cu4631.xyz')
# n_times = 100
# execution_time = timeit.timeit(lambda: system.distance_from_system([100, 10, 10]), number=n_times)
# print(f'time to execute function: {execution_time/n_times}')