import sys
import toml
from ase import units

class SimulationParameters:
    """
    This class is responsible for loading and storing simulation parameters from a TOML file.

    Parameters:
    -----------
    filename : str
        The path to the TOML file containing the simulation parameters.

    Attributes:
    -----------
    n_atoms : int
        The number of atoms to be used in the simulation.
    radius_offset : float
        The radial offset used in the simulation.
    temperature : float
        The temperature for the molecular dynamics simulation.
    timestep : float
        The time step for the molecular dynamics simulation.
    steps : int
        The number of steps for the molecular dynamics simulation.
    write_interval : int
        The interval at which the simulation data will be written to a file.
    """

    def __init__(self, filename):
        self.__filename = filename

        parameters = self.__get()

        growth_params = parameters['growth']
        self.n_atoms = growth_params['n_atoms']
        self.radius_offset = growth_params['radius_offset']

        dynamics_params = parameters['dynamics']
        self.temperature = dynamics_params['temperature']
        # Why units.fs is not 10**-15
        self.timestep = dynamics_params['timestep'] * units.fs
        self.steps = dynamics_params['steps']
        self.write_interval = dynamics_params['write_interval']

    def __get(self):
        try:
            file = open(self.__filename, "r")
        except OSError:
            print("Can't read parameters from", self.__filename)
            sys.exit()

        with file:
            parameters = toml.load(file)
        
        return parameters
    
    def __check():
        # TODO check parameters
        print()
    
    def __str__(self):
        """
        Returns a readable string representation of the object for printing.
        """
        return (f"Simulation Parameters:\n"
                f"  Number of Atoms to add: {self.n_atoms}\n"
                f"  Radius Offset: {self.radius_offset} Å\n"
                f"  Temperature: {self.temperature} K\n"
                f"  Timestep: {self.timestep} fs\n"
                f"  Steps: {self.steps}\n"
                f"  Write Interval: {self.write_interval} steps\n"
                f"  Parameter File: {self.__filename}")

from enum import Enum

class Algorithm(Enum):
    langevin = "langevin"
    verlet = "verlet"


from lammps import PyLammps
from ase.io import read, write
import tempfile
import numpy as np
from scipy.spatial.distance import pdist, cdist
import random

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
        
        self.L.fix('mylgv', 'initial_atoms', 'langevin', 300.0, 300.0, 1, random.randint(1, 999999)) # langevin thermostat

        self.L.timestep(0.001) # 1 femtosecond. In metal units time is in picoseconds

    def depo(self, angular_positions, directions):
        if len(angular_positions) != len(directions):
            sys.exit(f'Angular positions number of elements: {len(angular_positions)} must be equal of number of elements of directions: {len(directions)}')

        radius = self.get_max_diameter() / 2 + self.get_cutoff_from_log() # Radius where the atoms that will be deposited will be placed initially
        atom_pos = [(spherical_to_cartesian(radius, angular_position[0], angular_position[1])) for angular_position in angular_positions]
        atom_pos = [pos + self.get_center_of_mass() for pos in atom_pos]
        positions = self.get_positions()
        positions = np.vstack((positions, atom_pos)) # Add the new atom positions to the already present atom's positions

        min_coords = positions.min(axis=0) # Min coordinate of the positions
        max_coords = positions.max(axis=0) # Max coordinate of the positions

        # Expand the simulation box so to include newly added atom and change the boundary to fixed (neccesary when creating new atoms)
        self.L.change_box('all', 
                    'x', 'final', min_coords[0], max_coords[0], 
                    'y', 'final', min_coords[1], max_coords[1],
                    'z', 'final', min_coords[2], max_coords[2],
                    'boundary', 'f f f')

        
        ids = [] # array that will contain the id of the newly added atoms
        for i, position in enumerate(atom_pos):
            self.L.create_atoms(1, 'single', position[0], position[1], position[2]) # Create atom
            velocity_direction = directions[i] - position
            velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            self.L.atoms[self.L.atoms.natoms - 1].velocity = velocity_direction * 50 # Assign initial velocity to the newly created atom with direction the center of mass
            ids.append(self.L.atoms[self.L.atoms.natoms - 1].id)

        self.L.change_box('all', 'boundary', 's s s') # Change to shrink boundary for the simulation box again

        treshold = 0.1
        # Position all the atoms exactly where they start to feel a force
        while any((np.linalg.norm(self.get_atom_from_id(id).force) < treshold) for id in ids): # While any of the atoms feels a force weaker than treshold
            self.L.run(1, 'pre no post no') # Evolve
            for id in ids: # Stop atom if it starts to feel force bigger than treshold
                atom = self.get_atom_from_id(id)
                if (np.linalg.norm(atom.force) > treshold):
                    atom.velocity = [0, 0, 0]


        two_radius = atomic_radius(3.6150) * 2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!1hardcoded lattice constant!!!!!!!!!!!!!!!!!!!!!
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
        atom = next((atom for atom in self.L.atoms if atom.id == id), None)
        return atom
    
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
        positions = np.array([self.L.atoms[i].position for i in range(self.L.atoms.natoms)])  # Get atomic positions
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

def thermal_velocity(mass, temperature):
    """
    Returns velocity of the thermal motion of particles
    at a given temperature
    """
    return np.sqrt((3 * units.kB * temperature) / mass)

def eigen_vector(position, target):
    """
    Returns eigen vector that points from position to target
    """
    
    direction = target - position
    direction = direction / np.linalg.norm(direction)
    return direction

def get_position(radius):
    """
    Generate a random position on a sphere of a given radius around a given center. 
    
    Parameters:
    -----------
    radius : float
        The radius of the sphere on which the random position is generated.

    Returns:
    --------
    numpy.ndarray
        A 3D vector (x, y, z) representing the random position in Cartesian coordinates.
    """
    
    # Fixed colatitude angle (theta) = pi/2 (the equator plane in spherical coordinates)
    theta = np.random.uniform(0, np.pi)
    
    # Random longitude angle (phi) uniformly distributed between 0 and 2*pi
    phi = np.random.uniform(0, 2 * np.pi)
    
    # Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)  # z = 0 because theta = pi/2

    return np.array([x, y, z])

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Parameters:
    -----------
    r : float
        The radial distance from the origin (radius).
    theta : float
        The polar angle (in radians), measured from the positive z-axis.
    phi : float
        The azimuthal angle (in radians), measured from the positive x-axis in the xy-plane.
    
    Returns:
    --------
    numpy.ndarray
        A 3D vector (x, y, z) representing the random position in Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x, y, z])

import math
def atomic_radius(lattice_constant, structure='fcc'):
    if structure == 'fcc':
        # Face-Centered Cubic (FCC)
        return (math.sqrt(2) / 4) * lattice_constant
    elif structure == 'bcc':
        # Body-Centered Cubic (BCC)
        return (math.sqrt(3) / 4) * lattice_constant
    elif structure == 'hcp':
        # Hexagonal Close-Packed (HCP)
        return 0.5 * lattice_constant
    else:
        raise ValueError("Unknown crystal structure")