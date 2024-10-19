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
from ase import Atoms
import tempfile
import numpy as np
from scipy.spatial.distance import pdist

class System:
    def __init__(self, filename):
        # Initialize LAMMPS interface
        self.lmp = PyLammps()
        
        # Read atoms from xyz file
        atoms = read(filename)

        # Initialization
        self.lmp.units("metal")
        self.lmp.boundary("s s s")  # Shrink-wrap boundaries
        self.lmp.atom_style("atomic")

        # System definition
        positions = atoms.get_positions()  # Get atomic positions
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        buffer = 10.0
        cell_x = max_pos[0] - min_pos[0] + buffer
        cell_y = max_pos[1] - min_pos[1] + buffer
        cell_z = max_pos[2] - min_pos[2] + buffer
        atoms.set_cell([cell_x, cell_y, cell_z])
        atoms.center()
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        write(self.tmp.name, atoms, format='lammps-data')
        self.lmp.read_data(self.tmp.name)

        # Simulation settings
        self.lmp.mass(1, 63.546)  # Set mass for Copper (atom type 1)
        self.lmp.pair_style("eam")
        self.lmp.pair_coeff('*', '*', 'Cu_u3.eam')

        # Initially, no MD algorithm is set for atoms
        self.md_algorithm = [None] * len(atoms)
    
    def get_max_diameter(self):
        """
        Calculate the maximum distance between any two atoms in the system.

        Returns:
        --------
        float
            The maximum distance between two atoms in the system.
        """
        positions = self.atoms.get_positions()  # Get atomic positions
        distances = pdist(positions)  # Calculate all pairwise distances
        max_rad = np.max(distances)  # Find the maximum distance
        return max_rad
    
    def add_atom(self, position):
        """
        Adds a new atom to the system at the given position, ensuring the box is large enough.
        
        Parameters
        ----------
        position : list or array-like
            The position (x, y, z) where the new atom will be placed (in box units).
        """
        # Expand the box if the atom's position is outside the current bounds
        
        # Add a new atom using PyLammps create_atoms method
        self.lmp.create_atoms(1, 'single', position[0], position[1], position[2], 'units', 'box')
        
        # Check if an atom was successfully created
        new_atom_count = self.lmp.get_natoms()
        if new_atom_count <= len(self.atoms):
            print("Error: No atom created. Make sure the position is inside the box.")
        else:
            # Update the ASE atoms object to keep track of the new atom
            self.atoms += Atoms('Cu', positions=[position])  # Example with Copper atom
        
        # Append None to the md_algorithm list for the new atom
        self.md_algorithm.append(None)
    
    def __getitem__(self, key):
        """Enable slicing, so we can select specific atoms."""
        return self.atoms[key]

    def set_algorithm(self, indices, algorithm):
        """Assign MD algorithm to specific atoms."""
        for i in indices:
            self.md_algorithm[i] = algorithm
    
    def evolve(self, parameters):
        """Run the MD simulation using LAMMPS based on the specified parameters."""
        
        # Set MD algorithms for different atom groups
        if Algorithm.langevin in self.md_algorithm:
            group_langevin = "group langevin id " + " ".join(str(i+1) for i, algo in enumerate(self.md_algorithm) if algo == Algorithm.langevin)
            self.lmp.command(group_langevin)
            self.lmp.fix('lngnve', 'langevin', 'nve')
            self.lmp.fix('lnglng', "langevin", "langevin", 300.0, 300.0, 0.1, 12345)

        if Algorithm.verlet in self.md_algorithm:
            group_verlet = "group verlet id " + " ".join(str(i+1) for i, algo in enumerate(self.md_algorithm) if algo == Algorithm.verlet)
            self.lmp.command(group_verlet)
            self.lmp.fix('vrl', "verlet", "nve")
        
        # Save trajectory
        self.lmp.dump('mydmp', 'all', 'atom', 100, 'dump.lammpstrj')

        # Set the time step and run the simulation
        self.lmp.timestep(parameters.timestep)
        self.lmp.run(parameters.steps)

def thermal_velocity(mass, temperature):
    """
    Returns velocity of the thermal motion of particles
    at a given temperature
    """
    
    # Calculate the velocity corresponding to the temperature
    # v = sqrt((3 * kB * T) / m)
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