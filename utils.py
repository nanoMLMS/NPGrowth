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


from ase.io import read
from ase.visualize import view
import numpy as np
import tempfile
from ase import Atoms
from scipy.spatial.distance import pdist, cdist
from asap3.io.trajectory import Trajectory # Used trajectory from asap3 module

class System:
    """
    This class is responsible for loading and storing the atoms and trajectory.

    Parameters:
    -----------
    filename : str
        The path to the xyz file containing the initial atoms positions.

    Attributes:
    -----------
    atoms : ASE Atoms object
        The group of atoms in the system.
    
    trajectory_file : TemporaryFile
        Temporary file for storing the trajectory of the system.

    Methods:
    --------
    max_distance_between():
        Return maximum distance between any two atoms in the system.
    max_distance_from(position):
        Return maximum distance of the system from a given position.
    view(indices = None):
       Visualize the atoms with the default ase viewer.
    view_trajectory():
       Visualize the atoms trajectory with the default ase viewer.
    temperature():
       Returns the temperature of the system.
    center_of_mass(indices = None):
        Return center of mass of the system.
    add(atom)
        Add atom to the system.
    add_atom(temperature, direction, position, element = None):
        Add atom to the system with given initial temperature, direction, position.
    open_trajectory():
        Returns trajectory associated with the system.
    max_atom_separation():
        Returns the maximum separation between atoms in the system.
    """

    def __init__(self, filename):
        # TODO check if file exists
        self.atoms = read(filename)
        
        # Set unit cell for the system. Needed for asap calculator.
        cell_size = self.max_distance_between()
        self.atoms.set_cell((cell_size, cell_size, cell_size))
        self.atoms.center()

        # Where to store the trajectory associated with the system
        self.trajectory_file = tempfile.NamedTemporaryFile(delete=False)  # Binary mode file for the trajectory
        # track if already been written
        self.__first_time = True

    def max_distance_between(self):
        """
        Calculate the maximum distance between any two atoms in the system.

        Returns:
        --------
        float
            The maximum distance between two atoms in the system.
        """
        positions = self.atoms.get_positions()  # Get atomic positions
        distances = pdist(positions)  # Calculate all pairwise distances
        max_dist = np.max(distances)  # Find the maximum distance
        return max_dist
    
    def max_distance_from(self, position):
        """
        Calculate the distance of the further atom of the system from the given position.

        Parameters:
        -----------
        position : numpy.ndarray
            A 3D vector (x, y, z) representing the given position.

        Returns:
        --------
        float
            The maximum distance of the system of atoms from given position.
        """

        distances = np.linalg.norm(self.atoms.get_positions() - position, axis=1)

        return np.max(distances)

    def view(self, indices = None):
        """
        Visualize the atoms with the default ase viewer.
        """
        indices = indices if indices is not None else np.arange(len(self.atoms))
        view(self.atoms[indices])
    
    def view_trajectory(self):
        """
        Visualize the trajectory with the default ase viewer.
        """
        traj = Trajectory(self.trajectory_file.name, 'r')
        view(traj)
        traj.close()

    def temperature(self):
        """
        Return temperature of the system calculated from the equipartition theorem equation.
        """
        return self.atoms.get_temperature()

    def center_of_mass(self, indices = None):
        """
        Return center of mass of the system.
        """
        indices = indices if indices is not None else np.arange(len(self.atoms))
        positions = self.atoms.get_positions()[indices]
        masses = self.atoms.get_masses()[indices]
        total_mass = np.sum(masses)
        com = np.sum(positions.T * masses, axis=1) / total_mass
        return com

    def add(self, atom):
        """
        Add atom to the system.

        Parameters
        ----------
        atom : Ase Atom object
            Atom to be added.
        """
        self.atoms += atom
    
    def add_atom(self, temperature, direction, position, element = None):
        """
        Add atom to the system.

        Parameters
        ----------
        temperature : float
            Temperature of the atom to be added. This is converted to the velocity of the atom.
        direction : numpy.ndarray
            A 3D vector (x, y, z) representing the target of the atom.
        position : umpy.ndarray
            A 3D vector (x, y, z) representing the position of the atom that will be added.
        element : string
            Optional chemical symbol of the atom to be added. If not specified and if all the atoms
            in the system have same chemical symbol, this will be chosen.
        """
        if element == None:
            elements = self.atoms.get_chemical_symbols()
            if all(i == elements[0] for i in elements):
                element = elements[0]
            else:
                print('No element specified and not unique element in system')
                sys.exit()
        atom = Atoms(element, positions=[position])
        velocity_direction = eigen_vector(position, direction)
        velocity = thermal_velocity(atom.get_masses()[0], temperature) * velocity_direction
        atom.set_velocities([velocity])
        self.add(atom)
    
    def open_trajectory(self):
        """Open the trajectory in 'w' mode the first time, then 'a' mode."""
        if self.__first_time:
            mode = 'w'  # First time, so write mode
            self.__first_time = False  # Update flag
        else:
            mode = 'a'  # Subsequent calls use append mode
        
        return Trajectory(self.trajectory_file.name, mode, self.atoms)
    
    def max_atom_separation(self):
        """
        Compute max distance between atoms.

        This helps determine if any atoms are detached from the structure. 
        For example, we can check if max_distance_between() returns a value greater than the lattice constant.
        If the distance exceeds the lattice constant, it indicates that one or more atoms are detached
        from the main structure.

        Returns:
        float: Max distance between atoms in Angstrom
        """
        positions = self.atoms.get_positions()
        n = len(self.atoms)
        
        # Compute pairwise distances
        pairwise_distances = cdist(positions, positions)
        
        # Set diagonal (self-distances) to infinity so they don't interfere with finding the minimum
        np.fill_diagonal(pairwise_distances, np.inf)

        # Find the minimum distance for each atom, excluding the self-distance
        min_distances = np.min(pairwise_distances, axis=1)

        return max(min_distances)
    

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
    
    # # Fixed colatitude angle (theta) = pi/2 (the equator plane in spherical coordinates)
    # theta = np.random.uniform(0, np.pi)
    
    # # Random longitude angle (phi) uniformly distributed between 0 and 2*pi
    # phi = np.random.uniform(0, 2 * np.pi)
    
    # # Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z)
    # x = radius * np.sin(theta) * np.cos(phi)
    # y = radius * np.sin(theta) * np.sin(phi)
    # z = radius * np.cos(theta)  # z = 0 because theta = pi/2

    return np.array([0, 0, radius])