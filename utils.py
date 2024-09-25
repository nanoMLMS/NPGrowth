import sys
import toml

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

    Methods:
    --------
    __init__(filename: str):
        Initializes the object by loading parameters from the given TOML file.
    
    __get() -> dict:
        Reads and parses the TOML file containing the parameters. Returns a dictionary 
        of parameters. Exits the program if the file cannot be opened.

    __check():
        Placeholder for checking the validity of parameters.
    """

    def __init__(self, filename):
        parameters = self.__get()

        self.n_atoms = parameters['n_atoms']
        self.radius_offset = parameters['radius_offset']

        md_params = parameters['dynamics']
        self.temperature = md_params['temperature']
        self.timestep = md_params['timestep']
        self.steps = md_params['steps']
        self.write_interval = md_params['write_interval']

        self.__filename = filename

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


from ase.io import read
from ase.visualize import view
import numpy as np
from scipy.spatial.distance import pdist

class System:
    """
    This class is responsible for loading and storing the atoms.

    Parameters:
    -----------
    filename : str
        The path to the xyz file containing the initial atoms positions.

    Attributes:
    -----------
    atoms : ASE Atoms object
        The group of atoms in the system.

    Methods:
    --------
    __init__():
        Initializes the object by loading the atoms from the given xyz file.
    max_distance():
        Calculate the maximum distance between any two atoms in the system.
    view():
       Visualize the atoms with the default ase viewer.
    """

    def __init__(self, filename):
        # TODO check if file exists
        self.atoms = read(filename)

    def max_distance(self):
        """
        Calculate the maximum distance between any two atoms in the system.

        Parameters:
        -----------
        atoms : ASE Atoms object
            The system of atoms for which to calculate the maximum distance.

        Returns:
        --------
        float
            The maximum distance between two atoms.
        """
        positions = self.atoms.get_positions()  # Get atomic positions
        distances = pdist(positions)  # Calculate all pairwise distances
        max_dist = np.max(distances)  # Find the maximum distance
        return max_dist

    def view(self):
        """
        Visualize the atoms with the default ase viewer.
        """
        view(self.atoms)