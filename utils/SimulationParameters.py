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