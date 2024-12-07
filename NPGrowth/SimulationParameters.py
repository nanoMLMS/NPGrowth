import sys
import toml
from ase import units

class SimulationParameters:
    def __init__(self, filename):
        self.__filename = filename

        parameters = self.__get()

        self.temperature = parameters['temperature']
        self.force_treshold = parameters['force_treshold']
        self.trajectory_filename = parameters['trajectory_filename']
        self.lattice_constant = parameters['lattice_constant']
        self.write_interval = parameters['write_interval']
        self.minimize_before_simulation = parameters['minimize_before_simulation']
        self.timestep = parameters['timestep'] # picoseconds
        self.seed_filename = parameters['seed_filename']
        self.thermo_data_filename = parameters['thermo_data_filename']
        self.termalize_steps = parameters['termalize_steps']
        self.n_depo_repeat = parameters['n_depo_repeat']
        self.steps_to_next_depo = parameters['steps_to_next_depo']

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