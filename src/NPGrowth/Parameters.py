import sys
import toml

class Parameters:
    def __init__(self, filename):
        """
        Initialize an object with the parameters specified in the file with name
        given as argument.
        """
        self.__filename = filename

        parameters = self.__get()

        # Filenames
        self.seed_filename: str = parameters['seed_filename']
        self.trajectory_filename: str = parameters['trajectory_filename']
        self.potential_filename: str = parameters['potential_filename']

        # System parameters
        self.temperature: float = parameters['temperature']
        self.lattice_constant: float = parameters['lattice_constant']
        self.force_treshold: float = parameters['force_treshold']
        self.force_treshold_reached: float = parameters['force_treshold_reached']
        self.species_masses: list[object] = parameters['species_masses']
        self.damping_parameter: float = parameters['damping_parameter']

        # Dynamics parameters
        self.timestep: float = parameters['timestep']
        self.write_interval: int = parameters['write_interval']
        self.termalize_steps: int = parameters['termalize_steps']
        self.n_depo_repeat: int = parameters['n_depo_repeat']
        self.steps_to_next_depo: int = parameters['steps_to_next_depo']
        self.simulation_box_offset: float = parameters['simulation_box_offset']

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
        return (
            f"Filenames:\n"
            f"  Seed Filename: {self.seed_filename}\n"
            f"  Trajectory Filename: {self.trajectory_filename}\n"
            f"  Potential Filename: {self.potential_filename}\n\n"
            f"System Parameters:\n"
            f"  Temperature: {self.temperature}\n"
            f"  Lattice Constant: {self.lattice_constant}\n"
            f"  Force Threshold: {self.force_treshold}\n"
            f"  Force Threshold Reached: {self.force_treshold_reached}\n"
            f"  Species masses: {self.species_masses}\n"
            f"  Langevin damping parameter: {self.damping_parameter}\n\n"
            f"Dynamics Parameters:\n"
            f"  Timestep: {self.timestep}\n"
            f"  Write Interval: {self.write_interval}\n"
            f"  Thermalization Steps: {self.termalize_steps}\n"
            f"  Deposition Repeats: {self.n_depo_repeat}\n"
            f"  Steps to Next Deposition: {self.steps_to_next_depo}"
            f"  Simulation box offset from surface: {self.simulation_box_offset}"
        )
