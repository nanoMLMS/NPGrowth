from asap3.md.langevin import Langevin # Used langevin from asap module. Don't know if works for all calculators.
from asap3 import EMT
from tqdm import tqdm
from ase.md.velocitydistribution import ZeroRotation

# Asap has special versions of ASE objects. Need to see if they works with others calculators
# https://asap3.readthedocs.io/en/latest/manual/Using_asap3_with_the_Atomic_Simulation_Environment.html



def remove_rotation(atoms):
    ZeroRotation(atoms)


def langevin(system, parameters, trajectory, showProgress=False):
    """
    Perform Langevin dynamics simulation with constant temperature on a system of atoms.

    This method sets up and runs a Langevin molecular dynamics simulation using the EMT
    (Effective Medium Theory) calculator for forces and energies. It allows for optional
    progress display using `tqdm`.

    Parameters
    ----------
    system : :class:`utils.System`
        The system object containing the atoms for the simulation. It must have an `atoms`
        attribute, which is an ASE `Atoms` object.
    
    parameters : :class:`utils.SimulationParameters`
        An object containing the simulation parameters such as timestep, temperature,
        steps, and write_interval. The attributes expected from this object are:
        
        - timestep : float
            The timestep for the dynamics (in ASE units).
        - temperature : float
            The temperature of the system in Kelvin.
        - steps : int
            The number of steps to run the simulation.
        - write_interval : int
            The interval at which the trajectory is written.
    
    trajectory : object
        The trajectory object where the atomic configurations will be saved during the
        simulation. Must have a `write` method.
    
    showProgress : bool, optional
        If set to `True`, a progress bar will be shown using the `tqdm` library (default is `False`).

    Returns
    -------
    None
        The function runs the Langevin dynamics simulation and writes the atomic trajectories
        to the specified `trajectory` object at the specified interval. The progress bar is 
        optional and can be enabled via the `showProgress` flag.
    
    Notes
    -----
    - This function uses the ASE `Langevin` class to perform constant-temperature molecular dynamics.
    - The system uses the `EMT` (Effective Medium Theory) calculator to compute forces and energies.
    - The `removeRotation` function is attached to ensure that any rotational motion is removed.
    - Progress is tracked and displayed with the `tqdm` library if `showProgress` is set to `True`.

    Example
    -------
    >>> import utils
    >>> import dynamics
    >>> from ase.io.trajectory import Trajectory
    >>> system = utils.System('file.xyz')
    >>> parameters = utils.SimulationParameters('parameters.toml')
    >>> trajectory = Trajectory("output.traj")
    >>> dynamics.langevin(system, parameters, trajectory, showProgress=True)
    """

    # Black box that compute forces and energies
    system.atoms.calc = EMT()

    # Langevin dynamics (constant temperature)
    dyn = Langevin(system.atoms, parameters.timestep, temperature_K=parameters.temperature, friction=0.002)
    dyn.attach(trajectory.write, interval=parameters.write_interval)
    dyn.attach(remove_rotation, atoms=system.atoms)
    
    # If showProgress set to True show progress with tqdm lib
    if showProgress:
        pbar = tqdm(total=parameters.steps/parameters.write_interval, desc="Progress")
        dyn.attach(pbar.update, interval=parameters.write_interval)
    dyn.run(parameters.steps)