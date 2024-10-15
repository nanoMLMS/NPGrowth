from asap3.md.langevin import Langevin # Used langevin from asap module. Don't know if works for all calculators.
from asap3 import EMT
from tqdm import tqdm
from ase.md.velocitydistribution import ZeroRotation
from asap3.io.trajectory import Trajectory # Used trajectory from asap3 module

# Asap has special versions of ASE objects. Need to see if they works with others calculators
# https://asap3.readthedocs.io/en/latest/manual/Using_asap3_with_the_Atomic_Simulation_Environment.html



def remove_rotation(atoms):
    ZeroRotation(atoms)


def langevin(system, parameters, showProgress=False):
    """
    Perform Langevin dynamics simulation with constant temperature on a system of atoms.

    This method sets up and runs a Langevin molecular dynamics simulation using the EMT
    (Effective Medium Theory) calculator for forces and energies. It allows for optional
    progress display using `tqdm`.

    Parameters
    ----------
    system : :class:`utils.System`
        The system object containing the atoms and trajectory for the simulation. 
    
    parameters : :class:`utils.SimulationParameters`
        An object containing the simulation parameters.
    
    showProgress : bool, optional
        If set to `True`, a progress bar will be shown using the `tqdm` library (default is `False`).

    Returns
    -------
    None
    
    Notes
    -----
    - The `removeRotation` function is attached to ensure that any rotational motion is removed.
    - Progress is tracked and displayed with the `tqdm` library if `showProgress` is set to `True`.

    Example
    -------
    >>> import utils
    >>> import dynamics
    >>> system = utils.System('file.xyz')
    >>> parameters = utils.SimulationParameters('parameters.toml')
    >>> dynamics.langevin(system, parameters, showProgress=True)
    """

    # Black box that compute forces and energies
    system.atoms.calc = EMT()

    # Langevin dynamics (constant temperature)
    dyn = Langevin(system.atoms, parameters.timestep, temperature_K=parameters.temperature, friction=0.002)

    traj = system.open_trajectory()
    dyn.attach(traj.write, interval=parameters.write_interval)
    dyn.attach(remove_rotation, atoms=system.atoms)
    
    # If showProgress set to True show progress with tqdm lib
    if showProgress:
        pbar = tqdm(total=parameters.steps/parameters.write_interval, desc="Progress")
        dyn.attach(pbar.update, interval=parameters.write_interval)
    
    dyn.run(parameters.steps)


from ase.md.verlet import VelocityVerlet
from ase.constraints import FixAtoms

import numpy as np

def velocityVerlet(system, parameters, showProgress=False):
    """
    Perform dynamics simulation using velocity verlet alghorithm to integrate Newton's equation
    of motion.

    This method sets up and runs a molecular dynamics simulation using the EMT
    (Effective Medium Theory) calculator for forces and energies. It allows for optional
    progress display using `tqdm`.

    Parameters
    ----------
    system : :class:`utils.System`
        The system object containing the atoms and trajectory for the simulation. 
    
    parameters : :class:`utils.SimulationParameters`
        An object containing the simulation parameters.
    
    showProgress : bool, optional
        If set to `True`, a progress bar will be shown using the `tqdm` library (default is `False`).

    Returns
    -------
    None
    
    Notes
    -----
    - The `removeRotation` function is attached to ensure that any rotational motion is removed.
    - Progress is tracked and displayed with the `tqdm` library if `showProgress` is set to `True`.

    Example
    -------
    >>> import utils
    >>> import dynamics
    >>> system = utils.System('file.xyz')
    >>> parameters = utils.SimulationParameters('parameters.toml')
    >>> dynamics.velocityVerlet(system, parameters, showProgress=True)
    """

    # Black box that compute forces and energies
    system.atoms.calc = EMT()

    indices =[atom.index for atom in system.atoms]
    indices.pop()

    constraint = FixAtoms(indices=indices)
    system.atoms.set_constraint(constraint)

    # Langevin dynamics (constant temperature)
    dyn = VelocityVerlet(system.atoms, parameters.timestep)

    traj = system.open_trajectory()
    dyn.attach(traj.write, interval=parameters.write_interval)
    dyn.attach(remove_rotation, atoms=system.atoms)
    
    # If showProgress set to True show progress with tqdm lib
    if showProgress:
        pbar = tqdm(total=parameters.steps/parameters.write_interval, desc="Progress")
        dyn.attach(pbar.update, interval=parameters.write_interval)
    
    dyn.run(parameters.steps)