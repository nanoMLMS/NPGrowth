from asap3.md.langevin import Langevin # Used langevin from asap module. Don't know if works for all calculators.
from asap3.io.trajectory import Trajectory # Used trajectory from asap3 module
from asap3 import EMT
from tqdm.notebook import tqdm

# Asap has special versions of ASE objects. Need to see if they works with others calculators
# https://asap3.readthedocs.io/en/latest/manual/Using_asap3_with_the_Atomic_Simulation_Environment.html

def MD(atoms, trajectory, showProgress=False):
    """
    Compute dynamic using Langevin algorithm.
    Saves trajectory of atoms in trajectory parameter.
    Modify parameters.toml to change temperature, timestep,
    number of steps, write interval and calculator.
    """

    # Black box that compute forces and energies
    atoms.calc = EMT()

    # Langevin dynamics (constant temperature)
    dyn = Langevin(atoms, parameters.timestep, temperature_K=TemperatureK, friction=0.002)
    dyn.attach(trajectory.write, interval=writeInterval)
    dyn.attach(removeRotation, atoms=atoms)
    
    # If showProgress set to True show progress with tqdm lib
    if showProgress:
        pbar = tqdm(total=steps/writeInterval, desc="Progress")
        dyn.attach(pbar.update, interval=writeInterval)
    dyn.run(steps)