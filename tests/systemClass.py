from mpi4py import MPI
import NPGrowth.System
import NPGrowth.Parameters
import numpy as np

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

if me == 0:
    parameters = NPGrowth.Parameters('examples/parameters/isotrope.toml')
else:
    parameters = None

parameters = comm.bcast(parameters, root=0)

system = NPGrowth.System(parameters)

nAtoms = system.L.get_natoms()

newAtomsPositions = np.array([[25, 25, 25], [25, 0, 0]])
newAtomsVelocities = np.array([[-10, -10, -10], [-10, 0, 0]])
newAtomsTypes = np.array(['Cu', 'Cu'])
ids = np.array([nAtoms + 1, nAtoms + 2])

system.depo(newAtomsPositions, newAtomsVelocities, newAtomsTypes, ids)

system.run(1000)

MPI.Finalize()