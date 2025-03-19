from NPGrowth.utils import read_xyz
from mpi4py import MPI

me = MPI.COMM_WORLD.Get_rank()

if me == 0:
    atoms = read_xyz('examples/seeds/Cu4631.xyz')
    print(atoms)

MPI.Finalize()