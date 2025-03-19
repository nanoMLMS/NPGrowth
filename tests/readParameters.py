from mpi4py import MPI
import NPGrowth.Parameters

me = MPI.COMM_WORLD.Get_rank()

if me == 0:
    parameters = NPGrowth.Parameters("examples/parameters/isotrope.toml")
    print(parameters)

MPI.Finalize()