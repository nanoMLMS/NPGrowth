# Setup
Install [openMPI](https://www.open-mpi.org/) or something similar ([MPICH](https://www.mpich.org/)) that provides the libraries needed for compiling LAMMPS with [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) standard.

Setup a virtual environment and do all the next steps with the environment activated.
## Build LAMMPS
Make build directory inside LAMMPS cloned folder and run the cmake command showed above inside the build directory. 
```Bash
cmake -D BUILD_MPI=yes -D PKG_MANYBODY=yes -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV_PATH ../cmake
```
Replace $VIRTUAL_ENV_PATH with the actual path of the python virtual environment where NPGrowth will be used.

Compile with 
```Bash
make -j #N
```
#N number of processors for speeding things up.

## Install
Inside the build directory run:
```Bash
make install-python
```
For parallel runs install mpi4py and if you have nvidia gpu check cuda support.

If struggling while installing mpi4py try 
```Bash
rm $MINICONDA3_INSTALLATION_PATH/envs/codelab/compiler_compat/ld
```
Replace $MINICONDA3_INSTALLATION_PATH with actual location.

Using conda for mpi4py installation provides own MPI version that can differ from the system installation. For LAMMPS and mpi4py to work together they need to be built with same MPI version. So use pip install and not conda install.

# Usage
Here is an example of a simple deposition. An example of a file containing the parameters needed for the simulation can be found inside examples/parameters folder in the project.
```Python
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

newAtomsPositions = np.array([[25, 0, 0], [25, 25, 25]])
newAtomsVelocities = np.array([[-10, 0, 0], [-10, -10, -10]])
newAtomsTypes = np.array(['Cu', 'Cu'])

for i in range(10):
    nAtoms = system.L.get_natoms()
    ids = np.array([nAtoms + 1, nAtoms + 2])
    system.depo(newAtomsPositions, newAtomsVelocities, newAtomsTypes, ids)
    system.run(2000)

MPI.Finalize()
```