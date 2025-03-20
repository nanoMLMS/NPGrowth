import random
from lammps import lammps
import NPGrowth.Parameters
from NPGrowth.utils import read_xyz
import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from ctypes import c_int
import uuid

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

class System:
    def __init__(self, parameters: NPGrowth.Parameters):
        # 1) Initialization
        # 2) System definition
        # 3) Simulation settings
        # 4) Visualization

        self.L = lammps()
        self.parameters = parameters
        self.atomsUniqueTypes = None
        self.typesToNumber = None
        
        self._initialize()
        self._definition()
        self._settings()
        self._visualization()

    def _initialize(self):
        self.L.command('units metal')           # mass=grams/mole, distance=Angstroms, energy eV
        self.L.command('atom_style atomic')     # atoms are points with position
        self.L.command('boundary f f f')        # the simulation box is fixed
        self.L.command('atom_modify map yes')   # map atoms, needed for scatter and gather
    
    def _definition(self):
        if me == 0:
            atomsTypes, atomsPositions = read_xyz(self.parameters.seed_filename)
        else:
            atomsTypes = None
            atomsPositions = None
        
        atomsTypes = comm.bcast(atomsTypes, root=0)
        atomsPositions = comm.bcast(atomsPositions, root=0)

        xMin, yMin, zMin = np.min(atomsPositions, axis=0) - self.parameters.simulation_box_offset
        xMax, yMax, zMax = np.max(atomsPositions, axis=0) + self.parameters.simulation_box_offset

        # Create dictionary mapping unique species to numbers
        self.atomsUniqueTypes = np.unique(atomsTypes)
        self.typesToNumber = { species: i + 1 for i, species in enumerate(self.atomsUniqueTypes) }

        # Define the simulation box
        self.L.command(f"region mybox block {xMin} {xMax} {yMin} {yMax} {zMin} {zMax} units box")

        self.L.command(f'create_box {len(self.parameters.species_masses)} mybox')

        # Array containing numbers corresponding to species
        atomNumbers = np.array([self.typesToNumber[atomType] for atomType in atomsTypes])
        # Create atoms
        # Create atoms from XYZ data
        for (atomType, position) in zip(atomNumbers, atomsPositions):
            self.L.command(f'create_atoms {atomType} single {position[0]} {position[1]} {position[2]} units box')
        
        self.L.command(f'group initial_atoms id 1:{len(atomsPositions)}')
    
    def _settings(self):
        self.L.command('pair_style eam')
        self.L.command(f'pair_coeff * * {self.parameters.potential_filename}')

        # Assign to each type of atom his mass
        for atomType in self.atomsUniqueTypes:
            self.L.command(f'mass {self.typesToNumber[atomType]} {self.parameters.species_masses[atomType]}')
        
        # No complessive rotation of the system
        # From lammps docs: The rescale keyword enables conserving the kinetic energy of the group or chunk of atoms by rescaling the velocities after the momentum was removed.
        self.L.command('fix momentum_fix initial_atoms momentum 1 linear 1 1 1 angular rescale')

        # Updates positions and velocities of the atoms at every step
        self.L.command('fix mynve all nve')

        # Set simulation timestep
        self.L.command(f' timestep {self.parameters.timestep}')

        # Langevin thermostat
        self.L.command(f'fix mylgv initial_atoms langevin {self.parameters.temperature} {self.parameters.temperature} {self.parameters.damping_parameter} {random.randint(1, 999999)}')

    def _visualization(self):
        atomTypesStr = " ".join(self.atomsUniqueTypes)
        print(atomTypesStr)
        self.L.command(f'thermo {self.parameters.write_interval}')
        self.L.command(f'thermo_style custom step temp pe ke etotal press')
        self.L.command(f'dump dump1 all custom {self.parameters.write_interval} {self.parameters.trajectory_filename} id type element x y z')
        self.L.command(f'dump_modify dump1 element {atomTypesStr}')

    def addAtoms(self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64], types: npt.NDArray[np.str_], ids: npt.NDArray[np.int32]):
        numberTypes = [self.typesToNumber[t] for t in types]
        self.L.create_atoms(len(positions), ids, numberTypes, positions.flatten(), velocities.flatten())
        
    def minimize(self, eTol: float, fTol: float, maxIter: int, maxEval: int):
        """
        Perform energy minimization of the system adjusting atoms coordinates.
        The arguments are the same as the minimize lammps command:
        -   etol = stopping tolerance for energy (unitless)
        -   ftol = stopping tolerance for force (force units)
        -   maxiter = max iterations of minimizer
        -   maxeval = max number of force/energy evaluations
        """
        self.L.command(f'minimize {eTol} {fTol} {maxIter} {maxEval}')
    
    def remove(self, targets, tolerance=0.1):
        for i, pos in enumerate(targets):
            region_name = f"target_region_{uuid.uuid4().hex[:8]}" # Define a region for each target position
            self.L.command(f'region {region_name} sphere {pos[0]} {pos[1]} {pos[2]} {tolerance}') # Add atoms in this region to a group for deletion
            self.L.command(f'group to_delete region {region_name}')
        self.L.command('delete_atoms group to_delete') # Delete the atoms in the current region
    
    def getIndex(self, id):
        atomIds = self.L.numpy.extract_atom('id')
        try:
            index = np.where(atomIds == id)[0][0]
        except:
            index = None
        return index

    def depo(self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64], types: npt.NDArray[np.str_], ids: npt.NDArray[np.int32]):
        self.addAtoms(positions, velocities, types, ids)
                
        # Group where atoms are added once they are near the surface
        self.L.command('group reached empty')

        # Variable with counts of atoms in reached group
        self.L.command('variable reached_count equal count(reached)')

        reachedIds = np.array([], dtype=int)

        count = 3 # number or elements in per-atom data
        while self.L.extract_variable('reached_count') != len(positions):
            ndata = len(ids) # number of ids
            c_ids = (c_int * ndata)(*ids)
            f = self.L.gather_atoms_subset('f', 1, count, ndata, c_ids)
            for i, id in enumerate(ids):
                i3 = i*3
                fMag = f[i3 + 0]*f[i3 + 0] + f[i3 + 1]*f[i3 + 1] + f[i3 + 2]*f[i3 + 2]
                if fMag > self.parameters.force_treshold:
                    c_ids = (c_int)(id)
                    v = self.L.gather_atoms_subset('v', 1, count, 1, c_ids)
                    v[0] = v[1] = v[2] = 0
                    self.L.scatter_atoms_subset('v', 1, count, 1, c_ids, v)
                    ids = np.delete(ids, np.where(ids == id))
                    reachedIds = np.append(reachedIds, id)
            if len(reachedIds):
                ndata = len(reachedIds) # number of ids
                c_ids = (c_int * ndata)(*reachedIds)
                f = self.L.gather_atoms_subset('f', 1, count, ndata, c_ids)
                for i, id in enumerate(reachedIds):
                    i3 = i*3
                    fMag = f[i3 + 0]*f[i3 + 0] + f[i3 + 1]*f[i3 + 1] + f[i3 + 2]*f[i3 + 2]
                    if fMag > self.parameters.force_treshold_reached:
                        reachedIds = np.delete(reachedIds, np.where(reachedIds == id))
                        self.L.command(f'group reached id {id}')
                        self.L.command(f'group initial_atoms id {id}')

            self.L.command('run 1 pre no post no')
        
        self.L.command('run 0 pre no')

    def distanceFromSystem(self, position):
        distances = np.linalg.norm(self.getPositions() - position, axis=1) # Compute distances between position and all positions
        distances = distances[distances > 0] # Exclude the distance from itself if position atom of the system
        return np.min(distances)

    def run(self, steps: int):
        self.L.command(f'run {steps}')

    def getPositions(self):
        positions = self.L.numpy.extract_atom('x', 3)
        return positions