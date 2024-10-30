from ase.build import bulk, cut

from ase.cluster.cubic import FaceCenteredCubic

surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
layers = [7, 7, 7]
lc = 3.61000
atoms = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc)

from ase.io import write

write('cu_cube.xyz', atoms)