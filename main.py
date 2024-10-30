import utils.functions
import numpy as np

system = utils.System('./seeds/Cu4631.xyz')

system.run(1000)

seed_center_of_mass = system.get_center_of_mass()

atom_radius = utils.functions.atomic_radius(lattice_constant=3.6150)
cube_edge = system.get_max_diameter() / (3.**(1./2))

bezier = utils.BezierPoints(cube_edge, atom_radius)
points = bezier.get_points()

positions_on_surface = [np.array([point[0], -(cube_edge/2 + atom_radius * 2), point[1]]) for point in points]

angular_positions = [utils.functions.cartesian_to_spherical(position[0], position[1], position[2]) for position in positions_on_surface]

cutoff = system.get_cutoff_from_log()
for j in range(2):
    for i, position in enumerate(angular_positions):
        print(i+j*len(angular_positions))
        radius = system.get_max_diameter() / 2 + cutoff # Radius where the atoms that will be deposited will be placed initially
        atom_positions = [utils.functions.spherical_to_cartesian(radius, position[1], position[2] + np.pi/2 * i) for i in range(4)]
        atom_positions = [atom_position + seed_center_of_mass for atom_position in atom_positions]
        targets = [seed_center_of_mass for i in range(4)]
        system.depo(atom_positions, targets)
        system.run(100)