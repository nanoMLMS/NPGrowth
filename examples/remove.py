import numpy as np
import sys
import NPGrowth

parameters_filename = sys.argv[1]

system = NPGrowth.System(parameters_filename) # Load system with parameters in parameters_filename

system.run(5000) # Run some steps to termalize system

# Nedded for later use
seed_center_of_mass = system.get_center_of_mass() 
atom_radius = NPGrowth.functions.atomic_radius(lattice_constant=3.6150)
cube_edge = system.get_max_diameter() / (3.**(1./2))

# Points of a curve
bezier = NPGrowth.BezierPoints(cube_edge, atom_radius/20)
points = bezier.get_points()

positions_on_surface = [np.array([cube_edge/2, point[0], point[1]]) for point in points] # Corresponding points of the curve on the surface
angular_positions = [NPGrowth.functions.cartesian_to_spherical(position[0], position[1], position[2]) for position in positions_on_surface]

# Remove positions on the 4 surface of the cube
for i in range(4):
    positions = [NPGrowth.functions.spherical_to_cartesian(p[0], p[1], p[2] + np.pi/2 * i) for p in angular_positions]
    positions = [p + seed_center_of_mass for p in positions]
    system.remove(positions, atom_radius)


system.run(10000) # Run some steps