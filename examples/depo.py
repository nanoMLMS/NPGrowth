import numpy as np
import sys
import NPGrowth
import NPGrowth.functions

parameters_filename = sys.argv[1]

system = NPGrowth.System(parameters_filename) # Load system with parameters in parameters_filename

system.run(5000) # Run some steps to termalize system

# Nedded for later use
seed_center_of_mass = system.get_center_of_mass() 
atom_radius = NPGrowth.functions.atomic_radius(lattice_constant=3.6150)
cube_edge = system.get_max_diameter() / (3.**(1./2))

# Points of a curve
bezier = NPGrowth.BezierPoints(cube_edge, atom_radius)
points = bezier.get_points()

positions_on_surface = [np.array([cube_edge/2 + 2 * atom_radius, point[0], point[1]]) for point in points] # Corresponding points of the curve on the surface
angular_positions = [NPGrowth.functions.cartesian_to_spherical(position[0], position[1], position[2]) for position in positions_on_surface]

cutoff = system.get_cutoff_from_log()
# Remove positions on the 4 surface of the cube
for angular_position in angular_positions:
    radius = system.get_max_diameter() / 2 + cutoff
    targets = [NPGrowth.functions.spherical_to_cartesian(angular_position[0], angular_position[1], angular_position[2] + np.pi/2 * i) for i in range(4)]
    targets = [p + seed_center_of_mass for p in targets]
    positions = [NPGrowth.functions.spherical_to_cartesian(radius, angular_position[1], angular_position[2] + np.pi/2 * i) for i in range(4)]
    positions = [p + seed_center_of_mass for p in positions]
    system.depo(positions, targets)
    system.run(100)