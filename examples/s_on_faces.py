import copy
import numpy as np
import sys
import NPGrowth
import NPGrowth.functions

parameters_filename = sys.argv[1]

system = NPGrowth.System(parameters_filename) # Load system with parameters in parameters_filename
parameters = NPGrowth.SimulationParameters(parameters_filename)

system.run(parameters.termalize_steps) # Run some steps to termalize system

# Nedded for later use
seed_center_of_mass = system.get_center_of_mass() 
atom_radius = NPGrowth.functions.atomic_radius(lattice_constant=3.6150)
cube_edge = system.get_max_diameter() / (3.**(1./2))

# Points of a curve
bezier = NPGrowth.BezierPoints(cube_edge, atom_radius)
points = bezier.get_points()

p_yz_right = np.array([np.array([cube_edge/2 + 2 * atom_radius, point[0], point[1]]) for point in points])
p_yz_right = p_yz_right[::-1]
p_yz_left = np.array([np.array([-(cube_edge/2 + 2 * atom_radius), -point[0], point[1]]) for point in points])
p_xy_top = np.array([np.array([point[0], point[1], cube_edge/2 + 2 * atom_radius]) for point in points])
p_xy_bottom = np.array([np.array([point[0], point[1], -(cube_edge/2 + 2 * atom_radius)]) for point in points])
p_xz_front = np.array([np.array([point[0],-(cube_edge/2 + 2 * atom_radius), point[1]]) for point in points])
p_xz_back = np.array([np.array([-point[0], cube_edge/2 + 2 * atom_radius, point[1]]) for point in points])
p_xz_front = p_xz_front[::-1]

to_depo_for_step = [[p_yz_right[i], p_yz_left[i], p_xy_top[i], p_xy_bottom[i], p_xz_front[i], p_xz_back[i]] for i in range(len(points))]

cutoff = system.get_cutoff_from_log()
for _ in range(parameters.n_depo_repeat):
    for targets in to_depo_for_step:
        radius = system.get_further_atom(seed_center_of_mass) + cutoff
        angular_positions = [NPGrowth.functions.cartesian_to_spherical(p[0], p[1], p[2]) for p in targets]
        positions = [NPGrowth.functions.spherical_to_cartesian(radius, p[1], p[2]) for p in angular_positions]
        positions = [p + seed_center_of_mass for p in positions]
        targets = [p + seed_center_of_mass for p in targets]
        system.depo(positions, targets)
        system.run(parameters.steps_to_next_depo)