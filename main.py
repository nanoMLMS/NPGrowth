import utils
import numpy as np

system = utils.System('./seeds/Cu675.xyz')

system.run(5000)

seed_center_of_mass = system.get_center_of_mass()

for i in range(50):
    angular_positions = [[np.pi / 2, np.pi/2 * i] for i in range(4)] # [theta, phi] angular position relative to the center of mass of the system
    directions = [seed_center_of_mass for i in range(len(angular_positions))]
    system.depo(angular_positions, directions)
    system.run(1000)