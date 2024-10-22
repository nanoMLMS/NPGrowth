import utils
import dynamics
import numpy as np

system = utils.System('Cu675.xyz')
seed_center_of_mass = system.get_center_of_mass()

for i in range(50):
    angular_positions = [[np.pi / 2, np.pi/2 * i] for i in range(4)] # [theta, phi]
    directions = [seed_center_of_mass for i in range(len(angular_positions))]
    system.depo(angular_positions, directions)
    system.run(1000)

thermo = system.get_thermo()

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(thermo['steps'], thermo['temp'])
plt.savefig('temperature')

plt.figure(1)
plt.plot(thermo['steps'], thermo['potE'])
plt.savefig('potential energy')