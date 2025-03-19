import NPGrowth
import NPGrowth.Parameters
import NPGrowth.System
import NPGrowth.algorithms

parameters = NPGrowth.Parameters()

system = NPGrowth.System(parameters)

system.algorithm = velocityVerlet

system.run(10)