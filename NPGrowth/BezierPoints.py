import numpy as np
import bezier
from scipy.integrate import quad
from scipy.optimize import root_scalar

class BezierPoints:
    def __init__(self, edge, atom_radius):
        nodes = np.asfortranarray([
            [-edge/2, -edge/2, edge/2, edge/2],  # x-coordinates
            [-edge/2, edge/2, -edge/2, edge/2],  # y-coordinates
        ])
        self.curve = bezier.Curve(nodes, degree=3)
        self.num_points = int(self.get_total_length() / (2 * atom_radius))

    def get_arc_length_integrand(self,t, curve):
        dx, dy = curve.evaluate_hodograph(t).flatten()
        return np.sqrt(dx**2 + dy**2)
    
    def get_total_length(self):
        total_length, _ = quad(self.get_arc_length_integrand, 0, 1, args=(self.curve,))
        return total_length
    
    def get_arc_length_from_start(self, t):
        return quad(self.get_arc_length_integrand, 0, t, args=(self.curve,))[0]
    
    def get_points(self):
        t_values = []
        for i in range(self.num_points):
            target_length = i * (self.get_total_length() / (self.num_points - 1))
            root = root_scalar(lambda t: self.get_arc_length_from_start(t) - target_length, bracket=[0, 1], method='bisect')
            t_values.append(root.root)

        points = np.array([self.curve.evaluate(t).flatten() for t in t_values])
        return points