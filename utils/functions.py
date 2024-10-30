import numpy as np
import math
import bezier
from scipy.integrate import quad
from scipy.optimize import root_scalar

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Parameters:
    -----------
    r : float
        The radial distance from the origin (radius).
    theta : float
        The polar angle (in radians), measured from the positive z-axis.
    phi : float
        The azimuthal angle (in radians), measured from the positive x-axis in the xy-plane.
    
    Returns:
    --------
    numpy.ndarray
        A 3D vector (x, y, z) representing the random position in Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x, y, z])

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    ----------
    x : float
        The x-coordinate in Cartesian coordinates.
    y : float
        The y-coordinate in Cartesian coordinates.
    z : float
        The z-coordinate in Cartesian coordinates.

    Returns:
    -------
    tuple
        A tuple containing (r, theta, phi) in spherical coordinates:
        - r: Radial distance.
        - theta: Polar angle, in radians, from 0 to pi.
        - phi: Azimuthal angle, in radians, from 0 to 2*pi.
    """
    # Compute the radial distance
    r = math.sqrt(x**2 + y**2 + z**2)
    
    # Compute the polar angle (handling r=0 to avoid division by zero)
    theta = math.acos(z / r) if r != 0 else 0

    # Compute the azimuthal angle (handling x=y=0 to avoid division by zero)
    phi = math.atan2(y, x)
    
    return np.array([r, theta, phi])

def atomic_radius(lattice_constant, structure='fcc'):
    if structure == 'fcc':
        # Face-Centered Cubic (FCC)
        return (math.sqrt(2) / 4) * lattice_constant
    elif structure == 'bcc':
        # Body-Centered Cubic (BCC)
        return (math.sqrt(3) / 4) * lattice_constant
    elif structure == 'hcp':
        # Hexagonal Close-Packed (HCP)
        return 0.5 * lattice_constant
    else:
        raise ValueError("Unknown crystal structure")
