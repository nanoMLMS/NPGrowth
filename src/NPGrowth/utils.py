import numpy as np

def read_xyz(filename):
    """Reads an XYZ file and returns atom types and positions."""
    with open(filename, "r") as f:
        lines = f.readlines()

    numAtoms = int(lines[0])  # First line: number of atoms
    atomData = [line.split() for line in lines[2:numAtoms + 2]]

    # Convert to numpy array (atom type, x, y, z)
    atomTypes = np.array([str(line[0]) for line in atomData])
    positions = np.array([[float(line[1]), float(line[2]), float(line[3])] for line in atomData])

    return atomTypes, positions