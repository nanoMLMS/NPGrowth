class Atom:
    def __init__(self, id, type, element, position):
        self.id = id
        self.type = type
        self.element = element
        self.position = position

def write_step(lines, timestep, atoms, f_out):
    f_out.write('ITEM: TIMESTEP\n')
    f_out.write(f'{timestep}\n')
    f_out.write('ITEM: NUMBER OF ATOMS\n')
    f_out.write(f'{len(atoms)}\n')
    for line in lines:
        f_out.write(line)
    f_out.write('ITEM: ATOMS id type element x y z\n')
    for atom in atoms:
        f_out.write(f'{atom.id} {atom.type} {atom.element} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n')

def process_timestep(file_in_name, file_out_name, callback):
    f_in = open(file_in_name, 'r')
    f_out = open(file_out_name, 'w')

    lines = f_in.readlines()
    n = 0

    while n < len(lines):
        timestep = int(lines[1 + n])
        n_atoms = int(lines[3 + n])
        atoms = []
        for i in range(n_atoms):
            atom_props = lines[9 + n + i].split()
            id = int(atom_props[0])
            type = int(atom_props[1])
            element = atom_props[2]
            position = [float(atom_props[3]), float(atom_props[4]), float(atom_props[5])]
            atoms.append(Atom(id, type, element, position))
        n += n_atoms + 9
        callback(atoms)
        write_step(lines[(n+4):(n+8)], timestep, atoms, f_out)
        print(timestep)

def callback(atoms):
    for atom in atoms:
        if atom.id > 4631:
            atom.type = 2

process_timestep('output/s_on_faces.lammpstrj', 'output/s_on_facespp.lammpstrj', callback)
