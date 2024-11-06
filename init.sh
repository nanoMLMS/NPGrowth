#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating virtualenv"
cd "$DIR"
python3 -m venv .venv 
source .venv/bin/activate

echo "Installing requirements"
pip install -r requirements.txt

# Update and initialize the LAMMPS submodule
echo "Initializing LAMMPS submodule..."
git submodule update --init --recursive

# Navigate to the LAMMPS source directory
cd "$DIR"
cd external/lammps/src

# Load LAMMPS build dependencies (optional)
# You can add instructions here to load required modules or set up the environment

# Build LAMMPS with desired packages (modify this to suit your needs)
echo "Building LAMMPS..."
make yes-PYTHON
make yes-MANYBODY
make mode=shared mpi

# Install Python bindings for LAMMPS
echo "Installing LAMMPS Python bindings..."
make install-python

# Return to the root of the project
echo "LAMMPS is ready and Python bindings are installed!"

echo "All finished!"