#!/bin/bash

# Update and initialize the LAMMPS submodule
echo "Initializing LAMMPS submodule..."
git submodule update --init --recursive

# Navigate to the LAMMPS source directory
cd external/lammps/src

# Load LAMMPS build dependencies (optional)
# You can add instructions here to load required modules or set up the environment

# Build LAMMPS with desired packages (modify this to suit your needs)
echo "Building LAMMPS..."
make yes-PYTHON
make yes-MANYBODY
make mode=shared serial

# Install Python bindings for LAMMPS
echo "Installing LAMMPS Python bindings..."
make install-python

# Return to the root of the project
cd ../../

echo "LAMMPS is ready and Python bindings are installed!"
