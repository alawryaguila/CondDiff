#!/bin/bash

# Navigate to the 'autoencoders' directory and install
cd autoencoders
pip install -e ./

# Return to the previous directory
cd ..

# Install dependencies from environment.txt
pip install -r environment.txt

# Upgrade diffusers with the torch extra dependencies
pip install --upgrade diffusers[torch]