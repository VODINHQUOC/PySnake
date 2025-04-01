#!/bin/bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate

# Your terminal prompt should change, showing (venv) at the beginning
# Install packages as needed
pip install pygame numpy

# When you're done working, deactivate the environment
deactivate