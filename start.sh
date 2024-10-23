#!/bin/bash

# Navigate to the directory of the script
cd "$(dirname "$0")"

# Activate the Python virtual environment and run the Python script
. venv/bin/activate && python main.py

# Run balena-idle if app exits (helpful for debug)
balena-idle