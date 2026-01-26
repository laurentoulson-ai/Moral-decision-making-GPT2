"""
Path configuration for the project.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths — point to project_root/data (not parent of project root)
MORAL_DATA = os.path.join(BASE_DIR, 'data', 'Moral_mixed_500.csv')
NEUTRAL_DATA = os.path.join(BASE_DIR, 'data', 'Neutral_500.csv')

# Output directories (stay under project_root/findings/...)
ACTIVATION_DIR = os.path.join(BASE_DIR, 'findings', 'discovery_phase', 'activations')
PLOT_DIR = os.path.join(BASE_DIR, 'findings', 'discovery_phase', 'plots')
STATS_DIR = os.path.join(BASE_DIR, 'findings', 'discovery_phase', 'stats')

# Create directories if missing
os.makedirs(ACTIVATION_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)