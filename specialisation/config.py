"""
Path configuration for the project.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths - where your paired CSV files are stored (point to project_root/data)
DATA_DIR = os.path.join(BASE_DIR, 'data')
VIRTUE_PAIRS_PATH = os.path.join(DATA_DIR, 'virtue_500.csv')
DEONT_PAIRS_PATH = os.path.join(DATA_DIR, 'deont_500.csv')
UTIL_PAIRS_PATH = os.path.join(DATA_DIR, 'util_500.csv')

# Output directories
ACTIVATION_DIR = os.path.join(BASE_DIR, 'findings', 'specialisation_phase', 'activations')
PLOT_DIR = os.path.join(BASE_DIR, 'findings', 'specialisation_phase', 'plots')
STATS_DIR = os.path.join(BASE_DIR, 'findings', 'specialisation_phase', 'stats')

# Create directories if missing
os.makedirs(ACTIVATION_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)
