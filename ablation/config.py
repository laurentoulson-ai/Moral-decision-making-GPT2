"""
Path configuration for the project.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths - point to project_root/data
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Where specialisation phase writes its stats (used as source for ablation inputs)
SPECIALISATION_STATS_DIR = os.path.join(BASE_DIR, 'findings', 'specialisation_phase', 'stats')

# Output directories for ablation
STATS_DIR = os.path.join(BASE_DIR, 'findings', 'ablation_phase', 'stats')

# Create directories if missing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPECIALISATION_STATS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)



