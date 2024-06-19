import concurrent.futures
import subprocess
import os
import glob

config_folder = 'params/'

config_files = glob.glob(os.path.join(config_folder, '*'))

def run_simulation(config_file):
    subprocess.run(['python3', 'simulation.py', config_file, 'results'])

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(run_simulation, config_files)
