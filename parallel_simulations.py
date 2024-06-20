import concurrent.futures
import subprocess
import os
import glob
import sys

if len(sys.argv)!=3:
    print("Usage: python3 parallel_simulations.py config_folder output_folder")

config_folder = sys.argv[1]

config_files = glob.glob(os.path.join(config_folder, '*'))

def run_simulation(config_file):
    subprocess.run(['python3', 'simulation.py', config_file, sys.argv[2]])
    print(f"Finished parameter file {config_file}")

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(run_simulation, config_files)
