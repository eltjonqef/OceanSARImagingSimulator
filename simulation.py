from plots import plots
from parameters import parameters
import sys
import os

if len(sys.argv)!=3:
    print("Usage:\n\t\tpython3 simulation.py parameter_file.yml output_folder")
    # exit()
CONFIG_FILE= sys.argv[1]
OUTPUT_FOLDER = os.path.join(os.getcwd(), f"{sys.argv[2]}/{CONFIG_FILE.split('.')[0].split('/')[-1]}")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
params=parameters(CONFIG_FILE)

#%% Surface Generation
from surface import surfaceGenerator
surface=surfaceGenerator(params)
surface.generate()
# surface.animate()

#%% Sar Imaging
from SAR_imaging import SAR_imaging
sar=SAR_imaging(surface, params)
sar.generate()
plot=plots(OUTPUT_FOLDER, sar)