import os
import numpy as np
wind_speeds=[5,10,15]
wind_directions=[0,45,90]
incidence_angles=[20,30,50]
betas=[10,50, 100]
with open('template_latex_results.yaml', 'r') as file:
    template = file.read()
i=0
OUTPUT_FOLDER='params'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
for wind_speed in wind_speeds:
    for wind_direction in wind_directions:
        for incidence_angle in incidence_angles:
            for beta in betas:
                i+=1
                values={
                    'wind_speed':wind_speed,
                    'wind_direction':wind_direction,
                    'incidence_angle':incidence_angle,
                    'beta':beta,
                    'case':i
                }
                filled_template=template.format(**values)
                print(filled_template)
