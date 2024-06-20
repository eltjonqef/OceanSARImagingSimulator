import sys
import os
if len(sys.argv)!=3:
    print("Usage: python3 generateParameters.py input.yaml output_folder")

OUTPUT_FOLDER = os.path.join(os.getcwd(), f"{sys.argv[2]}")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
wind_speeds=[5,10,15]
wind_directions=[0,45,90]
incidence_angles=[20,30,50]
betas=[10,50, 100]
with open(sys.argv[1], 'r') as file:
    template = file.read()
i=0
for wind_speed in wind_speeds:
    for wind_direction in wind_directions:
        for incidence_angle in incidence_angles:
            for beta in betas:
                values={
                    'wind_speed':wind_speed,
                    'wind_direction':wind_direction,
                    'incidence_angle':incidence_angle,
                    'beta':beta
                }
                filled_template=template.format(**values)
                i+=1
                with open(f'{OUTPUT_FOLDER}/{i}.yml','w') as file:
                    file.write(filled_template)
print(f"Generated {i} parameter files.")