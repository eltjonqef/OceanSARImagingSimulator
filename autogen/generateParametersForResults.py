import sys
import os
if len(sys.argv)!=3:
    print("Usage: python3 generateParametersForResults.py input.yaml output_folder")

OUTPUT_FOLDER = os.path.join(os.getcwd(), f"{sys.argv[2]}")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
cases =[
    [5,45,1,30],
    [5,45,10,30],
    [5,45,30,30],
    [5,45,70,30],
    [5,45,100,30],
    [5,45,50,30],
    [15,45,50,30],
    [10,0,50,30],
    [10,45,50,30],
    [10,90,50,30],
    [10,45,10,30],
    [10,45,100,30],
]

with open(sys.argv[1], 'r') as file:
    template = file.read()
i=0
for case in cases:
    values={
        'wind_speed':case[0],
        'wind_direction':case[1],
        'incidence_angle':case[3],
        'beta':case[2]
    }
    filled_template=template.format(**values)
    i+=1
    with open(f'{OUTPUT_FOLDER}/{i}.yml','w') as file:
        file.write(filled_template)
print(f"Generated {i} parameter files.")