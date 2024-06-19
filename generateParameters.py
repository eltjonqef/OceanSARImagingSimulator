wind_speeds=[5,10,15]
wind_directions=[0,45,90]
incidence_angles=[20,30,50]
betas=[0,50, 10]
with open('template.yaml', 'r') as file:
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
                with open(f'params/{i}.yml','w') as file:
                    file.write(filled_template)
print(f"Generated {i} parameter files.")