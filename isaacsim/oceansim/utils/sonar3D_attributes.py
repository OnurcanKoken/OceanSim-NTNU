import numpy as np

hor_resolution = 0.85   # deg
vert_resolution = 1.60  # deg
hor_fov = 90            # deg
vert_fov = 40           # deg
refresh_rate = 5        # Hz
max_range = 15          # m
min_range = 0.2         # m

num_rows = int(vert_fov / vert_resolution)
num_columns = int(hor_fov / hor_resolution)
total_emitters = int(num_rows * num_columns)
print(f'num rows: {num_rows}, num cols: {num_columns}, total_emitters: {total_emitters}')

fire_step_Ns = round(1E9 / (total_emitters * refresh_rate))
print(f'firestep is {fire_step_Ns} Ns, times total emitters is {fire_step_Ns*total_emitters*1E-9} s, which is {1/(fire_step_Ns*total_emitters*1E-9)} Hz')

fireNs_list = [fire_step_Ns * i for i in range(total_emitters)]
channelId_list = [i + 1 for i in range(total_emitters)]
zeros_list = [0 for i in range(total_emitters)]
rays_per_line_list = [num_columns for i in range(num_rows)]

row_azimuth_deg = np.linspace(-hor_fov/2, hor_fov/2, num_columns).tolist()
assert len(row_azimuth_deg) == num_columns
azimuth_deg_list = []
for i in range(num_rows):
    azimuth_deg_list.extend(row_azimuth_deg)
assert azimuth_deg_list[:num_columns] == azimuth_deg_list[num_columns:2*num_columns]
assert len(azimuth_deg_list) == total_emitters
assert isinstance(azimuth_deg_list, list)

column_elevation_deg = np.linspace(-vert_fov/2, vert_fov/2, num_rows).tolist()
elevation_deg_list = []
for elev_deg in column_elevation_deg:
    elevation_deg_list.extend([elev_deg for i in range(num_columns)])
assert elevation_deg_list[num_columns] == column_elevation_deg[1]
assert len(elevation_deg_list) == total_emitters
assert isinstance(elevation_deg_list, list)

bank_list = []
for i in range(num_rows, 0, -1):
    bank_list.extend([i-1 for elem in range(num_columns)])
assert len(bank_list) == total_emitters

sensor_attributes = {
                     "omni:sensor:Core:nearRangeM": min_range,
                     "omni:sensor:Core:farRangeM": max_range,

                     "omni:sensor:Core:scanRateBaseHz": refresh_rate,
                     "omni:sensor:Core:reportRateBaseHz": refresh_rate,

                     "omni:sensor:Core:numberOfEmitters": total_emitters,
                     "omni:sensor:Core:numberOfChannels": total_emitters,
                     
                     "omni:sensor:Core:numLines": num_rows,
                     "omni:sensor:Core:numRaysPerLine": rays_per_line_list, 
                     
                    #  "omni:sensor:Core:emitterState:s001:azimuthDeg": [-2, -1, 1, 2, 3,
                    #                                                    -1.5, -0.5, 0.5, 1.5, 3,
                    #                                                    -2.4, -1.4, 1.4, 2.4, 3],
                    #  "omni:sensor:Core:emitterState:s001:elevationDeg": [
                    #                                                     -8, -8, -8, -8, -8,
                    #                                                     0, 0,  0,  0, 0,
                    #                                                         8,  8,  8, 8, 8
                    #                                                     ],

                     "omni:sensor:Core:emitterState:s001:azimuthDeg": azimuth_deg_list,
                     "omni:sensor:Core:emitterState:s001:elevationDeg": elevation_deg_list,

                     "omni:sensor:Core:emitterState:s001:fireTimeNs": fireNs_list,
                     "omni:sensor:Core:emitterState:s001:channelId": channelId_list,
                    #  "omni:sensor:Core:emitterState:s001:bank": [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     "omni:sensor:Core:emitterState:s001:bank": bank_list,
                     "omni:sensor:Core:emitterState:s001:distanceCorrectionM": zeros_list,
                     "omni:sensor:Core:emitterState:s001:focalDistM": zeros_list,
                     "omni:sensor:Core:emitterState:s001:horOffsetM": zeros_list,
                     "omni:sensor:Core:emitterState:s001:focalSlope": zeros_list,
                     "omni:sensor:Core:emitterState:s001:rangeId": zeros_list,
                     "omni:sensor:Core:emitterState:s001:reportRateDiv": zeros_list,
                     "omni:sensor:Core:emitterState:s001:vertOffsetM": zeros_list,
                     }