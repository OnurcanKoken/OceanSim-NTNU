import numpy as np

sensor_attributes = {'omni:sensor:Core:validStartAzimuthDeg': 330,
                    'omni:sensor:Core:validEndAzimuthDeg': 30}

TRANSDUCER_FREQ = 1200000 # Hz
HORIZONTAL_FOV = 90 # deg
VERTICAL_FOV = 40 # deg
MAX_RANGE = 15 # meters
MIN_RANGE = 0.02 # meters
RANGE_RESOLUTION = 0.0015 # meters
BEAM_SEPARATION_HOR = 0.35 # deg
BEAM_SEPERATION_VERT = 0.60 # deg
ANGULAR_RESOLUTION_HOR = 0.85 # deg
ANGULAR_RESOLUTION_VERT = 1.60 # deg
UPDATE_RATE = 5 # Hz 

def calculate_numberOfEmitters(step_hor_deg, range_hor_deg, step_vert_deg, range_vert_deg):
    num_horizontal_elem = round(range_hor_deg/step_hor_deg)
    num_vertical_elem = round(range_vert_deg/step_vert_deg)
    return num_horizontal_elem, num_vertical_elem

def calculate_azimuthDeg(horizontal_fov, num_emitters_horizontal, num_emitters_vertical):
    half_horizontal_fov = horizontal_fov / 2
    row_azimuth_deg = np.linspace(-half_horizontal_fov, half_horizontal_fov, num_emitters_horizontal)

    azimuth_deg_matrix = np.zeros((num_emitters_vertical, num_emitters_horizontal))
    for i in range(num_emitters_vertical):
        azimuth_deg_matrix[i] = row_azimuth_deg
    azimuthDeg_list = azimuth_deg_matrix.flatten().tolist()
    print(len(azimuth_deg_matrix[0]))
    return azimuthDeg_list


if __name__ == '__main__':
    num_hor_emitters, num_vert_emitters = calculate_numberOfEmitters(step_hor_deg=ANGULAR_RESOLUTION_HOR,
                                                range_hor_deg=HORIZONTAL_FOV,
                                                step_vert_deg=ANGULAR_RESOLUTION_VERT,
                                                range_vert_deg=VERTICAL_FOV)
    total_emitters = num_hor_emitters * num_vert_emitters

    azimuth_deg = calculate_azimuthDeg(HORIZONTAL_FOV, num_hor_emitters, num_vert_emitters)