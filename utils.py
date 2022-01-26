
import math
import numpy as np
from matplotlib import pyplot as plt


# Map trigger code to the trigger type
# Trigger hierarchy ToT -> ToTd -> MoPs -> Th2 -> Th1
def get_trigger_class(trigger_code):
    if trigger_code & 4 != 0:     # bit 3
        return 'ToT'
    elif trigger_code & 8 != 0:  # bit 4
        return 'ToTd'
    elif trigger_code & 16 != 0:  # bit 5
        return 'MoPS'
    elif trigger_code & 2 != 0:  # bit 2
        return 'Th2'
    elif trigger_code & 1 != 0:  # bit 1
        return 'Th1'
    else:
        return 'None'


def savefig(filename, message):

    print(message + " " + filename + ".eps")
    plt.savefig(filename + ".eps")
    print("Selection window plotted in " + filename + ".jpg")
    plt.savefig(filename + ".jpg")
    print("Selection window plotted in " + filename + ".pdf")
    plt.savefig(filename + ".pdf")
    print("Selection window plotted in " + filename + ".svg")
    plt.savefig(filename + ".svg")

def polar2car(theta, phi):
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return (x, y, z)

""" Returns the angle in degrees between two directions"""
def directions_angle(theta1, phi1, theta2, phi2):
    v1 = polar2car(math.radians(theta1), math.radians(phi1))
    v2 = polar2car(math.radians(theta2), math.radians(phi2))
    return math.degrees(np.arccos(np.dot(v1, v2)))


# Run starts here
if __name__ == "__main__":

    theta1, phi1 = 11.5, 66.3
    theta2, phi2 = 17.9, 49.6

    delta_direction = directions_angle(theta1, phi1, theta2, phi2)
    print(f'Angle between directions ({theta1}, {phi1}) and ({theta2}, {phi2}) = {delta_direction:.1f} degrees')



