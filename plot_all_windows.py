# Plot the windows of the three SD arrays

import sys
import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams.update({'font.size': 22})

def main():

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.subplots()

    trigger = ("Th1", "Th2", "MoPS", "ToTd", "ToT")
    y_pos = np.arange(len(trigger))

    tlow1500 = np.flip(np.array([-750, -775, -525, -150, -100]))
    thigh1500 = np.flip(np.array([1900, 2775, 3275, 450, 675]))
    twidth1500 = thigh1500 - tlow1500

    tlow750 = np.flip(np.array([-775, -550, -350, -225, -225]))
    thigh750 = np.flip(np.array([1050, 2300, 1950, 450, 550]))
    twidth750 = thigh750 - tlow750

    tlow433 = np.flip(np.array([-925, -725, -300, -350, -225]))
    thigh433 = np.flip(np.array([925, 1725, 1425, 475, 500]))
    twidth433 = thigh433 - tlow433

    ax.barh(y_pos+0.25, height=0.20, left=tlow1500, width=twidth1500, label="SD1500")
    ax.barh(y_pos, height=0.20, left=tlow750, width=twidth750, label="SD750")
    ax.barh(y_pos-0.25, height=0.20, left=tlow433, width=twidth433, label="SD433")

    ax.set_yticks(y_pos, labels=trigger)

    ax.set_xlim(left=-1000)

    plt.xlabel('Selection window (ns)')
    plt.ylabel('Trigger')
    plt.legend(fontsize='small')

    plt.tight_layout()
    utils.savefig("plot_all_windows", "Windows of the three arrays plotted in ")


# Run starts here
if __name__ == "__main__":
    main()