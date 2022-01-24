# Plot a comparison of the new SD1500 windows against the old one

import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams.update({'font.size': 22})


def main():

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.subplots()

    trigger = ("Th1", "Th2", "MoPS", "ToTd", "ToT", "Old")
    y_pos = np.arange(len(trigger))

    tlow1500 = np.flip(np.array([-1000, -750, -775, -525, -150, -100]))
    thigh1500 =  np.flip(np.array([2000, 1900, 2775, 3275, 450, 675]))
    twidth1500 = thigh1500 - tlow1500

    ax.barh(y_pos, left=tlow1500, width=twidth1500,
            color=("tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:orange"))

    ax.set_yticks(y_pos, labels=trigger)

    ax.set_xlim(left=-1500)

    plt.xlabel('Selection window (ns)')
    plt.ylabel('Trigger')

    plt.tight_layout()
    utils.savefig("plot_compare_1500", "Plot comparison with current SD1500 windows in ")


# Run starts here
if __name__ == "__main__":
    main()