# Plot a comparison of the new sd750 windows against those in the sd750 paper

import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams.update({'font.size': 22})


def main():

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.subplots()

    trigger = ("Th1", "Th2", "MoPS", "ToTd", "ToT")
    y_pos = np.arange(len(trigger))

    tlow750 = np.flip(np.array([-775, -550, -350, -225, -225]))
    thigh750 = np.flip(np.array([1050, 2300, 1950, 450, 550]))
    twidth750 = thigh750 - tlow750

    tlow_paper = np.flip(np.array([-397, -468, -477, -485, -485]))
    thigh_paper = np.flip(np.array([1454, 2285, 2883, 1379, 1379]))
    twidth_paper = thigh_paper - tlow_paper

    ax.barh(y_pos+0.2, height=0.3, left=tlow750, width=twidth750, label="This work")
    ax.barh(y_pos-0.2, height=0.3, left=tlow_paper, width=twidth_paper, label="Paper")

    ax.set_yticks(y_pos, labels=trigger)

    ax.set_xlim(left=-1000, right=4000)

    plt.xlabel('Selection window (ns)')
    plt.ylabel('Trigger')
    plt.legend(fontsize='small', loc='lower right')

    plt.tight_layout()
    utils.savefig("plot_compare_750", "A comparison of new and paper windows for the SD750 plotted in ")


# Run starts here
if __name__ == "__main__":
    main()