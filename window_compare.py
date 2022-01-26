# Plot a comparison of the new windows by trigger type against the old single window

import sys
import numpy as np
import matplotlib.pyplot as plt
import json

import utils

plt.rcParams.update({'font.size': 22})


def main():

    if len(sys.argv) != 4:
        print("Usage " + sys.argv[0] + "tlow thigh [json input file]")
        exit(1)

    # Read command line arguments
    tlow_old = float(sys.argv[1])
    thigh_old = float(sys.argv[2])
    input_filename = sys.argv[3]

    # Read data
    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    tot_data = data["tot"]
    totd_data = data["totd"]
    mops_data = data["mops"]
    th2_data = data["th2"]
    th1_data = data["th1"]

    tlow = np.flip(np.array([tlow_old, tot_data["tlow"], totd_data["tlow"], mops_data["tlow"], th2_data["tlow"],
                             th1_data["tlow"]]))
    thigh = np.flip(np.array([thigh_old, tot_data["thigh"], totd_data["thigh"], mops_data["thigh"], th2_data["thigh"],
                              th1_data["thigh"]]))
    twidth = thigh - tlow

    # Plot
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.subplots()

    trigger = ("Th1", "Th2", "MoPS", "ToTd", "ToT", "Old")
    y_pos = np.arange(len(trigger))

    ax.barh(y_pos, left=tlow, width=twidth,
            color=("tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:orange"))

    ax.set_yticks(y_pos, labels=trigger)

    ax.set_xlim(left=-1500)

    plt.xlabel('Selection window (ns)')
    plt.ylabel('Trigger')

    plt.tight_layout()
    utils.savefig("window_compare", "Plot comparison with old window in ")


# Run starts here
if __name__ == "__main__":
    main()
