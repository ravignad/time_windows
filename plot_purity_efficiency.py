# Plot the purity and efficiency

import sys
import numpy as np
import matplotlib.pyplot as plt
import json

import utils

plt.rcParams.update({'font.size': 22})

def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [json input file]")
        exit(1)

    # Read  data
    input_filename = sys.argv[1]

    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    # Read purity data
    tot_pur = data["tot"]["purity"]
    totd_pur = data["totd"]["purity"]
    mops_pur = data["mops"]["purity"]
    th2_pur = data["th2"]["purity"]
    th1_pur = data["th1"]["purity"]
    all_pur = data["purity"]
    purity = np.array([all_pur, th1_pur, th2_pur, mops_pur, totd_pur, tot_pur])

    tot_effi = data["tot"]["efficiency"]
    totd_effi = data["totd"]["efficiency"]
    mops_effi = data["mops"]["efficiency"]
    th2_effi = data["th2"]["efficiency"]
    th1_effi = data["th1"]["efficiency"]
    all_effi = data["efficiency"]
    efficiency = np.array([all_effi, th1_effi, th2_effi, mops_effi, totd_effi, tot_effi])

    # Plot purity and efficiency
    fig = plt.figure(figsize=(9.6, 4.8))
    ax = fig.subplots()
    ax.set_xscale('logit')

    xmin = 0.5

    trigger = ("$\mathbf{All}$", "Th1", "Th2", "MoPS", "ToTd", "ToT")
    y_pos = np.arange(len(trigger))

    ax.barh(y_pos+0.2, height=0.3, width=purity-xmin, left=xmin, label="Purity")
    ax.barh(y_pos-0.2, height=0.3, width=efficiency-xmin, left=xmin, label="Efficiency")

    ax.set_yticks(y_pos, labels=trigger)
    ax.set_xticks((.5, 0.9, 0.99, 0.999), labels=("50%", '90%', '99%', '99.9%'))

    plt.ylabel('Trigger')
    plt.legend(fontsize='small', bbox_to_anchor=(1., 1.))

    plt.tight_layout()

    utils.savefig("purity_efficiency", "Purity and efficiency plotted in ")

# Run starts here
if __name__ == "__main__":
    main()