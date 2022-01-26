# Plot the purity and efficiency of two conditions specified in the input file

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

    # Load data
    purity = np.array(data["purity"])
    efficiency = np.array(data["efficiency"])

    # Plot purity and efficiency
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_yscale('logit')

    ymin = 0.5
    ax.set_ylim(bottom=ymin, top=.9999)

    x_pos = np.arange(len(data["label"]))

    ax.bar(x_pos-0.2, width=0.3, height=purity-ymin, bottom=ymin, label="Purity")
    ax.bar(x_pos+0.2, width=0.3, height=efficiency-ymin, bottom=ymin, label="Efficiency")

    ax.set_xticks(x_pos, labels=data["label"])
    ax.set_yticks((.5, 0.9, 0.99, 0.999), labels=("50%", '90%', '99%', '99.9%'))

    plt.legend(fontsize='small')

    plt.tight_layout()

    utils.savefig("pureffi_cmp", "Purity and efficiency plotted in ")


# Run starts here
if __name__ == "__main__":
    main()
