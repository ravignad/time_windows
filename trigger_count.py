# Draw a pie chart with the distribution of the number of triggers

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

    ntot = data["tot"]["counts"]
    ntotd = data["totd"]["counts"]
    nmops = data["mops"]["counts"]
    nth2 = data["th2"]["counts"]
    nth1 = data["th1"]["counts"]

    labels = 'ToT', 'ToTd', 'MoPS', 'Th2', 'Th1'
    sizes = [ntot, ntotd, nmops, nth2, nth1]

    # Plot
    fig = plt.figure(figsize=(3., 2.4))
    ax = fig.subplots()

    ax.pie(sizes, labels=labels, startangle=90, counterclock=False)
    ax.axis('equal')

    plt.tight_layout()

    utils.savefig("trigger_count", "Number of triggers plotted in ")


# Run starts here
if __name__ == "__main__":
    main()