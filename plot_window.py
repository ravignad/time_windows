# Plot the residual histograms with the respective selection window
# A different plot is produced for each trigger type

import sys
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_window(bin_time, bin_counts, pedestal, threshold, tlow, thigh, trigger_label):

    plt.figure()
    ax = plt.gca()
    plt.yscale("log")

    plt.text(0.9, 0.9, trigger_label, fontsize='large', ha='right', transform=ax.transAxes)

    mask = np.all((tlow < bin_time, bin_time < thigh), axis=0)

    plt.plot(bin_time, bin_counts, drawstyle='steps', lw=0.5, label='Data')
    plt.fill_between(bin_time[mask], bin_counts[mask], step="pre", alpha=0.4)

    x = (bin_time[0], bin_time[-1])
    p = plt.plot(x, (pedestal, pedestal), lw=0.5, ls='--', label='Pedestal')
    plt.text(bin_time[mask][0], 0.9*pedestal, 'Pedestal', fontsize='small', va='top', color=p[0].get_color())

    p = plt.plot(x, (threshold, threshold), lw=0.5, ls='--', label='Threshold')
    plt.text(0.6, 1.1*threshold, 'Threshold', fontsize='small', color=p[0].get_color(),
             transform=ax.get_yaxis_transform())

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    filename = "window_" + trigger_label.lower()
    savefig(filename)


def savefig(filename):

    print("Selection window plotted in " + filename + ".eps")
    plt.savefig(filename + ".eps")
    print("Selection window plotted in " + filename + ".jpg")
    plt.savefig(filename + ".jpg")
    print("Selection window plotted in " + filename + ".pdf")
    plt.savefig(filename + ".pdf")

def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [json input file]")
        exit(1)

    # Read  data
    input_filename = sys.argv[1]

    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    bin_time = np.array(data["bin_time"])

    tot_data = data["tot"]
    histo_tot = np.array(tot_data["histo"])
    plot_window(bin_time, histo_tot, tot_data["pedestal"], tot_data["threshold"],
                tot_data["tlow"], tot_data["thigh"], "ToT")

    totd_data = data["totd"]
    histo_totd = np.array(totd_data["histo"])
    plot_window(bin_time, histo_totd, totd_data["pedestal"], totd_data["threshold"],
                totd_data["tlow"], totd_data["thigh"], "ToTd")

    mops_data = data["mops"]
    histo_mops = np.array(mops_data["histo"])
    plot_window(bin_time, histo_mops, mops_data["pedestal"], mops_data["threshold"],
                mops_data["tlow"], mops_data["thigh"], "MoPS")

    th2_data = data["th2"]
    histo_th2 = np.array(th2_data["histo"])
    plot_window(bin_time, histo_th2, th2_data["pedestal"], th2_data["threshold"],
                th2_data["tlow"], th2_data["thigh"], "Th2")

    th1_data = data["th1"]
    histo_th1 = np.array(th1_data["histo"])
    plot_window(bin_time, histo_th1, th1_data["pedestal"], th1_data["threshold"],
                th1_data["tlow"], th1_data["thigh"], "Th1")


# Run starts here
if __name__ == "__main__":
    main()
