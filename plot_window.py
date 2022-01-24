# Plot the residual histograms with the respective selection window
# A different plot is produced for each trigger type

import sys
import numpy as np
import matplotlib.pyplot as plt
import json

import utils

plt.rcParams.update({'font.size': 22})


def plot_window(bin_time, bin_counts, pedestal, threshold, tlow, thigh, trigger_label):

    plt.figure()
    plt.yscale("log")

    plt.text(0.9, 0.85, trigger_label, fontsize='large', ha='right', transform=plt.gca().transAxes)

    ymin = 1
    mask = np.all((tlow < bin_time, bin_time < thigh, bin_counts > ymin), axis=0)

    plt.plot(bin_time, bin_counts, drawstyle='steps', label='Data')
    plt.fill_between(bin_time[mask], y1=bin_counts[mask], y2=ymin, step="pre", alpha=0.5)  # selected
#    plt.fill_between(bin_time[mask], y1=bin_counts[mask], y2=pedestal, step="pre", alpha=1)  # signal selected

#    mask2 = np.all((-2250 < bin_time, bin_time < 5000), axis=0)
#    plt.fill_between(bin_time[mask2], y1=bin_counts[mask2], y2=pedestal, step="pre", alpha=0.25)   # signal

    x = (bin_time[0], bin_time[-1])
    plt.plot(x, (pedestal, pedestal), ls='--', color='tab:orange', label='Pedestal')
    plt.text(0.7, 1.3*pedestal, 'Pedestal', fontsize='small', color='black',
             transform=plt.gca().get_yaxis_transform())

    plt.plot(x, (threshold, threshold), ls='--', color='tab:orange', label='Threshold')
    plt.text(0.7, 1.2*threshold, 'Threshold', fontsize='small', color='black',
             transform=plt.gca().get_yaxis_transform())

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    plt.gca().set_ylim(bottom=ymin)
    plt.tight_layout()

    filename = "window_" + trigger_label.lower()
    utils.savefig(filename, "Selection window plotted in ")


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
