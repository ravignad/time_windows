# Plot the event histograms for all trigger types
# The pedestals are plot separately

import sys
import numpy as np
import matplotlib.pyplot as plt
import json

import utils


def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [json input file]")
        exit(1)

    # Read  data
    input_filename = sys.argv[1]

    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    tot_data = data["tot"]
    totd_data = data["totd"]
    mops_data = data["mops"]
    th2_data = data["th2"]
    th1_data = data["th1"]

    bin_time = np.array(data["bin_time"])
    histo_tot = np.array(tot_data["histo"])
    histo_totd = np.array(totd_data["histo"])
    histo_mops = np.array(mops_data["histo"])
    histo_th2 = np.array(th2_data["histo"])
    histo_th1 = np.array(th1_data["histo"])

    plt.figure()
    plt.yscale("log")

    plt.plot(bin_time, histo_tot, drawstyle='steps', lw=0.5, label='ToT')
    plt.plot(bin_time, histo_totd, drawstyle='steps', lw=0.5, label='ToTd')
    plt.plot(bin_time, histo_mops, drawstyle='steps', lw=0.5, label='MoPS')
    plt.plot(bin_time, histo_th2, drawstyle='steps', lw=0.5, label='Th2')
    plt.plot(bin_time, histo_th1, drawstyle='steps', lw=0.5, label='Th1')

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    plt.legend()
    utils.savefig("residual", "Residuals plotted in ")

    # Plot pedestals
    plt.figure()

    tot_pedestal_range = tot_data["pedestal_range"]
    tot_mask = np.all((tot_pedestal_range[0] < bin_time, bin_time < tot_pedestal_range[1]), axis=0)

    totd_pedestal_range = totd_data["pedestal_range"]
    totd_mask = np.all((totd_pedestal_range[0] < bin_time, bin_time < totd_pedestal_range[1]), axis=0)

    mops_pedestal_range = mops_data["pedestal_range"]
    mops_mask = np.all((mops_pedestal_range[0] < bin_time, bin_time < mops_pedestal_range[1]), axis=0)

    th2_pedestal_range = th2_data["pedestal_range"]
    th2_mask = np.all((th2_pedestal_range[0] < bin_time, bin_time < th2_pedestal_range[1]), axis=0)

    th1_pedestal_range = th1_data["pedestal_range"]
    th1_mask = np.all((th1_pedestal_range[0] < bin_time, bin_time < th1_pedestal_range[1]), axis=0)

    xmin = min(tot_pedestal_range[0], totd_pedestal_range[0], mops_pedestal_range[0],
               th2_pedestal_range[0], th1_pedestal_range[0])

    xmax = max(tot_pedestal_range[1], totd_pedestal_range[1], mops_pedestal_range[1],
               th2_pedestal_range[1], th1_pedestal_range[1])

    mask = np.all((xmin < bin_time, bin_time < xmax), axis=0)

    plt.plot(bin_time[mask], histo_tot[mask], drawstyle='steps', lw=0.5, color='lightgray')
    plt.plot(bin_time[mask], histo_totd[mask], drawstyle='steps', lw=0.5, color='lightgray')
    plt.plot(bin_time[mask], histo_mops[mask], drawstyle='steps', lw=0.5, color='lightgray')
    plt.plot(bin_time[mask], histo_th2[mask], drawstyle='steps', lw=0.5, color='lightgray')
    plt.plot(bin_time[mask], histo_th1[mask], drawstyle='steps', lw=0.5, color='lightgray')

    plt.plot(bin_time[tot_mask], histo_tot[tot_mask], drawstyle='steps', lw=0.5, label='ToT')
    plt.plot(bin_time[totd_mask], histo_totd[totd_mask], drawstyle='steps', lw=0.5, label='ToTd')
    plt.plot(bin_time[mops_mask], histo_mops[mops_mask], drawstyle='steps', lw=0.5, label='MoPS')
    plt.plot(bin_time[th2_mask], histo_th2[th2_mask], drawstyle='steps', lw=0.5, label='Th2')
    plt.plot(bin_time[th1_mask], histo_th1[th1_mask], drawstyle='steps', lw=0.5, label='Th1')

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    # Plot fitted pedestals
    plt.gca().set_prop_cycle(None)

    x = (bin_time[tot_mask][0], bin_time[tot_mask][-1])
    y = (tot_data["pedestal"], tot_data["pedestal"])
    plt.plot(x, y, lw=0.5)

    x = (bin_time[totd_mask][0], bin_time[totd_mask][-1])
    y = (totd_data["pedestal"], totd_data["pedestal"])
    plt.plot(x, y, lw=0.5)

    x = (bin_time[mops_mask][0], bin_time[mops_mask][-1])
    y = (mops_data["pedestal"], mops_data["pedestal"])
    plt.plot(x, y, lw=0.5)

    x = (bin_time[th2_mask][0], bin_time[th2_mask][-1])
    y = (th2_data["pedestal"], th2_data["pedestal"])
    plt.plot(x, y, lw=0.5)

    x = (bin_time[th1_mask][0], bin_time[th1_mask][-1])
    y = (th1_data["pedestal"], th1_data["pedestal"])
    plt.plot(x, y, lw=0.5)

    utils.savefig("pedestal", "Pedestals plotted in ")


# Run starts here
if __name__ == "__main__":
    main()
