# Calculate the performance of time window

import sys
import numpy as np
import json

import class_perf

# Weight factor to calculate the F-score
BETA = 1


def main():

    if len(sys.argv) != 4:
        print("Usage " + sys.argv[0] + "tlow thigh [json input file]")
        exit(1)

    # Read command line arguments
    tlow = float(sys.argv[1])
    thigh = float(sys.argv[2])
    input_filename = sys.argv[3]

    selection_window = (tlow, thigh)

    # Read data
    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    bin_time = np.array(data["bin_time"])

    print(f'ToT trigger')
    tot_data = data["tot"]
    histo_tot = np.array(tot_data["histo"])
    trigger_perf(bin_time, histo_tot, tot_data["pedestal"], selection_window)

    print(f'ToTd trigger')
    totd_data = data["totd"]
    histo_totd = np.array(totd_data["histo"])
    trigger_perf(bin_time, histo_totd, totd_data["pedestal"], selection_window)

    print(f'MoPS trigger')
    mops_data = data["mops"]
    histo_mops = np.array(mops_data["histo"])
    trigger_perf(bin_time, histo_mops, mops_data["pedestal"], selection_window)

    print(f'Th2 trigger')
    th2_data = data["th2"]
    histo_th2 = np.array(th2_data["histo"])
    trigger_perf(bin_time, histo_th2, th2_data["pedestal"], selection_window)

    print(f'Th1 trigger')
    th1_data = data["th1"]
    histo_th1 = np.array(th1_data["histo"])
    trigger_perf(bin_time, histo_th1, th1_data["pedestal"], selection_window)

    print(f'All triggers')
    histo_all = histo_tot + histo_totd + histo_mops + histo_th2 + histo_th1
    pedestal_all = tot_data["pedestal"] + totd_data["pedestal"] + mops_data["pedestal"] \
                   + th2_data["pedestal"] + th1_data["pedestal"]
    trigger_perf(bin_time, histo_all, pedestal_all, selection_window)


def trigger_perf(bin_time, histo, pedestal, selection_window):
    purity = class_perf.get_purity(bin_time, histo, pedestal, selection_window)
    print(f'Purity: {100 * purity:.3f}%')
    efficiency = class_perf.get_efficiency(bin_time, histo, pedestal, selection_window)
    print(f'Efficiency: {100 * efficiency:.3f}%')
#    f_score = class_perf.get_fscore(efficiency, purity, BETA)
#    print(f'F-score: {100 * f_score:.3f}%')


# Run starts here
if __name__ == "__main__":
    main()
