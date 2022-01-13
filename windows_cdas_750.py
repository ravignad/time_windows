# Calculate the performance of sd750 time windows in CDAS
# Values taken from the thesis of Alan Coleman (GAP_2018_054)

import sys
import numpy as np
import json

import class_perf

# Weight factor to calculate the F-score
BETA = 1


def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + "tlow thigh [json input file]")
        exit(1)

    # Read command line arguments
    input_filename = sys.argv[1]

    # Read data
    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    bin_time = np.array(data["bin_time"])

    print(f'ToT trigger')
    tot_data = data["tot"]
    histo_tot = np.array(tot_data["histo"])
    tot_window = (-397, 1454)
    pur_tot, eff_tot = trigger_perf(bin_time, histo_tot, tot_data["pedestal"], tot_window)

    print(f'ToTd trigger')
    totd_data = data["totd"]
    histo_totd = np.array(totd_data["histo"])
    totd_window = (-468, 2285)
    pur_totd, eff_totd = trigger_perf(bin_time, histo_totd, totd_data["pedestal"], totd_window)

    print(f'MoPS trigger')
    mops_data = data["mops"]
    histo_mops = np.array(mops_data["histo"])
    mops_window = (-477, 2883)
    pur_mops, eff_mops = trigger_perf(bin_time, histo_mops, mops_data["pedestal"], mops_window)

    print(f'Th2 trigger')
    th2_data = data["th2"]
    histo_th2 = np.array(th2_data["histo"])
    th2_window = (-485, 1379)
    pur_th2, eff_th2 = trigger_perf(bin_time, histo_th2, th2_data["pedestal"], th2_window)

    print(f'Th1 trigger')
    th1_data = data["th1"]
    histo_th1 = np.array(th1_data["histo"])
    th1_window = (-651, 2360)
    pur_th1, eff_th1 = trigger_perf(bin_time, histo_th1, th1_data["pedestal"], th1_window)

    print(f'All triggers')

    # Count residuals within the time_range
    ntot = histo_tot.sum()
    ntotd = histo_totd.sum()
    nmops = histo_mops.sum()
    nth2 = histo_th2.sum()
    nth1 = histo_th1.sum()
    nresiduals = ntot + ntotd + nmops + nth2 + nth1

    purity = (ntot * pur_tot + ntotd * pur_totd + nmops * pur_mops
              + nth2 * pur_th2 + nth1 * pur_th1) / nresiduals

    efficiency = (ntot * eff_tot + ntotd * eff_totd + nmops * eff_mops
                  + nth2 * eff_th2 + nth1 * eff_th1) / nresiduals

    print(f'Purity: {100 * purity:.2f}%')
    print(f'Efficiency: {100 * efficiency:.2f}%')


def trigger_perf(bin_time, histo, pedestal, selection_window):
    purity = class_perf.get_purity(bin_time, histo, pedestal, selection_window)
    print(f'Purity: {100 * purity:.2f}%')
    efficiency = class_perf.get_efficiency(bin_time, histo, pedestal, selection_window)
    print(f'Efficiency: {100 * efficiency:.2f}%')
    f_score = class_perf.get_fscore(efficiency, purity, BETA)
    print(f'F-score: {100 * f_score:.2f}%')

    return purity, efficiency


# Run starts here
if __name__ == "__main__":
    main()
