# Calculate the time windows to select the SD detectors that participated in the reconstruction of events
# A difference selection window is calculated for the trigger types: ToT, ToTd, MoPS, Th2, and Th1

import sys
import math
import numpy as np
import pandas
import json

import class_perf
import utils

# All times in nanoseconds

# Global constants
EVENT_RANGE = (140000000000, 182440000000)
BIN_PURITY = 0.95
PLOT_TYPE = ".pdf"
PEDESTAL_RANGE = {"tot": (5000, 8000),
                  "totd": (9000, 12000),
                  "mops": (9000, 12000),
                  "th2": (5000, 8000),
                  "th1": (5000, 8000),
                  }
TIME_RANGE = (-6000, 12000)    # Range of the residual histograms
BIN_WIDTH = 25  # Bin width of the residual histograms


def main():

    if len(sys.argv) != 3:
        print("Usage " + sys.argv[0] + " [residual file] [json output file]")
        exit(1)

    # Read  data
    residual_file = sys.argv[1]
    output_filename = sys.argv[2]
    df = pandas.read_csv(residual_file, names=('event', 'station', 'residual', 'trigger_code'))

    # Select 10% of the data to speed up testing
    # df = df.sample(frac=0.1)

    # Select data from Jan 1, 2014 to Aug 31, 2018 as per sd750 paper
    df = df[(EVENT_RANGE[0] < df['event']) & (df['event'] < EVENT_RANGE[1])]
    print(f'Number of selected residuals: {len(df.index)}')

    # Add trigger class
    df['trigger_class'] = df['trigger_code'].apply(utils.get_trigger_class)

    bin_edges = np.arange(start=TIME_RANGE[0], stop=TIME_RANGE[1], step=BIN_WIDTH)

    histo_tot, _ = np.histogram(df.loc[df['trigger_class'] == 'ToT', 'residual'], bins=bin_edges)
    histo_totd, _ = np.histogram(df.loc[df['trigger_class'] == 'ToTd', 'residual'], bins=bin_edges)
    histo_mops, _ = np.histogram(df.loc[df['trigger_class'] == 'MoPS', 'residual'], bins=bin_edges)
    histo_th2, _ = np.histogram(df.loc[df['trigger_class'] == 'Th2', 'residual'], bins=bin_edges)
    histo_th1, _ = np.histogram(df.loc[df['trigger_class'] == 'Th1', 'residual'], bins=bin_edges)

    # Count residuals within the time_range
    ntot = histo_tot.sum()
    ntotd = histo_totd.sum()
    nmops = histo_mops.sum()
    nth2 = histo_th2.sum()
    nth1 = histo_th1.sum()
    nresiduals = ntot + ntotd + nmops + nth2 + nth1

    # Bin centers
    bin_time = (bin_edges[:-1]+bin_edges[1:])/2

    print('ToT trigger')
    print(f'Number of ToT: {ntot} ({100*ntot/nresiduals:.1f}%)')
    pedestal_tot = get_pedestal(bin_time, histo_tot, PEDESTAL_RANGE["tot"])
    threshold_tot, tlow_tot, thigh_tot, pur_tot, effi_tot = get_window(bin_time, histo_tot, pedestal_tot)

    tot_output = {
        "threshold": threshold_tot,
        "tlow": tlow_tot,
        "thigh": thigh_tot,
        "purity": pur_tot,
        "efficiency": effi_tot,
        "histo": histo_tot.tolist(),
        "pedestal": pedestal_tot,
        "pedestal_range": PEDESTAL_RANGE["tot"]
    }

    print('ToTd trigger')
    print(f'Number of ToTd: {ntotd} ({100*ntotd/nresiduals:.1f}%)')
    pedestal_totd = get_pedestal(bin_time, histo_totd, PEDESTAL_RANGE["totd"])
    threshold_totd, tlow_totd, thigh_totd, pur_totd, effi_totd = get_window(bin_time, histo_totd, pedestal_totd)

    totd_output = {
        "threshold": threshold_totd,
        "tlow": tlow_totd,
        "thigh": thigh_totd,
        "purity": pur_totd,
        "efficiency": effi_totd,
        "histo": histo_totd.tolist(),
        "pedestal": pedestal_totd,
        "pedestal_range": PEDESTAL_RANGE["totd"]
    }

    print('MoPS trigger')
    print(f'Number of MoPS: {nmops} ({100*nmops/nresiduals:.1f}%)')
    pedestal_mops = get_pedestal(bin_time, histo_mops, PEDESTAL_RANGE["mops"])
    threshold_mops, tlow_mops, thigh_mops, pur_mops, effi_mops = get_window(bin_time, histo_mops, pedestal_mops)

    mops_output = {
        "threshold": threshold_mops,
        "tlow": tlow_mops,
        "thigh": thigh_mops,
        "purity": pur_mops,
        "efficiency": effi_mops,
        "histo": histo_mops.tolist(),
        "pedestal": pedestal_mops,
        "pedestal_range": PEDESTAL_RANGE["mops"]
    }

    print('T2 Threshold trigger')
    print(f'Number of Th2: {nth2} ({100*nth2/nresiduals:.1f}%)')
    pedestal_th2 = get_pedestal(bin_time, histo_th2, PEDESTAL_RANGE["th2"])
    threshold_th2, tlow_th2, thigh_th2, pur_th2, effi_th2 = get_window(bin_time, histo_th2, pedestal_th2)

    th2_output = {
        "threshold": threshold_th2,
        "tlow": tlow_th2,
        "thigh": thigh_th2,
        "purity": pur_th2,
        "efficiency": effi_th2,
        "histo": histo_th2.tolist(),
        "pedestal": pedestal_th2,
        "pedestal_range": PEDESTAL_RANGE["th2"]
    }
    
    print('T1 Threshold trigger')
    print(f'Number of Th1: {nth1} ({100*nth1/nresiduals:.1f}%)')
    pedestal_th1 = get_pedestal(bin_time, histo_th1, PEDESTAL_RANGE["th1"])
    threshold_th1, tlow_th1, thigh_th1, pur_th1, effi_th1 = get_window(bin_time, histo_th1, pedestal_th1)

    th1_output = {
        "threshold": threshold_th1,
        "tlow": tlow_th1,
        "thigh": thigh_th1,
        "purity": pur_th1,
        "efficiency": effi_th1,
        "histo": histo_th1.tolist(),
        "pedestal": pedestal_th1,
        "pedestal_range": PEDESTAL_RANGE["th1"]
    }
    
    # Global classification performance

    purity = (ntot * pur_tot + ntotd * pur_totd + nmops * pur_mops
              + nth2 * pur_th2 + nth1 * pur_th1) / nresiduals

    efficiency = (ntot * effi_tot + ntotd * effi_totd + nmops * effi_mops
                  + nth2 * effi_th2 + nth1 * effi_th1) / nresiduals

    f_score = 2 * efficiency * purity / (efficiency + purity)

    print('Classification performance')
    print(f'Number of residuals between {TIME_RANGE[0]} and {TIME_RANGE[1]} ns: {nresiduals}')
    print(f'Purity: {100*purity:.2f}%')
    print(f'Efficiency: {100*efficiency:.2f}%')
    print(f'F-score: {100*f_score:.2f}%')

    json_output = {
        "bin_time": bin_time.tolist(),
        "purity": purity,
        "efficiency": efficiency,
        "f_score": f_score,
        "tot": tot_output,
        "totd": totd_output,
        "mops": mops_output,
        "th2": th2_output,
        "th1": th1_output
    }

    with open(output_filename, "w") as output_file:
        json.dump(json_output, output_file)

    return


def get_window(bin_time, residual_histo, pedestal):

    threshold = 1 / (1-BIN_PURITY) * pedestal   # minimum number of counts to select a bin

    mini, maxi = find_limits(residual_histo, threshold)

    bin_width = bin_time[1] - bin_time[0]
    tlow = bin_time[mini] - 0.5 * bin_width
    thigh = bin_time[maxi] + 0.5 * bin_width

    print(f'Selection window: ({tlow:.0f}, {thigh:.0f}) ns')

    selection_window = (tlow, thigh)

    purity = class_perf.get_purity(bin_time, residual_histo, pedestal, selection_window)
    print(f'Purity: {100*purity:.2f}%')

    efficiency = class_perf.get_efficiency(bin_time, residual_histo, pedestal, selection_window)
    print(f'Efficiency: {100*efficiency:.2f}%')

    f_score = class_perf.get_fscore(efficiency, purity)
    print(f'F-score: {100*f_score:.2f}%')

    return threshold, tlow, thigh, purity, efficiency


# Find the bins for the selection window
# Return: index of the minimum and maximum bins of the selection window
def find_limits(residual_histo, threshold):

    max_bin = np.argmax(residual_histo)
    i = max_bin
    while residual_histo[i] > threshold:
        i -= 1
    mini = i+1

    j = max_bin
    while residual_histo[j] > threshold:
        j += 1
    maxi = j-1

    return mini, maxi


def get_pedestal(binx_time, histo, pedestal_range):

    mask = np.all((pedestal_range[0] < binx_time, binx_time < pedestal_range[1]), axis=0)
    noise_histo = histo[mask]
    pedestal = np.mean(noise_histo)
    nbins = len(noise_histo)
    pedestal_error = np.std(noise_histo) / math.sqrt(nbins)
    print(f'Pedestal = {pedestal:.1f} ± {pedestal_error:.1f}')

    return pedestal


# Run starts here
if __name__ == "__main__":
    main()
