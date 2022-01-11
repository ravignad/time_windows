# Calculate the time windows to select the SD detectors that participated in the reconstruction of events
# A difference selection window is calculated for the trigger types: ToT, ToTd, MoPS, Th2, and Th1

import sys
import math
import numpy as np
import pandas
import json

# All times in nanoseconds

# Global constants
BIN_PURITY = 0.95
PLOT_TYPE = ".pdf"
PEDESTAL_RANGE = (5000, 8000)    # Range to fit the noise pedestal
TIME_RANGE = (-4000, 8000)    # Range of the residual histograms
NBINS = 480  # Number of bins of the residual histograms


def main():

    if len(sys.argv) != 3:
        print("Usage " + sys.argv[0] + " [residual file] [json output file]")
        exit(1)

    # Read  data
    residual_file = sys.argv[1]
    output_filename = sys.argv[2]
    df = pandas.read_csv(residual_file, names=('event', 'station', 'residual', 'trigger_code'))

    # Select 10% of the data to speed up testin
    # df = df.sample(frac=0.1)

    # Select data from Jan 1, 2014 to Aug 31, 2018 as per sd750 paper
    df = df[(140000000000 < df['event']) & (df['event'] < 182440000000)]
    print(f'Number of selected residuals: {len(df.index)}')

    # Add trigger class
    df['trigger_class'] = df['trigger_code'].apply(get_trigger_class)

    histo_tot, bin_edges = np.histogram(df.loc[df['trigger_class'] == 'ToT', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_totd, _ = np.histogram(df.loc[df['trigger_class'] == 'ToTd', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_mops, _ = np.histogram(df.loc[df['trigger_class'] == 'MoPS', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_th2, _ = np.histogram(df.loc[df['trigger_class'] == 'Th2', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_th1, _ = np.histogram(df.loc[df['trigger_class'] == 'Th1', 'residual'], bins=NBINS, range=TIME_RANGE)

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
    pedestal_tot = get_pedestal(bin_time, histo_tot)
    threshold_tot, tlow_tot, thigh_tot, pur_tot, effi_tot = get_window(bin_time, histo_tot, pedestal_tot)

    tot_output = {
        "threshold": threshold_tot,
        "tlow": tlow_tot,
        "thigh": thigh_tot,
        "purity": pur_tot,
        "efficiency": effi_tot,
        "histo": histo_tot.tolist(),
        "pedestal": pedestal_tot,
    }

    print('ToTd trigger')
    print(f'Number of ToTd: {ntotd} ({100*ntotd/nresiduals:.1f}%)')
    pedestal_totd = get_pedestal(bin_time, histo_totd)
    threshold_totd, tlow_totd, thigh_totd, pur_totd, effi_totd = get_window(bin_time, histo_totd, pedestal_totd)

    totd_output = {
        "threshold": threshold_totd,
        "tlow": tlow_totd,
        "thigh": thigh_totd,
        "purity": pur_totd,
        "efficiency": effi_totd,
        "histo": histo_totd.tolist(),
        "pedestal": pedestal_totd,
    }

    print('MoPS trigger')
    print(f'Number of MoPS: {nmops} ({100*nmops/nresiduals:.1f}%)')
    pedestal_mops = get_pedestal(bin_time, histo_mops)
    threshold_mops, tlow_mops, thigh_mops, pur_mops, effi_mops = get_window(bin_time, histo_mops, pedestal_mops)

    mops_output = {
        "threshold": threshold_mops,
        "tlow": tlow_mops,
        "thigh": thigh_mops,
        "purity": pur_mops,
        "efficiency": effi_mops,
        "histo": histo_mops.tolist(),
        "pedestal": pedestal_mops,
    }

    print('T2 Threshold trigger')
    print(f'Number of Th2: {nth2} ({100*nth2/nresiduals:.1f}%)')
    pedestal_th2 = get_pedestal(bin_time, histo_th2)
    threshold_th2, tlow_th2, thigh_th2, pur_th2, effi_th2 = get_window(bin_time, histo_th2, pedestal_th2)

    th2_output = {
        "threshold": threshold_th2,
        "tlow": tlow_th2,
        "thigh": thigh_th2,
        "purity": pur_th2,
        "efficiency": effi_th2,
        "histo": histo_th2.tolist(),
        "pedestal": pedestal_th2,
    }
    
    print('T1 Threshold trigger')
    print(f'Number of Th1: {nth1} ({100*nth1/nresiduals:.1f}%)')
    pedestal_th1 = get_pedestal(bin_time, histo_th1)
    threshold_th1, tlow_th1, thigh_th1, pur_th1, effi_th1 = get_window(bin_time, histo_th1, pedestal_th1)

    th1_output = {
        "threshold": threshold_th1,
        "tlow": tlow_th1,
        "thigh": thigh_th1,
        "purity": pur_th1,
        "efficiency": effi_th1,
        "histo": histo_th1.tolist(),
        "pedestal": pedestal_th1,
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
        "pedestal_range": PEDESTAL_RANGE,
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


def get_window(bin_time, residuals_histo, pedestal):

    threshold = 1 / (1-BIN_PURITY) * pedestal   # minimum number of counts to select a bin

    mini, maxi = find_limits(residuals_histo, threshold)

    bin_width = bin_time[1] - bin_time[0]
    tlow = bin_time[mini] - 0.5 * bin_width
    thigh = bin_time[maxi] + 0.5 * bin_width

    print(f'Selection window: ({tlow:.0f}, {thigh:.0f}) ns')

    signal_histo = residuals_histo - pedestal
    selection_window = np.arange(mini, maxi+1)  # include maximum bin

    purity = get_purity(signal_histo[selection_window], pedestal)
    print(f'Purity: {100*purity:.2f}%')

    efficiency = signal_histo[selection_window].sum() / signal_histo.sum()
    print(f'Efficiency: {100*efficiency:.2f}%')

    f_score = 2 * efficiency * purity / (efficiency + purity)
    print(f'F-score: {100*f_score:.2f}%')

    return threshold, tlow, thigh, purity, efficiency


# Find the bins for the selection window
# Return: index of the minimum and maximum bins of the selection window
def find_limits(residuals_histo, threshold):

    max_bin = np.argmax(residuals_histo)
    i = max_bin
    while residuals_histo[i] > threshold:
        i -= 1
    mini = i+1

    j = max_bin
    while residuals_histo[j] > threshold:
        j += 1
    maxi = j-1

    return mini, maxi


# Map trigger code to the trigger type
# Trigger hierarchy ToT -> ToTd -> MoPs -> Th2 -> Th1
def get_trigger_class(trigger_code):
    if trigger_code & 4 != 0:     # bit 3
        return 'ToT'
    elif trigger_code & 8 != 0:  # bit 4
        return 'ToTd'
    elif trigger_code & 16 != 0:  # bit 5
        return 'MoPS'
    elif trigger_code & 2 != 0:  # bit 2
        return 'Th2'
    elif trigger_code & 1 != 0:  # bit 1
        return 'Th1'
    else:
        return 'None'


def get_pedestal(binx_time, histo):

    mask = np.all((PEDESTAL_RANGE[0] < binx_time, binx_time < PEDESTAL_RANGE[1]), axis=0)
    noise_histo = histo[mask]
    pedestal = np.mean(noise_histo)
    nbins = len(noise_histo)
    pedestal_error = np.std(noise_histo) / math.sqrt(nbins)
    print(f'Pedestal = {pedestal:.1f} ± {pedestal_error:.1f}')

    return pedestal


# Get the purity of a signal histogram given a noise pedestal
def get_purity(signal_histo, pedestal):

    signal = signal_histo.sum()
    noise = pedestal * len(signal_histo)
    purity = signal / (signal+noise)

    return purity


# Run starts here
if __name__ == "__main__":
    main()
