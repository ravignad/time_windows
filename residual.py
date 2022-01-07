#!/usr/bin/env python3

# Calculate the time windows to select the SD detectors that participated in the reconstruction of events
# A difference acceptance window is calculated for the trigger types: ToT, ToTd, MoPS, Th2, and Th1

import sys
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# All times in nanoseconds

# Global constants
BIN_PURITY = 0.95
PLOT_TYPE = ".pdf"
TIME_RANGE = (-5000, 12500)    # Range of the residual histograms
NOISE_RANGE = (7500, 12500)    # Range to fit the noise pedestal
NBINS = 700  # Number of bins of the residual histograms


def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [residual file]")
        exit(1)

    # Read  data
    residual_file = sys.argv[1]
    df = pandas.read_csv(residual_file, names=('event', 'station', 'residual', 'trigger_code'))

    # Select 10% of the data to speed up testin
    # df = df.sample(frac=0.1)

    # Select data from Jan 1, 2014 to Aug 31, 2018 as per sd750 paper
    df = df[(140000000000 < df['event']) & (df['event'] < 182440000000)]
    print(f'Number of selected residuals: {len(df.index)}')

    # Add trigger class
    df['trigger_class'] = df['trigger_code'].apply(get_trigger_class)

    histo_tot, _ = np.histogram(df.loc[df['trigger_class'] == 'ToT', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_totd, _ = np.histogram(df.loc[df['trigger_class'] == 'ToTd', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_mops, _ = np.histogram(df.loc[df['trigger_class'] == 'MoPS', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_th2, bin_edges = np.histogram(df.loc[df['trigger_class'] == 'Th2', 'residual'], bins=NBINS, range=TIME_RANGE)
    histo_th1, bin_edges = np.histogram(df.loc[df['trigger_class'] == 'Th1', 'residual'], bins=NBINS, range=TIME_RANGE)

    # Count residuals within the time_range
    ntot = histo_tot.sum()
    ntotd = histo_totd.sum()
    nmops = histo_mops.sum()
    nth2 = histo_th2.sum()
    nth1 = histo_th1.sum()
    nresiduals = ntot + ntotd + nmops + nth2 + nth1

    # Bin centers
    xbin = (bin_edges[:-1]+bin_edges[1:])/2

    # Plot residual histograms
    plot_residual(xbin, histo_tot, histo_totd, histo_mops, histo_th2, histo_th1)

    print('ToT trigger')
    print(f'Number of ToT: {ntot} ({100*ntot/nresiduals:.1f}%)')
    tlow_tot, thigh_tot, pur_tot, effi_tot = get_window(xbin, histo_tot, 'ToT')

    print('ToTd trigger')
    print(f'Number of ToTd: {ntotd} ({100*ntotd/nresiduals:.1f}%)')
    tlow_totd, thigh_totd, pur_totd, effi_totd = get_window(xbin, histo_totd, 'ToTd')

    print('MoPS trigger')
    print(f'Number of MoPS: {nmops} ({100*nmops/nresiduals:.1f}%)')
    tlow_mops, thigh_mops, pur_mops, effi_mops = get_window(xbin, histo_mops, 'MoPS')

    print('T2 Threshold trigger')
    print(f'Number of Th2: {nth2} ({100*nth2/nresiduals:.1f}%)')
    tlow_th2, thigh_th2, pur_th2, effi_th2 = get_window(xbin, histo_th2, 'Th2')

    print('T1 Threshold trigger')
    print(f'Number of Th1: {nth1} ({100*nth1/nresiduals:.1f}%)')
    tlow_th1, thigh_th1, pur_th1, effi_th1 = get_window(xbin, histo_th1, 'Th1')

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

    return


def get_window(binxs, residuals_histo, trigger_label):

    # Calculate the pedestal in ns
    pedestal, pedestal_error = get_pedestal(binxs, residuals_histo, NOISE_RANGE)
    plot_pedestal(binxs, residuals_histo, pedestal, trigger_label)

    threshold = 1 / (1-BIN_PURITY) * pedestal   # minimum number of counts to select a bin

    mini = np.min(np.nonzero(residuals_histo > threshold))
    maxi = np.min(np.nonzero(residuals_histo[mini:] < threshold)) + mini - 1

    bin_width = binxs[1] - binxs[0]
    tlow = binxs[mini] - 0.5 * bin_width
    thigh = binxs[maxi] + 0.5 * bin_width

    print(f'Acceptance window: ({tlow:.0f}, {thigh:.0f}) ns')

    signal_histo = residuals_histo - pedestal
    selection_window = np.arange(mini, maxi+1)  # include maximum bin

    purity = get_purity(signal_histo[selection_window], pedestal)
    print(f'Purity: {100*purity:.2f}%')

    efficiency = signal_histo[selection_window].sum() / signal_histo.sum()
    print(f'Efficiency: {100*efficiency:.2f}%')

    f_score = 2 * efficiency * purity / (efficiency + purity)
    print(f'F-score: {100*f_score:.2f}%')

    return tlow, thigh, purity, efficiency


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


def get_pedestal(binxs, histo, time_range):

    mask = np.all((time_range[0] < binxs, binxs < time_range[1]), axis=0)
    noise_histo = histo[mask]
    pedestal = np.mean(noise_histo)
    nbins = len(noise_histo)
    pedestal_error = np.std(noise_histo) / math.sqrt(nbins)
    print(f'Pedestal = {pedestal:.1f} ± {pedestal_error:.1f}')

    return pedestal, pedestal_error


# Get the purity of a signal histogram given a noise pedestal
def get_purity(signal_histo, pedestal):

    signal = signal_histo.sum()
    noise = pedestal * len(signal_histo)
    purity = signal / (signal+noise)

    return purity


def plot_residual(xbin, histo_tot, histo_totd, histo_mops, histo_th2, histo_th1):

    plt.figure()
    plt.yscale("log")

    plt.plot(xbin, histo_tot, drawstyle='steps', lw=0.5, label='ToT')
    plt.plot(xbin, histo_totd, drawstyle='steps', lw=0.5, label='ToTd')
    plt.plot(xbin, histo_mops, drawstyle='steps', lw=0.5, label='MoPS')
    plt.plot(xbin, histo_th2, drawstyle='steps', lw=0.5, label='Th2')
    plt.plot(xbin, histo_th1, drawstyle='steps', lw=0.5, label='Th1')

    plt.xlabel('Residual (ns)')
    plt.ylabel('Counts')

    plt.xlim((5000, 15000))
    # plt.ylim(bottom=10)

    plt.legend()
    filename = "residual_trigger" + PLOT_TYPE
    print("Residuals plot save in " + filename)
    plt.savefig(filename)


def plot_pedestal(binxs, bin_counts, pedestal, trigger_label):

    plt.figure()
    plt.yscale("log")

    plt.plot(binxs, bin_counts, drawstyle='steps', lw=0.5, label='Data')

    plt.xlabel('Residual (ns)')
    plt.ylabel('Counts')

    xmin, xmax = binxs[0], binxs[-1]
    pedestal_line = Line2D((xmin, xmax), (pedestal, pedestal), lw=0.5, color='tab:orange', label='Pedestal')

    ax = plt.gca()
    ax.add_line(pedestal_line)

    plt.xlim((-5000, 12500))

    plt.legend()

    filename = "residual_" + trigger_label + PLOT_TYPE
    print("Residuals plotted in " + filename)
    plt.savefig(filename)


# Run starts here
if __name__ == "__main__":
    main()
