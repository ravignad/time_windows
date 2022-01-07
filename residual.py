#!/usr/bin/env python3

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

def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [residual file]")
        exit(1)

    # Read  data
    residual_file = sys.argv[1]
    df = pandas.read_csv(residual_file, names=('event', 'station', 'residual', 'trigger_code') )

    # Select 10% of the data to speed up testin
    # df = df.sample(frac=0.1)

    # Select data from Jan 1, 2014 to Aug 31, 2018 as per sd750 paper
    df = df[(140000000000 < df['event']) & (df['event'] < 182440000000)]
    print(f'Number of selected residuals: {len(df.index)}')

    # Add trigger class
    df['trigger_class'] = df['trigger_code'].apply(get_trigger_class)

    # Binning data
    time_range = (-5000, 12500)
    nbins = 350

    histo_th, bin_edges = np.histogram(df.loc[df['trigger_class'] == 'Th', 'residual'], bins=nbins, range=time_range)
    histo_tot, _ = np.histogram(df.loc[df['trigger_class'] == 'ToT', 'residual'], bins=nbins, range=time_range)
    histo_totd, _ = np.histogram(df.loc[df['trigger_class'] == 'ToTd', 'residual'], bins=nbins, range=time_range)
    histo_mops, _ = np.histogram(df.loc[df['trigger_class'] == 'MoPS', 'residual'], bins=nbins, range=time_range)

    # Bin centers
    xbin = (bin_edges[:-1]+bin_edges[1:])/2

    # Plot residual histograms
    plot_residual(xbin, histo_th, histo_tot, histo_totd, histo_mops)

    print('Threshold trigger')
    window_th = get_window(xbin, histo_th, 'Th')

    print('ToT trigger')
    window_tot = get_window(xbin, histo_tot, 'ToT')

    print('ToTd trigger')
    window_totd = get_window(xbin, histo_totd, 'ToTd')

    print('MoPS trigger')
    window_mops = get_window(xbin, histo_mops, 'MoPS')

    return


def get_window(binxs, residuals_histo, trigger_label):

    # Calculate the pedestal in ns
    noise_range = (7500, 12500)
    pedestal, pedestal_error = get_pedestal(binxs, residuals_histo, noise_range)
    plot_pedestal(binxs, residuals_histo, pedestal, trigger_label)

    signal_histo = residuals_histo - pedestal
    purity_histo = signal_histo / residuals_histo

    mini = np.min(np.nonzero(purity_histo > BIN_PURITY))
    maxi = np.max(np.nonzero(purity_histo > BIN_PURITY))

    bin_width = binxs[1] - binxs[0]
    tlow = binxs[mini] - 0.5 * bin_width
    thigh = binxs[maxi] + 0.5 * bin_width

    print(f'Acceptance window: ({tlow:.0f}, {thigh:.0f}) ns')

    selection_window = np.arange(mini, maxi+1)  # include maximum bin

    purity = get_purity(signal_histo[selection_window], pedestal)
    print(f'Purity: {100*purity:.2f}%')

    efficiency = signal_histo[selection_window].sum() / signal_histo.sum()
    print(f'Efficiency: {100*efficiency:.2f}%')

    f_score = 2 * efficiency * purity / (efficiency + purity)
    print(f'F-score: {100*f_score:.2f}%')

    plt.figure()
    plt.plot(binxs[selection_window], purity_histo[selection_window], drawstyle='steps', lw=0.5)
    plt.xlabel('Residual (ns)')
    plt.ylabel('Purity')

    filename = "purity_" + trigger_label + PLOT_TYPE
    print("Purity plotted in " + filename)
    plt.savefig(filename)

    return tlow, thigh, purity, efficiency, f_score


# Map t1code to trigger class (ToT, TH, ToTd, and MoPs)
# Trigger hierarchy ToT -> TH -> ToTd -> MoPs
def get_trigger_class(trigger_code):
    if trigger_code & 4 != 0:     # bit 3
        return 'ToT'
    elif trigger_code & 3 != 0 :  # bit 1 and 2
        return 'Th'
    elif trigger_code & 8 != 0:  # bit 4
        return 'ToTd'
    elif trigger_code & 16 != 0: # bit 5
        return 'MoPS'
    else:
        return 'None'

# Get the purity of bin sample given a noise pedestal
def get_purity(signal_histo, pedestal):

    signal = signal_histo.sum()
    noise = pedestal * len(signal_histo)
    purity = signal / (signal+noise)

    return purity


def get_pedestal(binxs, histo, range):

    mask = np.all((range[0] < binxs, binxs < range[1]), axis=0)
    noise_histo = histo[mask]
    pedestal = np.mean(noise_histo)
    nbins = len(noise_histo)
    pedestal_error = np.std(noise_histo) / math.sqrt(nbins)
    print(f'Pedestal = {pedestal:.1f} ± {pedestal_error:.1f}')

    return pedestal, pedestal_error


def plot_pedestal(binxs, bin_counts, pedestal, trigger_label):

    plt.figure()
    plt.yscale("log")

    plt.plot(binxs, bin_counts, drawstyle='steps', lw=0.5, label='Data')

    plt.xlabel('Residual (ns)')
    plt.ylabel('Counts')

    xmin, xmax = binxs[0], binxs[-1]
    pedestal_line = Line2D( (xmin, xmax), (pedestal, pedestal), lw=0.5, color='tab:orange', label='Pedestal')

    ax = plt.gca()
    ax.add_line(pedestal_line)

    plt.xlim((-5000, 12500))

    plt.legend()

    filename = "residual_" + trigger_label + PLOT_TYPE
    print("Residuals plotted in " + filename)
    plt.savefig(filename)


def plot_residual(xbin, histo_th, histo_tot, histo_totd, histo_mops):

    plt.figure()
    plt.yscale("log")

    plt.plot(xbin, histo_th, drawstyle='steps', lw=0.5, label='Th')
    plt.plot(xbin, histo_tot, drawstyle='steps', lw=0.5, label='ToT')
    plt.plot(xbin, histo_totd, drawstyle='steps', lw=0.5, label='ToTd')
    plt.plot(xbin, histo_mops, drawstyle='steps', lw=0.5, label='MoPS')

    plt.xlabel('Residual (ns)')
    plt.ylabel('Counts')

    plt.xlim((5000, 15000))
    # plt.ylim(bottom=10)

    plt.legend()
    filename = "residual_trigger" + PLOT_TYPE
    print("Residuals plot save in " + filename)
    plt.savefig(filename)


# Run starts here
if __name__ == "__main__":
    main()