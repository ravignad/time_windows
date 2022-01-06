#!/usr/bin/env python3

import sys
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Global constants
BIN_PURITY = 0.9
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

    # Convert residuals to microseconds
    df['residual'] = df['residual']/1000

    # Binning data
    time_range = (-5, 12.5)
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
    range_th = get_limits(xbin, histo_th, 'Th')

    print('ToT trigger')
    range_tot = get_limits(xbin, histo_tot, 'ToT')

    print('ToTd trigger')
    range_totd = get_limits(xbin, histo_totd, 'ToTd')

    print('MoPS trigger')
    range_mops = get_limits(xbin, histo_mops, 'MoPS')

    return


# Calculate purity vs. efficiency
def purity_efficiency(histo, pedestal, trigger_label):

    sorted_indexes = sort_adjacent(histo)
    sorted_histo = histo[sorted_indexes]
    cum_histo = np.cumsum(sorted_histo)

    pedestal_histo = np.full_like(cum_histo, fill_value=pedestal, dtype=float)
    cum_pedestal = np.cumsum(pedestal_histo)

    cum_signal = cum_histo - cum_pedestal
    total_signal = cum_signal[-1]

    efficiency = cum_signal / total_signal
    purity = cum_signal / cum_histo

    # Plot
    plt.figure()

    plt.plot(efficiency, purity, ls='None', marker='.', ms=1)

#    plt.xlim(left=0)
    plt.xlim((0.9, 1.0))
    plt.ylim((0.98, 1.0))

    plt.xlabel('Efficiency')
    plt.ylabel('Purity')

    filename = "purity_" + trigger_label + PLOT_TYPE
    print("Purity plotted in " + filename)
    plt.savefig(filename)

    # Plot the gradient ΔPurity/ΔEfficiency vs. efficiency
    gradient = np.gradient(purity, efficiency)
    plt.figure()
    plt.plot(efficiency, gradient, ls='None', marker='.', ms=1)
    plt.xlabel('Efficiency')
    plt.ylabel('ΔPurity/ΔEfficiency')
    plt.xlim((0.9, 1.0))
    plt.ylim((-1, 0.1))
    filename = "gradient_" + trigger_label + PLOT_TYPE
    print("Gradient plotted in " + filename)
    plt.savefig(filename)

    gradient_cut = -0.1
    window_index = np.argmax(gradient < gradient_cut)
    selection_purity = purity[window_index]
    selection_efficiency = efficiency[window_index]
    print(f'Purity = {100 * selection_purity:.2f}%')
    print(f'Efficiency = {100*selection_efficiency:.2f}%')



def get_limits(binxs, residuals_histo, trigger_label):

    # Calculate the pedestal in μs
    noise_range = (7.5, 12.5)
    pedestal, pedestal_error = get_pedestal(binxs, residuals_histo, noise_range)
    plot_pedestal(binxs, residuals_histo, pedestal, trigger_label)

    signal_histo = residuals_histo - pedestal
    purity_histo = signal_histo / residuals_histo

    mini = np.min(np.nonzero(purity_histo > BIN_PURITY))
    maxi = np.max(np.nonzero(purity_histo > BIN_PURITY))

    bin_width = binxs[1] - binxs[0]
    xmin = binxs[mini] - 0.5 * bin_width
    xmax = binxs[maxi] + 0.5 * bin_width

    print(f'Acceptance window: ({xmin*1000:.0f}, {xmax*1000:.0f}) ns')

    selection_window = np.arange(mini, maxi+1)  # include maximum bin

    purity = get_purity(signal_histo[selection_window], pedestal)
    print(f'Purity: {100*purity:.2f}%')

    efficiency = signal_histo[selection_window].sum() / signal_histo.sum()
    print(f'Efficiency: {100*efficiency:.2f}%')

    plt.figure()
    plt.plot(binxs[selection_window], purity_histo[selection_window], drawstyle='steps', lw=0.5)
    plt.xlabel('Residual (μs)')
    plt.ylabel('Purity')

    filename = "purity_" + trigger_label + PLOT_TYPE
    print("Purity plotted in " + filename)
    plt.savefig(filename)



    # purity_efficiency(residuals, pedestal, trigger_label)

    return


def get_histo(df, bin_edges, trigger_group):
    residual = df.loc[df['t1group'] == trigger_group, 'residual']
    histo, temp = np.histogram(residual, bins=bin_edges)
    return histo

# Map t1code to trigger category (ToT, TH, ToTd, and MoPs)
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

def get_purity(signal_histo, pedestal):

    signal = signal_histo.sum()
    noise = pedestal * len(signal_histo)
    purity = signal / (signal+noise)

    return purity


def get_signal(bin_counts, binxs, pedestal, signal_range):
    # Substract pedestal to obtain the signal
    signal = bin_counts - pedestal
    # Select signal region
    mask = (signal_range[0] <= binxs) & (binxs < signal_range[1])
    signal2 = signal[mask]
    binxs2 = binxs[mask]
    return binxs2, signal2


def get_pedestal(binxs, histo, range):
    mask = np.all((range[0] < binxs, binxs < range[1]), axis=0)
    noise_histo = histo[mask]
    pedestal = np.mean(noise_histo)
    nbins = len(noise_histo)
    pedestal_error = np.std(noise_histo) / math.sqrt(nbins)
    print(f'Pedestal = {pedestal:.1f} ± {pedestal_error:.1f}')
    return pedestal, pedestal_error


# Sort an array in descending order by adjacent elements
def sort_adjacent(array):

    sorted_indexes = np.zeros_like(array)
    nelements = len(array)

    # Initialization
    mini = maxi = np.argmax(array)
    sorted_indexes[0] = mini

    for i in range(1, nelements):
        if mini == 0:
            maxi += 1
            sorted_indexes[i] = maxi
        elif maxi == nelements-1:
            mini -= 1
            sorted_indexes[i] = mini
        elif array[maxi+1] > array[mini-1]:
            maxi += 1
            sorted_indexes[i] = maxi
        else:
            mini -= 1
            sorted_indexes[i] = mini

    return sorted_indexes


# Find limits
def find_limits(binxs, histo):

    # Initialize scan
    bin_min = np.argmax(histo)
    bin_max = bin_min
    cum_counts = signal_norm[bin_min]

    while proba < target_proba:
        if signal_norm[bin_max+1] > signal_norm[bin_min-1]:
            cum_counts += histo[bin_max+1]
            bin_max = bin_max + 1
        else:
            cum_counts += signal_norm[bin_min-1]
            bin_min = bin_min - 1

    bin_size = binxs[1] - binxs[0]
    residual_min = binxs[bin_min] - bin_size / 2
    residual_max = binxs[bin_max] + bin_size / 2

    return (bin_min, bin_max), (residual_min, residual_max)


# Probability between two bins


def plot_pedestal(binxs, bin_counts, pedestal, trigger_label):

    plt.figure()
    plt.yscale("log")

    plt.plot(binxs, bin_counts, drawstyle='steps', lw=0.5, label='Data')

    plt.xlabel('Residual (μs)')
    plt.ylabel('Counts')

    xmin, xmax = binxs[0], binxs[-1]
    pedestal_line = Line2D( (xmin, xmax), (pedestal, pedestal), lw=0.5, color='tab:orange', label='Pedestal')

    ax = plt.gca()
    ax.add_line(pedestal_line)

    plt.xlim((-5, 12.5))

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

    plt.xlabel('Residual (μs)')
    plt.ylabel('Counts')

    plt.xlim((5, 15))
    # plt.ylim(bottom=10)

    plt.legend()
    filename = "residual_trigger" + PLOT_TYPE
    print("Residuals plot save in " + filename)
    plt.savefig(filename)


# Run starts here
if __name__ == "__main__":
    main()