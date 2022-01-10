#!/usr/bin/env python3

# Calculate the time windows to select the SD detectors that participated in the reconstruction of events
# A difference acceptance window is calculated for the trigger types: ToT, ToTd, MoPS, Th2, and Th1

import sys
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
import json

# All times in nanoseconds

# Global constants
BIN_PURITY = 0.95
PLOT_TYPE = ".pdf"
PEDESTAL_RANGE = (5000, 8000)    # Range to fit the noise pedestal
TIME_RANGE = (-4000, 8000)    # Range of the residual histograms
NBINS = 480  # Number of bins of the residual histograms


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
    bin_time = (bin_edges[:-1]+bin_edges[1:])/2

    print('ToT trigger')
    print(f'Number of ToT: {ntot} ({100*ntot/nresiduals:.1f}%)')
    pedestal_tot = get_pedestal(bin_time, histo_tot)
    tlow_tot, thigh_tot, pur_tot, effi_tot = get_window(bin_time, histo_tot, pedestal_tot, 'ToT')

    tot_output ={
        "tlow": tlow_tot,
        "thigh": thigh_tot,
        "purity": pur_tot,
        "efficiency": effi_tot,
        "bin_time": bin_time.tolist(),
        "histo": histo_tot.tolist(),
        "pedestal": pedestal_tot,
    }

    print('ToTd trigger')
    print(f'Number of ToTd: {ntotd} ({100*ntotd/nresiduals:.1f}%)')
    pedestal_totd = get_pedestal(bin_time, histo_totd)
    tlow_totd, thigh_totd, pur_totd, effi_totd = get_window(bin_time, histo_totd, pedestal_totd, 'ToTd')

    totd_output ={
        "tlow": tlow_totd,
        "thigh": thigh_totd,
        "purity": pur_totd,
        "efficiency": effi_totd,
        "bin_time": bin_time.tolist(),
        "histo": histo_totd.tolist(),
        "pedestal": pedestal_totd,
    }

    print('MoPS trigger')
    print(f'Number of MoPS: {nmops} ({100*nmops/nresiduals:.1f}%)')
    pedestal_mops = get_pedestal(bin_time, histo_mops)
    tlow_mops, thigh_mops, pur_mops, effi_mops = get_window(bin_time, histo_mops, pedestal_mops, 'MoPS')

    mops_output ={
        "tlow": tlow_mops,
        "thigh": thigh_mops,
        "purity": pur_mops,
        "efficiency": effi_mops,
        "bin_time": bin_time.tolist(),
        "histo": histo_mops.tolist(),
        "pedestal": pedestal_mops,
    }

    print('T2 Threshold trigger')
    print(f'Number of Th2: {nth2} ({100*nth2/nresiduals:.1f}%)')
    pedestal_th2 = get_pedestal(bin_time, histo_th2)
    tlow_th2, thigh_th2, pur_th2, effi_th2 = get_window(bin_time, histo_th2, pedestal_th2, 'Th2')

    th2_output ={
        "tlow": tlow_th2,
        "thigh": thigh_th2,
        "purity": pur_th2,
        "efficiency": effi_th2,
        "bin_time": bin_time.tolist(),
        "histo": histo_th2.tolist(),
        "pedestal": pedestal_th2,
    }
    
    print('T1 Threshold trigger')
    print(f'Number of Th1: {nth1} ({100*nth1/nresiduals:.1f}%)')
    pedestal_th1 = get_pedestal(bin_time, histo_th1)
    tlow_th1, thigh_th1, pur_th1, effi_th1 = get_window(bin_time, histo_th1, pedestal_th1, 'Th1')

    th1_output ={
        "tlow": tlow_th1,
        "thigh": thigh_th1,
        "purity": pur_th1,
        "efficiency": effi_th1,
        "bin_time": bin_time.tolist(),
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
        "purity": purity,
        "efficiency": efficiency,
        "f_score": f_score,
        "tot": tot_output,
        "totd": totd_output,
        "mops": mops_output,
        "th2": th2_output,
        "th1": th1_output
    }

    with open("residual.json", "w") as output_file:
        json.dump(json_output, output_file)

    # Plot residual histograms
    plot_residual(bin_time, (histo_tot, histo_totd, histo_mops, histo_th2, histo_th1),
                  (pedestal_tot, pedestal_totd, pedestal_mops, pedestal_th2, pedestal_th1))



    return


def get_window(bin_time, residuals_histo, pedestal, trigger_label):

    threshold = 1 / (1-BIN_PURITY) * pedestal   # minimum number of counts to select a bin

    mini = np.min(np.nonzero(residuals_histo > threshold))
    maxi = np.min(np.nonzero(residuals_histo[mini:] < threshold)) + mini - 1

    bin_width = bin_time[1] - bin_time[0]
    tlow = bin_time[mini] - 0.5 * bin_width
    thigh = bin_time[maxi] + 0.5 * bin_width

    print(f'Acceptance window: ({tlow:.0f}, {thigh:.0f}) ns')

    signal_histo = residuals_histo - pedestal
    selection_window = np.arange(mini, maxi+1)  # include maximum bin

    purity = get_purity(signal_histo[selection_window], pedestal)
    print(f'Purity: {100*purity:.2f}%')

    efficiency = signal_histo[selection_window].sum() / signal_histo.sum()
    print(f'Efficiency: {100*efficiency:.2f}%')

    f_score = 2 * efficiency * purity / (efficiency + purity)
    print(f'F-score: {100*f_score:.2f}%')

    # Calculate the pedestal in ns
    plot_window(bin_time, residuals_histo, pedestal, threshold, mini, maxi, trigger_label)

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


def plot_residual(bin_time, histos, pedestals):

    plt.figure()
    plt.yscale("log")

    histo_tot = histos[0]
    histo_totd = histos[1]
    histo_mops = histos[2]
    histo_th2 = histos[3]
    histo_th1 = histos[4]

    plt.plot(bin_time, histo_tot, drawstyle='steps', lw=0.5, label='ToT')
    plt.plot(bin_time, histo_totd, drawstyle='steps', lw=0.5, label='ToTd')
    plt.plot(bin_time, histo_mops, drawstyle='steps', lw=0.5, label='MoPS')
    plt.plot(bin_time, histo_th2, drawstyle='steps', lw=0.5, label='Th2')
    plt.plot(bin_time, histo_th1, drawstyle='steps', lw=0.5, label='Th1')

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    plt.legend()
    filename = "residual" + PLOT_TYPE
    print("Residuals plotted in " + filename)
    plt.savefig(filename)

    # Plot pedestals
    plt.figure()

    mask = np.all((PEDESTAL_RANGE[0] < bin_time, bin_time < PEDESTAL_RANGE[1]), axis=0)

    plt.plot(bin_time[mask], histo_tot[mask], drawstyle='steps', lw=0.5, label='ToT')
    plt.plot(bin_time[mask], histo_totd[mask], drawstyle='steps', lw=0.5, label='ToTd')
    plt.plot(bin_time[mask], histo_mops[mask], drawstyle='steps', lw=0.5, label='MoPS')
    plt.plot(bin_time[mask], histo_th2[mask], drawstyle='steps', lw=0.5, label='Th2')
    plt.plot(bin_time[mask], histo_th1[mask], drawstyle='steps', lw=0.5, label='Th1')

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    # Plot fitted pedestals
    x = (bin_time[mask][0], bin_time[mask][-1])
    plt.gca().set_prop_cycle(None)
    y = np.array([pedestals, pedestals])
    plt.plot(x, y, lw=0.5)

    plt.legend()
    filename = "pedestal" + PLOT_TYPE
    print("Pedestals plotted in " + filename)
    plt.savefig(filename)


def plot_window(bin_time, bin_counts, pedestal, threshold, mini, maxi, trigger_label):

    plt.figure()
    ax = plt.gca()
    plt.yscale("log")

    plt.text(0.9, 0.9, trigger_label, fontsize='large', ha='right', transform=ax.transAxes)

    plt.plot(bin_time, bin_counts, drawstyle='steps', lw=0.5, label='Data')
    plt.fill_between(bin_time[mini:maxi], bin_counts[mini:maxi], step="pre", alpha=0.4)

    x = (bin_time[0], bin_time[-1])
    p = plt.plot(x, (pedestal, pedestal), lw=0.5, ls='--', label='Pedestal')
    plt.text(bin_time[mini], 0.9*pedestal, 'Pedestal', fontsize='small', va='top', color=p[0].get_color())

    p = plt.plot(x, (threshold, threshold), lw=0.5, ls='--', label='Threshold')
    plt.text(0.6, 1.1*threshold, 'Threshold', fontsize='small', color=p[0].get_color(),
             transform=ax.get_yaxis_transform())

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')

    filename = "window_" + trigger_label + PLOT_TYPE
    print("Acceptance window plotted in " + filename)
    plt.savefig(filename)


# Run starts here
if __name__ == "__main__":
    main()
