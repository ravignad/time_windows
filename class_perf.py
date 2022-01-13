# Calculate the precision, recall and F-score of a classification

import numpy as np

# Calculate the selection purity
def get_purity(bin_time, residual_histo, pedestal, window):

    mask = np.all((window[0] <= bin_time, bin_time < window[1]), axis=0)
    residual_selected = residual_histo[mask]

    signal_selected = residual_selected - pedestal
    purity = signal_selected.sum() / residual_selected.sum()

    return purity

# Calculate the selection efficiency
def get_efficiency(bin_time, residual_histo, pedestal, window):

    signal_histo = residual_histo - pedestal

    mask = np.all((window[0] <= bin_time, bin_time < window[1]), axis=0)
    signal_selected = signal_histo[mask]

    efficiency = signal_selected.sum() / signal_histo.sum()

    return efficiency

def get_fscore(efficiency, purity, beta=1):

    fscore = (1+beta**2)*efficiency*purity / (beta**2*efficiency+purity)

    return fscore