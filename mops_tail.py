import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize

PLOT_TYPE = ".pdf"


def model(x, theta, theta_fix):
    return np.exp(theta[0]-(x-theta_fix[0])/theta[1]) + theta_fix[1]


def bin_cost(mu, k):
    if k == 0:
        return 2*mu
    else:
        return 2*(mu-k)-2*k*np.log(mu/k)


def cost_function(theta, xdata, ydata, theta_fix):
    cost = np.zeros_like(theta[0])
    for (x1, y1) in zip(xdata, ydata):
        mu1 = model(x1, theta, theta_fix)
        cost += bin_cost(mu1, y1)
    return cost


def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [json input file]")
        exit(1)

    # Read  data
    input_filename = sys.argv[1]

    with open(input_filename, "r") as input_file:
        data = json.load(input_file)

    bin_time = np.array(data["bin_time"])
    mops_data = data["mops"]
    histo_mops = np.array(mops_data["histo"])

    # Fit data
    mask = np.all((1600 < bin_time, bin_time < 2500), axis=0)
    x0 = 2500
    pedestal = mops_data["pedestal"]
    theta_fix = (x0, pedestal)
    J = lambda theta: cost_function(theta, bin_time[mask], histo_mops[mask], theta_fix)
    res = minimize(J, x0=(2, 1000), method="Nelder-Mead")
    print(res)

    # Plot
    plt.figure()
    plt.yscale("log")

    plt.plot(bin_time[mask], histo_mops[mask], drawstyle='steps', lw=0.5, label='Data')

    yfit = model(bin_time[mask], res.x, theta_fix)
    plt.plot(bin_time[mask], yfit, lw=0.5, ls='--', label='Fit')

    plt.xlabel('Residual time (ns)')
    plt.ylabel('Counts')
    plt.legend()

    filename = "mops_tail" + PLOT_TYPE
    print("Selection window plotted in " + filename)
    plt.savefig(filename)

# Run starts here
if __name__ == "__main__":
    main()
