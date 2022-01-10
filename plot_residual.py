import sys
import numpy as np
import matplotlib.pyplot as plt
import json

PLOT_TYPE = ".pdf"

if len(sys.argv) != 2:
    print("Usage " + sys.argv[0] + " [json input file")
    exit(1)

# Read  data
input_filename = sys.argv[1]

with open(input_filename, "r") as input_file:
    data = json.load(input_file)

tot_data = data["tot"]
totd_data = data["totd"]
mops_data = data["mops"]
th2_data = data["th2"]
th1_data = data["th1"]

bin_time = np.array(data["bin_time"])
histo_tot = np.array(tot_data["histo"])
histo_totd = np.array(totd_data["histo"])
histo_mops = np.array(mops_data["histo"])
histo_th2 = np.array(th2_data["histo"])
histo_th1 = np.array(th1_data["histo"])

plt.figure()
plt.yscale("log")

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

pedestal_min = data["pedestal_range"][0]
pedestal_max = data["pedestal_range"][1]
mask = np.all((pedestal_min < bin_time, bin_time < pedestal_max), axis=0)

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
pedestals = (tot_data["pedestal"],
             totd_data["pedestal"],
             mops_data["pedestal"],
             th2_data["pedestal"],
             th1_data["pedestal"]
             )
y = np.array([pedestals, pedestals])
plt.plot(x, y, lw=0.5)

plt.legend()
filename = "pedestal" + PLOT_TYPE
print("Pedestals plotted in " + filename)
plt.savefig(filename)
