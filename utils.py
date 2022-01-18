from matplotlib import pyplot as plt


def savefig(filename, message):

    print(message + " " + filename + ".eps")
    plt.savefig(filename + ".eps")
    print("Selection window plotted in " + filename + ".jpg")
    plt.savefig(filename + ".jpg")
    print("Selection window plotted in " + filename + ".pdf")
    plt.savefig(filename + ".pdf")