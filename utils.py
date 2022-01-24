from matplotlib import pyplot as plt


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


def savefig(filename, message):

    print(message + " " + filename + ".eps")
    plt.savefig(filename + ".eps")
    print("Selection window plotted in " + filename + ".jpg")
    plt.savefig(filename + ".jpg")
    print("Selection window plotted in " + filename + ".pdf")
    plt.savefig(filename + ".pdf")
    print("Selection window plotted in " + filename + ".svg")
    plt.savefig(filename + ".svg")