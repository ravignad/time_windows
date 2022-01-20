import sys
import math
import numpy as np
import pandas

import utils

EVENT_RANGE = (140000000000, 182440000000)

def main():

    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " [residual csv file]")
        exit(1)

    # Read  data
    residual_file = sys.argv[1]
    df = pandas.read_csv(residual_file, names=('event', 'station', 'residual', 'trigger_code'))

    # Select data from Jan 1, 2014 to Aug 31, 2018 as per sd750 paper
    df = df[(EVENT_RANGE[0] < df['event']) & (df['event'] < EVENT_RANGE[1])]
    print(f'Number of selected residuals: {len(df.index)}')

   # Add trigger class
    df['trigger_class'] = df['trigger_code'].apply(utils.get_trigger_class)

    df_tot = df[ (df['trigger_class'] == 'ToT') & (df['residual'] > 1500) ].head()

    print(df_tot)


# Run starts here
if __name__ == "__main__":
    main()
