import pandas as pd
from dask import dataframe as dd
import numpy
from pathlib import Path
import os
import sys


class DistributeActions:

    def __init__(self, facility_file : Path, events_df : pd.DataFrame):
        self.events_df = events_df


    def distribute(self, a, b):
        pass

    def assign_facility(self):
        pass



if __name__ == "__main__":
    pass