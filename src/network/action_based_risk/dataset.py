import pandas as pd
import numpy as np
from configparser import ConfigParser
import networkx as nx
import sys
import yaml
import logging

class Dataset:

    def __init__(self):
        self.config = self._load_config()
        self._read_network()
        self._read_mobility_events()
        self._read_disease_infections()

    def _load_config(self, path_to_config="./action_based_risk/config.yaml"):
        with open(path_to_config, "r") as configfile:
            config = yaml.full_load(configfile)
        return config


    def _read_network(self):
        try:
            self.graph = nx.read_gpickle(self.config["dataset_creator"]["network_path"])
        except KeyError as ke:
            print("No found on the path you have given in the config!")
            sys.exit(-1)

    def _read_mobility_events(self):
        pass

    def _read_disease_infections(self):
        pass









if __name__ == "__main__":
    print("Running as main, work still in progress...")
    with open("./action_based_risk/config.yaml", "r") as configfile:
        config = yaml.full_load(configfile)
    print(config["aldi"])