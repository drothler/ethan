from xml.etree import ElementTree as ET
from dask.distributed import LocalCluster, Client, Event, fire_and_forget
import pandas as pd
import dask.dataframe as dd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import logging
import ntpath
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

base_path = Path("../simulation_data/v4/events/")

def read_tree(filepath : Path):
    return ET.parse(filepath)

class Job:

    def __init__(self):
        pass

    def start(self):
        pass


class PrepareDataset(Job):

    def __init__(self, event_folder : Path, facility_folder : Path):
        self.event_folder = event_folder
        self.facility_path = facility_folder


    def start(self, event_file : str):
        return self.parse_event_file(event_file)

    def parse_event_file(self, file : str):
        file_num = 0
        df_columns = ["person", "time", "day", "type", "location", "group_size", "close_contact"]
        data = []# = pd.DataFrame(None, columns=df_columns)
        try:
            day = self._get_day_from_path(file)

            event_name = f"day_{day}"
            logging.info(f"The name of the event to be later set is: {event_name}")
            #event = Event(event_name)

            with open(os.path.join(self.event_folder, file) , "r") as handle:
                content = handle.read()
                content = content.split("\n")
                content = [line.strip("\t") for line in content]
                content = content[2:-1]

                event_count = 0
                for line in content:
                    tree = ET.fromstring(line)
                    event_count += 1
                    bad_lines = 0
                    if event_count % 100000 == 0:
                        logging.info(f"{event_count} events have been processed for day {day}!")

                    row = None
                    if tree.attrib["type"] == "actend":
                        row = self._read_actend(tree, day)
                    elif tree.attrib["type"] == "PersonEntersVehicle" or tree.attrib["type"] == "PersonLeavesVehicle":
                        row = self._read_person_car(tree, day)
                    elif tree.attrib["type"] == "actstart":
                        row = self._read_actstart(tree, day)
                    elif tree.attrib["type"] == "episimContact":
                        row = self._read_episim_contact(tree, day)
                    elif tree.attrib["type"] == "episimAlmostContact":
                        row = self._read_episim_almost_contact(tree, day)
                    elif tree.attrib["type"] == "episimPersonStatus":
                        continue
                    else:
                        bad_lines += 1
                        continue

                    data.append({
                        "person": row[0],
                        "time": row[1],
                        "day": row[2],
                        "type": row[3],
                        "location": row[4],
                        "group_size": row[5],
                        "close_contact": row[6]
                    })

                    if sys.getsizeof(data) >= 4e9:
                        event_df = pd.DataFrame(data, columns=df_columns)
                        event_df.to_csv(f"./event_df_{file_num}_{day}.csv")
                        file_num += 1
                        del event_df



            logging.info(f"The length of the data is: {len(data)}")
            event_df = pd.DataFrame(data)
            event_df.to_csv(f"./event_df_{day}.csv")

            logging.info(f"Setting the {event_name}!")
            #event.set()
        except Exception as e:
            logging.error(str(e))
            logging.info(f"Error happened with {bad_lines} bad lines.")
            event_df = pd.DataFrame(data, columns=df_columns)
            event_df.to_csv(f"./event_df_{day}.csv")


    def _read_actend(self, instance, day):
        return instance.attrib["person"], instance.attrib["time"], day,  instance.attrib["type"], instance.attrib["facility"], \
                1, 0

    def _read_actstart(self, instance, day):
        return instance.attrib["person"], instance.attrib["time"], day, instance.attrib["type"], instance.attrib["facility"], \
               1, 0

    def _read_episim_contact(self, instance, day):
        return instance.attrib["person"], instance.attrib["time"], day, instance.attrib["type"], instance.attrib["container"], \
               instance.attrib["groupSize"], instance.attrib["contactPerson"]

    def _read_episim_almost_contact(self, instance, day):
        return instance.attrib["person"], instance.attrib["time"], day, instance.attrib["type"], instance.attrib["container"],\
               instance.attrib["groupSize"], instance.attrib["contactPerson"]

    def _read_person_car(self, instance, day):
        return instance.attrib["person"], instance.attrib["time"], day, instance.attrib["type"], instance.attrib["vehicle"], \
               1, 0

    def _get_day_from_path(self, path):
        filename = ntpath.basename(path)
        name, extension = filename.split(".")
        word, number = name.split("_")
        return int(number)


async def distribute_preparation():
    files = []
    workers = int(0.9*mp.cpu_count())
    local_cluster = LocalCluster(n_workers=workers, processes=True, threads_per_worker=1, memory_limit=5e9)
    client = await Client(local_cluster, asynchronous=True)
    num_days = 2

    for i in range(1, num_days+1):
        if i % 10 == i:
            files.append(f"day_00{i}.xml")
        else:
            files.append(f"day_0{i}.xml")

    frames = []
    logging.info("Scheduling the files...")
    for i, file in enumerate(files):
        prepare = PrepareDataset(event_folder="./simulation_data/v4/events", facility_folder=".")
        logging.info(f"Scheduling file: {file}")
        df_future = client.submit(prepare.start, file)
        logging.info(f"File {file}, scheduled successfully...")
        frames.append(df_future)

    logging.info("Waiting for the processing to be done.")

    for i in range(1, num_days+1):
        event_name = f"day_{i}"
        logging.info(f"Waiting for {event_name}")
        event = Event(event_name)
        event.wait()

    # logging.info("Joining the frames...")
    # df_frames = []
    # for frame in frames:
    #     df_frames.append(await frame.result())

    # result = pd.concat(df_frames)
    # client.close()

    #result.to_csv(f"./result_df_{num_days}.csv")
    logging.info("Done!")

class CreateActions(Job):

    def __init__(self, folder_location : str):
        self.folder_location = folder_location

    def _get_num_from_path(self, path):
        filename = ntpath.basename(path)
        name, extension = filename.split(".")
        number = name.split("_")[2]
        return int(number)

    def start(self, event_csv_path : Path):
        self.df = pd.read_csv(os.path.join(self.folder_location, event_csv_path))

        actions = []
        action_part = {}
        #event_name = f"file_{self._get_num_from_path(event_csv_path)}"
        #event = Event(event_name)

        group_by_person = self.df.groupby("person")

        cnt = 0
        for person_id, group in group_by_person:

            action_part = {}

            for i, row in group.iterrows():

                if cnt % 100000 == 0:
                    logging.info(f"{cnt} events have been processed!")
                cnt += 1
                
                if row["type"] == "actstart" or row["type"] == "PersonEntersVehicle":
                    action_part["person"] = person_id
                    action_part["time"] = row["time"]
                    action_part["container"] = row["location"]

                if row["type"] == "episimContact":
                    if "groupSize" in action_part:
                        action_part["groupSize"].append(row["group_size"])
                    else:
                        action_part["groupSize"] = []
                        action_part["groupSize"].append(row["group_size"])


                    if "contactPeople" in action_part:
                        action_part["contactPeople"].append(row["close_contact"])
                    else:
                        action_part["contactPeople"] = []
                        action_part["contactPeople"].append(row["close_contact"])

                if row["type"] == "episimAlmostContact":
                    if "groupSize" in action_part:
                        action_part["groupSize"].append(row["group_size"])
                    else:
                        action_part["groupSize"] = []
                        action_part["groupSize"].append(row["group_size"])

                    if "almostContactPeople" in action_part:
                        action_part["almostContactPeople"].append(row["close_contact"])
                    else:
                        action_part["almostContactPeople"] = []
                        action_part["almostContactPeople"].append(row["close_contact"])

                if row["type"] == "actend" or row["type"] == "PersonLeavesVehicle":
                    if "container" not in action_part and "time" not in action_part:
                        #First actend of the day
                        continue

                    action_part["end_time"] = row["time"]
                    action_part = self.complete_action_part(action_part)
                    actions.append(action_part)
                    action_part = {}

        df = pd.DataFrame(actions)
        folder_path = Path(self.folder_location)
        df.to_csv(os.path.join(folder_path / "processed", f"actions_day_{self._get_num_from_path(event_csv_path)}.csv"))
        #event.set()


    def complete_action_part(self, action_part):
        if len(action_part) == 6:
            return action_part
        if "groupSize" not in action_part:
            action_part["groupSize"] = [1]

        if "contactPeople" not in action_part:
            action_part["contactPeople"] = []

        if "almostContactPeople" in action_part:
            action_part["almostContactPeople"] = []

        return action_part



async def distribute_action_creation():
    files = []
    workers = int(0.9*mp.cpu_count())
    local_cluster = LocalCluster(n_workers=workers, processes=True, threads_per_worker=1, memory_limit=3e9)
    client = await Client(local_cluster, asynchronous=True)
    num_days = 3

    for i in range(1, num_days+1):
        files.append(f"event_df_{i}.csv")

    frames = []
    logging.info("Scheduling the files...")
    for i, file in enumerate(files):
        prepare = CreateActions(folder_location="../events_csv")
        logging.info(f"Scheduling file: {file}")
        df_future = client.submit(prepare.start, file)
        logging.info(f"File {file}, scheduled successfully...")
        frames.append(df_future)

    logging.info("Waiting for the processing to be done.")

    for i in range(1, num_days+1):
        event_name = f"file_{i}"
        logging.info(f"Waiting for {event_name}")
        event = Event(event_name)
        event.wait()

    # logging.info("Joining the frames...")
    # df_frames = []
    # for frame in frames:
    #     df_frames.append(await frame.result())

    # result = pd.concat(df_frames)
    # client.close()

    #result.to_csv(f"./result_df_{num_days}.csv")
    logging.info("Done!")


if __name__ == "__main__":
    print("Starting the creation of events...")
    #prepare_dataset = PrepareDataset(event_folder="./simulation_data/v4/events", facility_folder=None)
    #prepare_dataset.start(event_file="day_001.xml")
    print("Event_df created, creating actions...")
    create_actions = CreateActions(".")
    create_actions.start("event_df_1.csv")
    print("Done!")
