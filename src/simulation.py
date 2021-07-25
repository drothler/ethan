import numpy as np
import random
import math
import time
import gc
import json
import cython

# Naive simulator takes the pre-processed graph from DataLoader and outputs a set of encounter nodes
# encounter nodes are calculated with fixed properties based on the connection type
# any person will have a small probability to meet another person
# thus it is potentially possible for a person to have N encounters, although very very unlikely


def get_binary_event(probability):
    return probability > random.random()



class TimeTable:

    def __init__(self, node, attributes, rest_day, rest_day_probability, work_hours, config, facility_schedules={}, random_encounter_pool=None, facilities={}, population_count=100000):
        self.config = config
        self.week_schedule = None
        self.id = node[0]
        self.attributes = attributes
        self.working = False
        self.home = False
        self.school = False
        self.car = get_binary_event(float(self.config['POPULATION_STATS']['car']))
        self.gym = get_binary_event(float(self.config['POPULATION_STATS']['gym']))
        # Idea: happiness represents social happiness
        # social encounters increase happiness
        # lack of social encounters decrease happiness
        # low happiness might increase lack of obedience in terms of pandemic restrictions
        # high happiness increases support for measures

        self.happiness = int(config['SIMULATION']['default_happiness'])
        self.night_shift = get_binary_event(float(config['POPULATION_STATS']['night_shift']))
        self.rest_day = rest_day
        self.rest_day_probability = rest_day_probability
        self.work_hours = work_hours
        self.edu_hours = int(self.config['POPULATION_STATS']['average_school_hours'])
        self.shop_closure = int(self.config['LEGAL_CONDITIONS']['shop_closure'])
        self.half_time_hours = int(self.config['LEGAL_CONDITIONS']['max_halftime_hours'])
        self.sleep_avg = int(self.config['POPULATION_STATS']['average_sleep'])
        self.gym_avg = int(self.config['POPULATION_STATS']['gym_days_per_week'])
        self.half_time_hours = int(self.config['POPULATION_STATS']['average_halftime_work'])
        self.half_time_max = int(self.config['LEGAL_CONDITIONS']['max_halftime_hours'])
        self.gym_days = 0

        self.facility_schedules = facility_schedules
        self.random_encounter_pool = random_encounter_pool

        # still need to be created
        self.shopping_facilities = facilities['shopping']
        self.gyms = facilities['gym']
        self.busses = facilities['bus']
        self.cars = facilities['car']

        self.gym_facility = None
        self.bus_line = None
        self.vehicle = None

        self.favorite_shops = random.sample(self.shopping_facilities, 3)

        self.pool_id = random.choice(range(0, math.floor(population_count * int(self.config['INFRASTRUCTURE']['districts_per_million']) / 1e6)))

        self.work_id = -1
        self.home_id = -1
        self.edu_id = -1

        self.shopping_days = int(self.config['POPULATION_STATS']['shopping'])
        # print(attributes)
        fixed_facilities = {}
        for attribute in attributes:
            attribute_string = attribute[0].replace("\"", "")
            if 'work' in attribute_string:
                self.working = True
                self.work_id = attribute_string
                fixed_facilities['work'] = attribute_string
            elif 'home' in attribute_string:
                self.home = True
                self.home_id = attribute_string
                fixed_facilities['home'] = attribute_string
            elif 'edu' in attribute_string:
                self.school = True
                self.edu_id = attribute_string
                fixed_facilities['edu'] = attribute_string
        self.create_updated_schedule(fixed_facilities)


    # creating a 7 day schedule for a specific node, given its attributes
    # each schedule is a numpy array of shape (24, 7, 2), one tuple for each hour. index 0 is the type, index 1 the id
    # currently the class takes in the attributes in written form, maybe i will change that approach later on
    # id: -1 = sleep, 0 = free time 1 = work, 2 = edu



    def create_updated_schedule(self, attributes):

        if self.gym:
            self.gym_facility = random.choice(tuple(self.gyms))

        if self.car:
            self.vehicle = random.choice(tuple(self.cars))

        self.bus_line = random.choice(tuple(self.busses))


        offset = int(math.floor(random.gauss(mu=0, sigma=2)))
        self.week_schedule = np.empty(shape=(7, 24), dtype=object)

        sunday = set([6])
        work_days = random.sample(list(range(0, 6)), random.randint(4, 5))
        free_days = set(range(0, 6)).difference(work_days)

        if not self.working and not self.school:
            free_days = list(range(0, 6))

        for i, row in enumerate(self.week_schedule):
            for j, _ in enumerate(row):
                self.week_schedule[i][j] = None

        if self.gym:
            self.gym_days = math.floor(max(min(7, random.gauss(mu=self.gym_avg, sigma=1)), 1))
            if self.facility_schedules.get(self.gym_facility) is None:
                self.facility_schedules[self.gym_facility] = np.empty(shape=(7, 24), dtype=object)
                for day, row in enumerate(self.facility_schedules[self.gym_facility]):
                    for hour, _ in enumerate(row):
                        self.facility_schedules[self.gym_facility][day][hour] = set()



        # creating schedules
        # first we check for only workers and create their work days
        if self.working and not self.school:
            self.create_workdays(offset=offset, indices=work_days, facilities=attributes)

        # then we check for people who work and go to school and create their school-work days
        elif self.working and self.school:
            self.create_workschooldays(offset, indices=work_days, facilities=attributes)

        # finally we check for students that do not work
        elif self.school:
            self.create_schooldays(offset, indices=work_days, facilities=attributes)

        # in the end we create the free day schedules and sundays
        self.create_freeday(offset, indices=free_days, facilities=attributes)
        self.create_sunday(offset, indices=sunday, facilities=attributes)


    def create_workdays(self, offset, indices, facilities):
        initial_offset = offset
        work_facility = facilities['work']
        home_facility = facilities['home']
        transit_length = random.randint(1, 2)
        transit_method = None
        if self.car:
            transit_method = self.vehicle
        elif get_binary_event(0.9):
            transit_method = self.bus_line
        else:
            transit_method = 'outdoor'



        if self.facility_schedules.get(home_facility) is None:
            self.facility_schedules[home_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[home_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[home_facility][day][hour] = set()

        if self.facility_schedules.get(work_facility) is None:
            self.facility_schedules[work_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[work_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[work_facility][day][hour] = set()

        for index in indices:
            offset = initial_offset
            sleep_hours = max(min(math.floor(random.gauss(mu=self.sleep_avg, sigma=1)), 10), 1)
            for sleep in range(0, sleep_hours):

                # adding tuple of sleep and the facility to schedule
                self.week_schedule[index][(sleep + offset) % 24] = ('sleep', home_facility)

                self.facility_schedules[home_facility][index][(sleep + offset) % 24].add(self.id)

            # increasing offset by the sleep hour count
            offset = offset + sleep_hours

            for transit in range(0, transit_length):
                self.week_schedule[index][(transit + offset) % 24] = (transit_method.split('_')[0], transit_method)
                if transit_method == self.bus_line:
                    # TODO: add person to bus time table at index, hour and check for capacity
                    self.facility_schedules[self.bus_line][index][(transit + offset) % 24].add(self.id)
                else:
                    self.facility_schedules[transit_method][index][(transit + offset) % 24].add(self.id)

            # increasing offset by transit length
            offset = offset + transit_length

            # setting the work hour count
            work_hours = max(min(math.floor(random.gauss(mu=self.work_hours, sigma=1)), 10), self.work_hours-1)

            for work in range(0, work_hours):
                self.week_schedule[index][(work + offset) % 24] = ('work', work_facility)
                self.facility_schedules[work_facility][index][(work + offset) % 24].add(self.id)

            # increasing offset by the work hour count
            offset = offset + work_hours

            for transit in range(0, transit_length):
                self.week_schedule[index][(transit + offset) % 24] = (transit_method.split('_')[0], transit_method)
                if transit_method == self.bus_line:
                    # TODO: add person to bus time table at index, hour and check for capacity
                    self.facility_schedules[self.bus_line][index][(transit + offset) % 24].add(self.id)
                else:
                    self.facility_schedules[transit_method][index][(transit + offset) % 24].add(self.id)

            # increasing offset by transit length
            offset = offset + transit_length

            shopping_time = None
            shopping = get_binary_event(self.shopping_days/6)

            for freetime in range(0, 24 - offset + initial_offset):
                self.week_schedule[index][(freetime + offset) % 24] = ('freetime', home_facility)
                self.random_encounter_pool[index][(freetime + offset) % 24]['general'][self.pool_id].add(self.id)
                self.facility_schedules[home_facility][index][(freetime + offset) % 24].add(self.id)

            if self.gym and self.gym_days > 0 and get_binary_event(self.gym_avg / 7):

                gym_hours = math.floor(max(1, min(3, random.gauss(mu=1, sigma=0.5))))
                last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                while last_gym_start_time < 0 and gym_hours >= 1:
                    gym_hours -= 1
                    last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                if last_gym_start_time >= 0:
                    gym_time = random.randint(0, last_gym_start_time)

                    for i in range(0, gym_hours):
                        self.week_schedule[index][(offset + gym_time + i) % 24] = ('freetime', self.gym_facility)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24]['general'][self.pool_id].remove(self.id)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24][self.gym_facility].add(self.id)
                        self.facility_schedules[home_facility][index][(offset + gym_time + i) % 24].remove(self.id)
                        self.facility_schedules[self.gym_facility][index][(offset + gym_time + i) % 24].add(self.id)

            if shopping and self.shop_closure > offset:
                last_hour = self.shop_closure - offset + initial_offset
                if (last_hour) >= 0:
                    shopping_time = random.randint(0, (last_hour))

                    # could turn to endless loop if gym_hours == free time hours, need to keep in mind
                    counter = 0
                    while self.week_schedule[index][(offset + shopping_time) % 24][1] == self.gym_facility:
                        shopping_time = random.randint(0, (last_hour))
                        counter += 1
                        if counter > 20:
                            print('Probably stuck...')
                            print('Conflict with', self.week_schedule[index][(offset + shopping_time) % 24][1])
                            print(self.week_schedule[index])
                            print('shopping time', shopping_time)
                            print('Offset', offset)
                            print('Initial Offset', initial_offset)
                            print('Shops close at', self.shop_closure)
                            print('Last possible shopping hour at', last_hour)
                            wait = input("Loop is stuck, analyze schedule")

                    shop = random.choice(self.favorite_shops)
                    self.week_schedule[index][(offset + shopping_time) % 24] = ('freetime', shop)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24]['general'][self.pool_id].remove(self.id)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24][shop].add(self.id)
                    self.facility_schedules[home_facility][index][(shopping_time + offset) % 24].remove(self.id)
                    self.facility_schedules[shop][index][(offset + shopping_time) % 24].add(self.id)


    def create_schooldays(self, offset, indices, facilities):
        initial_offset = offset
        edu_facility = facilities['edu']
        home_facility = facilities['home']
        transit_length = random.randint(1, 2)
        transit_method = None
        if self.car:
            transit_method = self.vehicle
        elif get_binary_event(0.9):
            transit_method = self.bus_line
        else:
            transit_method = 'outdoor'

        if self.facility_schedules.get(home_facility) is None:
            self.facility_schedules[home_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[home_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[home_facility][day][hour] = set()

        if self.facility_schedules.get(edu_facility) is None:
            self.facility_schedules[edu_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[edu_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[edu_facility][day][hour] = set()

        for index in indices:
            offset = initial_offset
            sleep_hours = max(min(math.floor(random.gauss(mu=self.sleep_avg, sigma=1)), 10), 1)
            for sleep in range(0, sleep_hours):
                # adding tuple of sleep and the facility to schedule
                self.week_schedule[index][(sleep + offset) % 24] = ('sleep', home_facility)

                self.facility_schedules[home_facility][index][(sleep + offset) % 24].add(self.id)

            # increasing offset by the sleep hour count
            offset = offset + sleep_hours

            for transit in range(0, transit_length):
                self.week_schedule[index][(transit + offset) % 24] = (transit_method.split('_')[0], transit_method)
                if transit_method == self.bus_line:
                    # TODO: add person to bus time table at index, hour and check for capacity
                    self.facility_schedules[self.bus_line][index][(transit + offset) % 24].add(self.id)
                else:
                    self.facility_schedules[transit_method][index][(transit + offset) % 24].add(self.id)

            # increasing offset by transit length
            offset = offset + transit_length

            # setting the edu hour count
            edu_hours = max(min(math.floor(random.gauss(mu=self.edu_hours, sigma=1)), 10), self.edu_hours - 1)

            for edu in range(0, edu_hours):
                self.week_schedule[index][(edu + offset) % 24] = ('edu', edu_facility)
                self.facility_schedules[edu_facility][index][(edu + offset) % 24].add(self.id)

            # increasing offset by the edu hour count
            offset = offset + edu_hours

            for transit in range(0, transit_length):
                self.week_schedule[index][(transit + offset) % 24] = (transit_method.split('_')[0], transit_method)
                if transit_method == self.bus_line:
                    # TODO: add person to bus time table at index, hour and check for capacity
                    self.facility_schedules[self.bus_line][index][(transit + offset) % 24].add(self.id)
                else:
                    self.facility_schedules[transit_method][index][(transit + offset) % 24].add(self.id)

            # increasing offset by transit length
            offset = offset + transit_length

            shopping_time = None
            shopping = get_binary_event(self.shopping_days / 6)

            for freetime in range(0, 24 - offset + initial_offset):
                self.week_schedule[index][(freetime + offset) % 24] = ('freetime', home_facility)
                self.random_encounter_pool[index][(freetime + offset) % 24]['general'][self.pool_id].add(self.id)
                self.facility_schedules[home_facility][index][(freetime + offset) % 24].add(self.id)

            if self.gym and self.gym_days > 0 and get_binary_event(self.gym_avg / 7):

                gym_hours = math.floor(max(1, min(3, random.gauss(mu=1, sigma=0.5))))
                last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                while last_gym_start_time < 0 and gym_hours >= 1:
                    gym_hours -= 1
                    last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                if last_gym_start_time >= 0:
                    gym_time = random.randint(0, last_gym_start_time)

                    for i in range(0, gym_hours):
                        self.week_schedule[index][(offset + gym_time + i) % 24] = ('freetime', self.gym_facility)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24]['general'][self.pool_id].remove(self.id)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24][self.gym_facility].add(self.id)
                        self.facility_schedules[home_facility][index][(offset + gym_time + i) % 24].remove(self.id)
                        self.facility_schedules[self.gym_facility][index][(offset + gym_time + i) % 24].add(self.id)

            if shopping and self.shop_closure > offset:
                last_hour = self.shop_closure - offset + initial_offset
                if (last_hour) >= 0:
                    shopping_time = random.randint(0, (last_hour))

                    # could turn to endless loop if gym_hours == free time hours, need to keep in mind
                    counter = 0
                    while self.week_schedule[index][(offset + shopping_time) % 24][1] == self.gym_facility:
                        shopping_time = random.randint(0, (last_hour))
                        counter += 1
                        if counter > 20:
                            print('Probably stuck...')
                            print('Conflict with', self.week_schedule[index][(offset + shopping_time) % 24][1])
                            print(self.week_schedule[index])
                            print('shopping time', shopping_time)
                            print('Offset', offset)
                            print('Initial Offset', initial_offset)
                            print('Shops close at', self.shop_closure)
                            print('Last possible shopping hour at', last_hour)
                            wait = input("Loop is stuck, analyze schedule")

                    shop = random.choice(self.favorite_shops)
                    self.week_schedule[index][(offset + shopping_time) % 24] = ('freetime', shop)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24]['general'][self.pool_id].remove(self.id)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24][shop].add(self.id)
                    self.facility_schedules[home_facility][index][(shopping_time + offset) % 24].remove(self.id)
                    self.facility_schedules[shop][index][(offset + shopping_time) % 24].add(self.id)




    def create_workschooldays(self, offset, indices, facilities):
        initial_offset = offset
        edu_facility = facilities['edu']
        home_facility = facilities['home']
        work_facility = facilities['work']
        transit_length = random.randint(1, 2)
        transit_method = None
        if self.car:
            transit_method = self.vehicle
        elif get_binary_event(0.9):
            transit_method = self.bus_line
        else:
            transit_method = 'outdoor'

        if self.facility_schedules.get(home_facility) is None:
            self.facility_schedules[home_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[home_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[home_facility][day][hour] = set()

        if self.facility_schedules.get(edu_facility) is None:
            self.facility_schedules[edu_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[edu_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[edu_facility][day][hour] = set()

        if self.facility_schedules.get(work_facility) is None:
            self.facility_schedules[work_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[work_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[work_facility][day][hour] = set()

        for index in indices:
            offset = initial_offset
            sleep_hours = max(min(math.floor(random.gauss(mu=self.sleep_avg, sigma=1)), 10), 1)
            for sleep in range(0, sleep_hours):
                # adding tuple of sleep and the facility to schedule
                self.week_schedule[index][(sleep + offset) % 24] = ('sleep', home_facility)

                self.facility_schedules[home_facility][index][(sleep + offset) % 24].add(self.id)

            # increasing offset by the sleep hour count
            offset = offset + sleep_hours

            for transit in range(0, transit_length):
                self.week_schedule[index][(transit + offset) % 24] = (transit_method.split('_')[0], transit_method)
                if transit_method == self.bus_line:
                    # TODO: add person to bus time table at index, hour and check for capacity
                    self.facility_schedules[self.bus_line][index][(transit + offset) % 24].add(self.id)
                else:
                    self.facility_schedules[transit_method][index][(transit + offset) % 24].add(self.id)

            # increasing offset by transit length
            offset = offset + transit_length

            # setting the edu hour count
            edu_hours = max(min(math.floor(random.gauss(mu=self.edu_hours, sigma=1)), 10), self.edu_hours - 1)

            for edu in range(0, edu_hours):
                self.week_schedule[index][(edu + offset) % 24] = ('edu', edu_facility)
                self.facility_schedules[edu_facility][index][(edu + offset) % 24].add(self.id)

            # increasing offset by the edu hour count
            offset = offset + edu_hours


            work_hours = max(min(self.half_time_max, random.gauss(mu=self.half_time_hours, sigma=1)), 1)

            for work in range(0, work_hours):
                self.week_schedule[index][(work + offset) % 24] = ('work', work_facility)
                self.facility_schedules[work_facility][index][(work + offset) % 24].add(self.id)

            offset = offset + work_hours

            for transit in range(0, transit_length):
                self.week_schedule[index][(transit + offset) % 24] = (transit_method.split('_')[0], transit_method)
                if transit_method == self.bus_line:
                    # TODO: add person to bus time table at index, hour and check for capacity
                    self.facility_schedules[self.bus_line][index][(transit + offset) % 24].add(self.id)
                else:
                    self.facility_schedules[transit_method][index][(transit + offset) % 24].add(self.id)

            # increasing offset by transit length
            offset = offset + transit_length

            shopping_time = None
            shopping = get_binary_event(self.shopping_days / 6)

            for freetime in range(0, 24 - offset + initial_offset):
                self.week_schedule[index][(freetime + offset) % 24] = ('freetime', home_facility)
                self.random_encounter_pool[index][(freetime + offset) % 24]['general'][self.pool_id].add(self.id)
                self.facility_schedules[home_facility][index][(freetime + offset) % 24].add(self.id)

            if self.gym and self.gym_days > 0 and get_binary_event(self.gym_avg / 7):

                gym_hours = math.floor(max(1, min(3, random.gauss(mu=1, sigma=0.5))))
                last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                while last_gym_start_time < 0 and gym_hours >= 1:
                    gym_hours -= 1
                    last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                if last_gym_start_time >= 0:
                    gym_time = random.randint(0, last_gym_start_time)

                    for i in range(0, gym_hours):
                        self.week_schedule[index][(offset + gym_time + i) % 24] = ('freetime', self.gym_facility)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24]['general'][self.pool_id].remove(self.id)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24][self.gym_facility].add(self.id)
                        self.facility_schedules[home_facility][index][(offset + gym_time + i) % 24].remove(self.id)
                        self.facility_schedules[self.gym_facility][index][(offset + gym_time + i) % 24].add(self.id)

            if shopping and self.shop_closure > offset:
                last_hour = self.shop_closure - offset + initial_offset
                if (last_hour) >= 0:
                    shopping_time = random.randint(0, (last_hour))

                    # could turn to endless loop if gym_hours == free time hours, need to keep in mind
                    counter = 0
                    while self.week_schedule[index][(offset + shopping_time) % 24][1] == self.gym_facility:
                        shopping_time = random.randint(0, (last_hour))
                        counter += 1
                        if counter > 20:
                            print('Probably stuck...')
                            print('Conflict with', self.week_schedule[index][(offset + shopping_time) % 24][1])
                            print(self.week_schedule[index])
                            print('shopping time', shopping_time)
                            print('Offset', offset)
                            print('Initial Offset', initial_offset)
                            print('Shops close at', self.shop_closure)
                            print('Last possible shopping hour at', last_hour)
                            wait = input("Loop is stuck, analyze schedule")

                    shop = random.choice(self.favorite_shops)
                    self.week_schedule[index][(offset + shopping_time) % 24] = ('freetime', shop)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24]['general'][self.pool_id].remove(self.id)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24][shop].add(self.id)
                    self.facility_schedules[home_facility][index][(shopping_time + offset) % 24].remove(self.id)
                    self.facility_schedules[shop][index][(offset + shopping_time) % 24].add(self.id)




    def create_freeday(self, offset, indices, facilities):
        initial_offset = offset
        home_facility = facilities['home']
        transit_length = random.randint(1, 2)
        transit_method = None
        if self.car:
            transit_method = self.vehicle
        elif get_binary_event(0.9):
            transit_method = self.bus_line
        else:
            transit_method = 'outdoor'

        if self.facility_schedules.get(home_facility) is None:
            self.facility_schedules[home_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[home_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[home_facility][day][hour] = set()

        for index in indices:
            offset = initial_offset
            sleep_hours = max(min(math.floor(random.gauss(mu=self.sleep_avg, sigma=1)), 10), 1)
            for sleep in range(0, sleep_hours):
                # adding tuple of sleep and the facility to schedule
                self.week_schedule[index][(sleep + offset) % 24] = ('sleep', home_facility)

                self.facility_schedules[home_facility][index][(sleep + offset) % 24].add(self.id)

            # increasing offset by the sleep hour count
            offset = offset + sleep_hours

            shopping_time = None
            shopping = get_binary_event(self.shopping_days / 6)

            for freetime in range(0, 24 - offset + initial_offset):
                self.week_schedule[index][(freetime + offset) % 24] = ('freetime', home_facility)
                self.facility_schedules[home_facility][index][(freetime + offset) % 24].add(self.id)
                self.random_encounter_pool[index][(freetime + offset) % 24]['general'][self.pool_id].add(self.id)

            if self.gym and self.gym_days > 0 and get_binary_event(self.gym_avg / 7):

                gym_hours = math.floor(max(1, min(3, random.gauss(mu=1, sigma=0.5))))
                last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                while last_gym_start_time < 0 and gym_hours >= 1:
                    gym_hours -= 1
                    last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                if last_gym_start_time >= 0:
                    gym_time = random.randint(0, last_gym_start_time)

                    for i in range(0, gym_hours):
                        self.week_schedule[index][(offset + gym_time + i) % 24] = ('freetime', self.gym_facility)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24]['general'][self.pool_id].remove(self.id)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24][self.gym_facility].add(self.id)
                        self.facility_schedules[home_facility][index][(offset + gym_time + i) % 24].remove(self.id)
                        self.facility_schedules[self.gym_facility][index][(offset + gym_time + i) % 24].add(self.id)

            if shopping and self.shop_closure > offset:
                last_hour = self.shop_closure - offset + initial_offset
                if (last_hour) >= 0:
                    shopping_time = random.randint(0, (last_hour))

                    # could turn to endless loop if gym_hours == free time hours, need to keep in mind
                    counter = 0
                    while self.week_schedule[index][(offset + shopping_time) % 24][1] == self.gym_facility:
                        shopping_time = random.randint(0, (last_hour))
                        counter += 1
                        if counter > 20:
                            print('Probably stuck...')
                            print('Conflict with', self.week_schedule[index][(offset + shopping_time) % 24][1])
                            print(self.week_schedule[index])
                            print('shopping time', shopping_time)
                            print('Offset', offset)
                            print('Initial Offset', initial_offset)
                            print('Shops close at', self.shop_closure)
                            print('Last possible shopping hour at', last_hour)
                            wait = input("Loop is stuck, analyze schedule")

                    shop = random.choice(self.favorite_shops)
                    self.week_schedule[index][(offset + shopping_time) % 24] = ('freetime', shop)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24]['general'][self.pool_id].remove(self.id)
                    self.random_encounter_pool[index][(offset + shopping_time) % 24][shop].add(self.id)
                    self.facility_schedules[home_facility][index][(shopping_time + offset) % 24].remove(self.id)
                    self.facility_schedules[shop][index][(offset + shopping_time) % 24].add(self.id)




    def create_sunday(self, offset, facilities, indices):
        initial_offset = offset
        home_facility = facilities['home']
        transit_length = random.randint(1, 2)
        transit_method = None
        if self.car:
            transit_method = self.vehicle
        elif get_binary_event(0.9):
            transit_method = self.bus_line
        else:
            transit_method = 'outdoor'

        if self.facility_schedules.get(home_facility) is None:
            self.facility_schedules[home_facility] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_schedules[home_facility]):
                for hour, _ in enumerate(row):
                    self.facility_schedules[home_facility][day][hour] = set()

        for index in indices:
            offset = initial_offset
            sleep_hours = max(min(math.floor(random.gauss(mu=self.sleep_avg, sigma=1)), 10), 1)

            for sleep in range(0, sleep_hours):
                # adding tuple of sleep and the facility to schedule
                self.week_schedule[index][(sleep + offset) % 24] = ('sleep', home_facility)

                self.facility_schedules[home_facility][index][(sleep + offset) % 24].add(self.id)

            # increasing offset by the sleep hour count
            offset = offset + sleep_hours

            for freetime in range(0, 24 - offset + initial_offset):
                self.week_schedule[index][(freetime + offset) % 24] = ('freetime', home_facility)
                self.facility_schedules[home_facility][index][(freetime + offset) % 24].add(self.id)
                self.random_encounter_pool[index][(freetime + offset) % 24]['general'][self.pool_id].add(self.id)

            if self.gym and self.gym_days > 0 and get_binary_event(self.gym_avg / 7):

                gym_hours = math.floor(max(1, min(3, random.gauss(mu=1, sigma=0.5))))
                last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                while last_gym_start_time < 0 and gym_hours >= 1:
                    gym_hours -= 1
                    last_gym_start_time = self.shop_closure - offset + initial_offset - gym_hours

                if last_gym_start_time >= 0:
                    gym_time = random.randint(0, last_gym_start_time)

                    for i in range(0, gym_hours):
                        self.week_schedule[index][(offset + gym_time + i) % 24] = ('freetime', self.gym_facility)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24]['general'][self.pool_id].remove(self.id)
                        self.random_encounter_pool[index][(offset + gym_time + i) % 24][self.gym_facility].add(self.id)
                        self.facility_schedules[home_facility][index][(offset + gym_time + i) % 24].remove(self.id)
                        self.facility_schedules[self.gym_facility][index][(offset + gym_time + i) % 24].add(self.id)



    def return_facilites(self):
        return self.facility_schedules

    def return_random_encounter_pool(self):
        return self.random_encounter_pool

    def get_schedule(self):
        return self.week_schedule

    def get_work_id(self):
        return self.work_id

    def get_home_id(self):
        return self.home_id

    def get_edu_id(self):
        return self.edu_id

    def get_pool_id(self):
        return self.pool_id

    def __str__(self):
        output = self.week_schedule.__str__()
        return output


# Each Facility gets a schedule as well with one slot for each hour on all 7 days
# Each slot contains a list of all node ids that are present in that facility during that timeslot
# when passing the data from the LTSN to this simulator, each node gets assigned their schedule
# during that assignment they also get placed in lists of the according facility timetable slot

class FacilityTable:
    def __init__(self, id):
        self.id = id
        self.schedule = np.empty(shape=(24, 7), dtype=object)
        for i, row in enumerate(self.schedule):
            for j, _ in enumerate(row):
                self.schedule[i][j] = []

    def append_node(self, node_id, timeslot, day):
        if node_id not in self.schedule[timeslot][day]:
            self.schedule[timeslot][day].append(node_id)

    def remove_node_at_time(self, node_id, timeslot, day):
        if id in self.schedule[timeslot][day]:
            self.schedule[timeslot][day].remove(node_id)

    def eradicate_node(self, node_id):
        for i, row in enumerate(self.schedule):
            for j, _ in enumerate(row):
                self.remove_node_at_time(node_id=node_id, timeslot=i, day=j)

    def get_facility_schedule(self):
        return self.schedule

    def __str__(self):
        return self.id + '\n - '+ self.schedule.__str__()

# Each Encounter has a

class Encounter:
    def __init__(self, id_a, id_b, duration, intensity, facility_type, time):
        self.duration = duration
        self.intensity = intensity
        self.id_a = min(id_a, id_b)
        self.id_b = max(id_a, id_b)
        self.facility_type = facility_type
        self.time = time

    def get_duration(self):
        return self.duration

    def get_intensity(self):
        return self.intensity

    def get_id(self):
        return self.id

    def get_encounter(self):
        return self.encounter

    def __str__(self):
        return 'Pair: ' + str(self.id_a) + ', ' + str(self.id_b) + ', duration: ' + str(self.duration) + ', intensity: '+ str(self.intensity) + ', facility type: ' + str(self.facility_type) + ', time: '+ str(self.time)

    def __eq__(self, other):
        if isinstance(other, Encounter):
            return self.duration == other.duration \
                   and self.intensity == other.intensity \
                   and self.facility_type == other.facility_type \
                   and self.time == other.time \
                   and ((self.id_a == other.id_a and self.id_b == other.id_b)
                        or (self.id_a == other.id_b and self.id_b == other.id_a))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.duration, self.intensity, self.time, self.id_a, self.id_b, self.facility_type))

    def get_tuple(self, id):
        if self.id_a == id:
            enc = self.id_b
        else:
            enc = self.id_a
        return tuple([enc, self.intensity, self.duration, self.time, self.facility_type])


class Simulator:

    def __init__(self, graph, loader):
        print('Initializing Simulator Instance')
        self.graph = graph
        self.data_loader = loader

class TimetableSimulator(Simulator):

    def __init__(self, graph, info, loader, configuration, start_day=0, rest_day=6, rest_day_probability=0.9, work_hours=8):
        Simulator.__init__(self, graph, loader)
        print('Initializing Time Table Simulator Instance')
        self.config = configuration

        self.day = start_day
        self.rest_day = rest_day
        self.rest_day_probability = rest_day_probability

        self.district_fluctuation = float(self.config['POPULATION_STATS']['district_fluctuation'])
        self.district_count = math.floor(int(self.config['INFRASTRUCTURE']['districts_per_million']) / 1e6 * len(graph))
        self.districts = range(0, self.district_count)

        self.work_hours = work_hours
        self.info = info
        self.schedules = list()

        self.facilities = {}

        self.facility_tables = {}
        self.encounters = {}
        self.random_encounters = {}
        self.random_encounter_pool = np.empty(shape=(7, 24), dtype=object)
        self.potential_fixed_encounters = {}

        self.quarantine = None
        self.isolation = None

        for i, row in enumerate(self.random_encounter_pool):
            for j, _ in enumerate(row):
                self.random_encounter_pool[i][j] = {}
                self.random_encounter_pool[i][j]['general'] = list()
                for district in self.districts:
                    self.random_encounter_pool[i][j]['general'].append(set())
        self.create_facilities()
        self.create_schedules()
        print('Schedules created.')
        for node in self.graph:
            self.encounters[node[0]] = set()
            self.random_encounters[node[0]] = set()
            self.potential_fixed_encounters[node[0]] = list()
        self.pools_converted = False

    # creating the schedules for each node

    def create_schedules(self):
        start_time = time.time()
        for index, active_node in enumerate(self.graph):
            timetable = TimeTable(node=active_node, attributes=self.info[index], rest_day=6, rest_day_probability=0.9, work_hours=8, config=self.config, facility_schedules=self.facility_tables, random_encounter_pool=self.random_encounter_pool, facilities=self.facilities, population_count=len(self.graph))
            self.facility_tables = timetable.return_facilites()
            self.schedules.append(timetable)
            if index % 1000 == 0:
                print('Created', index, 'schedules')
        end_time = time.time()
        print(end_time - start_time)

    #creating all necessary facilities

    def create_facilities(self):

        self.facility_tables['outdoor'] = np.empty(shape=(7, 24), dtype=object)
        for day, row in enumerate(self.facility_tables['outdoor']):
            for hour, _ in enumerate(row):
                self.facility_tables['outdoor'][day][hour] = set()

        # reading facilities from config string, removing spaces and splitting on commas
        autogenerated_facilities = self.config['SIMULATION']['autogenerated_facilities'].replace(" ", "").split(",")
        seperate_pools = self.config['SIMULATION']['seperate_pools'].replace(" ", "").split(",")
        for facility_index, facility in enumerate(autogenerated_facilities):
            is_pooled = bool(seperate_pools[facility_index])
            count_string = facility + '_per_million'
            count = int(self.config['POPULATION_STATS'][count_string])
            self.create_facility(facility, count, is_pooled)


    def create_facility(self, facility_type, count, seperately_poooled):
        facility_count = math.floor(count * len(self.graph) / 1e6)
        self.facilities[facility_type] = set()
        for facility in range(facility_count):
            facility_id = facility_type + '_' + str(facility)
            self.facilities[facility_type].add(facility_id)
            self.facility_tables[facility_id] = np.empty(shape=(7, 24), dtype=object)
            for day, row in enumerate(self.facility_tables[facility_id]):
                for hour, _ in enumerate(row):
                    self.facility_tables[facility_id][day][hour] = set()
                    if seperately_poooled:
                        self.random_encounter_pool[day][hour][facility_id] = set()

    def convert_pools(self):
        for day in range(0, 7):
            for hour in range(0, 24):
                for key, values in self.random_encounter_pool[day][hour].items():
                    if key is 'general':
                        for set in self.districts:
                            self.random_encounter_pool[day][hour][key][set] = np.array(list(self.random_encounter_pool[day][hour][key][set]))
                    else:
                        self.random_encounter_pool[day][hour][key] = np.array(list(values))



    # simulating a single day / time step takes into account whether the current day is a rest day or not
    # basic structure is seperating the day into 24 hours (wow) and assigning time slots to certain activities
    # for now fixed activity slots, in the future i want to implement flexible slots for each person
    def create_encounters(self):
        if not self.pools_converted:
            self.convert_pools()
            self.pools_converted = True
        print('Encounters for Day', self.day)
        print('Creating Connected Encounters...')
        start = time.time()
        for index, node in enumerate(self.graph):
            self.create_connected_encounters(schedule=self.schedules[index], node=node)
            # print(self.encounters[node[0]])
        end = time.time()
        print('Time elapsed for connected Encounters: ' + str(end - start))
        print(self.encounters[0])
        start_ = time.time()
        for index, node in enumerate(self.graph):
            if index % 1000 == 0 and index > 0:
                #print('Created fixed encounters for node', index)
                end = time.time()
                #print('Took', end - start, 'seconds for 1000 nodes')
                #print(self.encounters[index - 1])
                start = time.time()
            self.create_fixed_encounters_from_schedule(schedule=self.schedules[index], node=node)

        end = time.time()
        print('Time elapsed for fixed Encounters: '+str(end-start_))
        #print(self.encounters[0])
        #print('Creating Random Encounters...')
        start_random = time.time()
        for index, node in enumerate(self.graph):
            if index % 1000 == 0 and index > 0:
                #print('Created random encounters for node', index)
                end = time.time()
                #print('Took', end - start, 'seconds for 1000 nodes')
                #print(self.random_encounters[index - 1])
                start = time.time()
            self.create_random_encounters_from_schedule(schedule=self.schedules[index], node=node)
        end = time.time()
        print('Time elapsed for random Encounters: ' + str(end - start_random))
        #print(self.encounters[0])
        for node in self.graph:
            for element in self.random_encounters[node[0]]:
                self.encounters[node[0]].add(element)


    def create_fixed_encounters_from_schedule(self, node, schedule):
        potential_encounters = self.potential_fixed_encounters[node[0]]                                               # O(1)
        if potential_encounters == []:
            active_encounters = set()                                                      # O(1)
            for hour in range(0, 24):                                                           # 24 *
                facility = schedule.get_schedule()[self.day][hour][1]                              # O(1)
                if facility != 'outdoor':
                    facility_encounters = self.facility_tables[facility]                                    # O(1)
                    candidates = self.prepare_hours(set(facility_encounters[self.day][hour]), node[0])      # O(1)
                    facility_encounters[self.day][hour].add(node[0])                                        # O(1)
                    finished_encounters, active_encounters = self.track_encounters(next_list=candidates, facility_type=facility, active_encounters=active_encounters)
                    potential_encounters.extend(finished_encounters)                                        # O(len(finished_encounters))
            potential_encounters.extend(active_encounters)
            self.potential_fixed_encounters[node[0]] = potential_encounters
        factor = 1
        if self.config['RESTRICTIONS']['masks'] == '1':
            factor = (1 - float(self.config['RESTRICTIONS']['mask_intensity_reduction']))
        for encounter in potential_encounters:
            if encounter is not []:
                encounter_type = encounter[2].split('_')[0]
                probability = self.config['ENCOUNTER_PROBABILITIES'][encounter_type]
                #print(encounter_type, type(encounter_type), probability, type(probability))
                if get_binary_event(float(probability)):
                    intensity = factor * math.floor(min(10, max(1, random.gauss(mu=float(self.config['ENCOUNTER_INTENSITIES'][encounter_type]), sigma=1))))


                    duration = math.floor(max(min(int(encounter[1]*10), random.gauss(mu=float(encounter[1]*5), sigma=float(encounter[1]*2.5))), 1))
                    facility_type = encounter_type
                    first_person = node[0]
                    second_person = encounter[0]
                    enc = Encounter(intensity=intensity, duration=duration, facility_type=facility_type, id_a=first_person, id_b=second_person, time=0)
                    if enc not in self.encounters[first_person]:
                        self.encounters[first_person].add(enc)
                        # print(enc)
                    if enc not in self.encounters[second_person]:
                        self.encounters[second_person].add(enc)
                        # print(enc)

    def prepare_hours(self, next_list, id):
        next_list.remove(id)
        return next_list

    def track_encounters(self, next_list, facility_type, active_encounters):    # O(len(next_list)*len(active_encounters))
        if active_encounters is {} and next_list is {}:
            return set(), set()
        elif active_encounters is {}:
            while next_list:
                encounter = next_list.pop()
                active_encounters.add((encounter, 1, facility_type))
            return set(), active_encounters
        else:
            new_active_encounters = set()
            old_encounters = set()
            while next_list:
                encounter = next_list.pop()
                found = False
                while active_encounters:
                    node = active_encounters.pop()
                    if encounter == node[0] and facility_type is node[2]:
                        count = node[1] + 1
                        new_active_encounters.add((node[0], count, node[2]))
                        found = True
                    else:
                        old_encounters.add(node)

                if not found:
                    new_active_encounters.add((encounter, 1, facility_type))

            return old_encounters, new_active_encounters

    def create_connected_encounters(self, node, schedule):
        free_hours = self.count_free_time(schedule=schedule.get_schedule()[self.day])*10
        count = 0
        for connection in node[1]:
            type = self.get_maximum_intensity(connection[1])
            if get_binary_event(float(self.config['FREETIME_PROBABILITIES'][type])):

                duration = math.floor(min(free_hours, max(1, random.gauss(mu=int(self.config['FREETIME_MEAN_DURATIONS'][type]), sigma=int(self.config['FREETIME_SIGMA_DURATIONS'][type])))))
                intensity = math.floor(random.gauss(mu=int(self.config['CONNECTION_INTENSITIES'][type]), sigma=int(self.config['CONNECTION_INTENSITIES_SIGMA'][type])))
                # print(intensity)
                location = ''
                if get_binary_event(0.33):
                    location = schedule.get_home_id()
                elif get_binary_event(0.33):
                    location = self.schedules[connection[0]].get_home_id()
                else:
                    if int(self.config['RESTRICTIONS']['masks']) == 1:
                        # might wanna implement a reduction in encounter intensity here depending on the facility and encounter
                        intensity *= (1 - float(self.config['RESTRICTIONS']['mask_intensity_reduction']))
                    location = 'outdoor'
                    intensity = max(1, math.floor(intensity/2))
                enc = Encounter(duration=duration, intensity=intensity, id_a=node[0], id_b=connection[0], facility_type=location, time=12 + math.floor(duration/10))
                if enc not in self.encounters[node[0]]:
                    # print('Added fixed connection for node', node[0], intensity)
                    self.encounters[node[0]].add(enc)
                if enc not in self.encounters[connection[0]]:
                    self.encounters[connection[0]].add(enc)
                count += 1
                if int(self.config['RESTRICTIONS']['contact_restrictions']) == 1 and count >= int(self.config['RESTRICTIONS']['allowed_contacts']):
                    break

    def get_maximum_intensity(self, connection_types):
        maximum = 0
        max_type = -1
        for type in connection_types:
            if int(self.config['CONNECTION_INTENSITIES'][type]) > maximum:
                maximum = int(self.config['CONNECTION_INTENSITIES'][type])
                max_type = type
        return max_type

    def count_free_time(self, schedule):
        count = 0
        for index, hour in enumerate(schedule):
            if hour[0] == 'freetime':
                count = count + 1
                schedule[index]= (hour[1].split('_')[0], hour[1])
        return count


    def create_random_encounters_from_schedule(self, node, schedule):
        activities = schedule.get_schedule()[self.day]
        #print('id', node[0], ':')
        possible_districts = range(0, )
        potential_random_daily_encounters = self.random_encounter_pool[self.day]
        for hour in range(0, 24):
            activity = activities[hour]

            #print('Activity:', activity[0], activity[1])

            own_district = get_binary_event(1 - self.district_fluctuation)
            pool_id = schedule.get_pool_id()

            if not own_district:
                pool_id = random.choice(self.districts)
            pool = self.get_pool_to_activity(activity=activity[1])
            if pool is 'general':
                potential_random_encounters = potential_random_daily_encounters[hour]['general'][pool_id]
            else:
                potential_random_encounters = potential_random_daily_encounters[hour][pool]
            #print('Potential encounter count:', potential_random_encounters.size)
            connection_mu, connection_sigma = self.get_random_encounter_values(activity=activity, hour=hour)

            connection_count = max(0, min(math.floor(random.gauss(mu=connection_mu, sigma=connection_sigma)), potential_random_encounters.size-len(self.random_encounters[node[0]])))
            #print('Actual Encounter count:', connection_count)
            selection = np.random.choice(potential_random_encounters, size=connection_count, replace=False)

            # selection = random.sample(potential_random_encounters, connection_count)

            for encounter in selection:
                duration = max(random.gauss(mu=0, sigma=2), 1)
                intensity = max(random.gauss(mu=0, sigma=1), 1)
                enc = Encounter(id_a=node[0], id_b=encounter, duration=duration, intensity=intensity, facility_type=activity[1], time=hour)
                self.random_encounters[node[0]].add(enc)
                self.random_encounters[encounter].add(enc)


    def get_pool_to_activity(self, activity):
        if 'gym' in activity or 'shopping' in activity:
            return activity
        else:
            return 'general'


    def get_random_encounter_values(self, hour, activity):
        activity_quotient = int(self.config['RANDOM_ENCOUNTER_MEAN'][activity[0]])
        activity_sigma = int(self.config['RANDOM_ENCOUNTER_SIGMA'][activity[0]])
        return activity_quotient, activity_sigma



    def print_encounters(self):
        print('Printing Encounter List for each node...')
        for node in self.graph:
            enc_list = self.encounters[node[0]]
            print(node[0], enc_list)

    def reset_encounters(self):
        for node in self.graph:
            self.encounters[node[0]] = set()
            self.random_encounters[node[0]] = set()

    def increase_day(self):
        self.day = (self.day + 1) % 7


    def day_loop(self):
        text = ''
        while True:
            #text = input('Press any key to simulate day '+str(self.day)+'\n')
            start = time.time()
            self.create_encounters()
            end = time.time()
            print('Time spent on simulating day', self.day, ':', end-start)
            self.day = (self.day + 1) % 7
            self.reset_encounters()
            gc.collect()
            #text = input('How many nodes encounters should be put out?\n')
            count = -1
            #while count < 0:
                #try:
                   # if text is not '\n':
                  #      count = int(text)
                 #       break
                #except ValueError:
                 #   print('Please input an integer.')
                #    continue
               # if count < 0:
              #      print('Please input a number between 0 and 1000.')
             #   text = input('How many nodes encounters should be put out?\n')

            #for i in range(0, count):
             #   print(self.encounters[self.graph[i][0]])
            #text = input('Type quit to stop the simulation\n')
            #if text == 'quit':
            #    break

    def return_encounters(self):
        converted_encounters = list()
        for node in self.graph:
            encounters = list()
            for index, encounter in enumerate(self.encounters[node[0]]):
                encounters.append(encounter.get_tuple(node[0]))
            converted_encounters.append(encounters)

        return converted_encounters

    def export_encounters(self):

        with open('../resources/encounters.json', 'w') as encounter_output:
            for index in range(0, 100):
                node = self.graph[index]
                # print(node)
                # print(self.encounters[node[0]])
                for index, encounter in enumerate(self.encounters[node[0]]):
                    encounter_output.write(json.dumps(encounter.get_tuple(node[0]), default=str))
                encounter_output.write('\n')