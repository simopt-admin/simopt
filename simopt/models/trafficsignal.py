"""
Summary
-------
Simulate a 2-hour window of a traffic signal model.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/trafficsignal.html>`_.
"""

import numpy as np
from math import ceil
from ..base import Model, Problem
import warnings
import csv
import string


"""
Defines the Road object class
"""
class Road:
    def __init__(self, roadid, startpoint, endpoint, direction):
        self.roadid = roadid
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.direction = direction
        self.queue = list()
        self.status = False
        self.queue_hist = {} # to store queue length at each time point
        self.road_length = 10 # length of this road
        self.overflow = False
        self.overflow_queue = {} # to store queue length of incoming roads at each time point when overflowed
        self.incoming_roads = list()
        self.schedule = list()

    """
    Updates the light from red to green and vice versa
    
    Arguments
    ---------
    schedule: list
        all times where a light changes status    
    t: float
        current time in system
    """
    def update_light(self, schedule, t):
        for time in schedule:
            if time == t:
                if self.status == True:
                    self.status = False
                else:
                    self.status = True
                    if len(self.queue) > 0 and self.queue[0] != 0:
                        self.queue[0].starttime = t
        
"""
Defines the Intersection object class
"""
class Intersection:
    def __init__(self, name, roads):
        self.name = name
        self.schedule = []
        self.horizontalroads = []
        self.verticalroads = []
        self.offset = 0

    """
    Sets specific roads as attributes of the intersection they belong to
    
    Arguments
    ---------
    roads: list
        list of all roads in the system
    """    
    def connect_roads(self, roads, offset):
        for Road in roads:
            if Road.endpoint == self.name:
                direction = Road.direction
                if direction == 'East' or direction == 'West':
                    self.horizontalroads.append(Road)
                    if offset == 0:
                        Road.status = True
                    else:
                        Road.status = False
                else:
                    self.verticalroads.append(Road)
                    if offset == 0:
                        Road.status = False
                    else:
                        Road.status = True

"""
Defines the Car object class
"""        
class Car:
    def __init__(self, carid, arrival, path, visits):
        self.identify = carid
        self.arrival = arrival
        self.initialarrival = arrival
        self.path = path
        self.locationindex = 0
        self.timewaiting = 0
        self.primarrival = arrival
        self.placeInQueue = None
        self.nextstart = None
        self.moving = False
        self.nextSecArrival = None
        self.prevstop = 0
        self.visits = visits
        self.finished = False
        
    def update_location(self):
        self.locationindex += 1

class TrafficLight(Model):
    """
    A model that simulates a series of intersections and their light \
    schedules. As cars travel through the system, their waiting \
    time is tracked.

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI and data validation)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """

    def __init__(self, fixed_factors={}):
        self.name = "TrafficLight"
        self.n_rngs = 3
        self.n_responses = 1
        self.specifications = {
            "lambdas": {
                "description": "Rate parameter of the time interval distribution, in seconds, for generating each car.",
                "datatype": list,
                "default": [2, 2, 0, 1, 2, 2, 0, 1]
            },
            
            "runtime": {
                "description": "The number of seconds that the traffic model runs", 
                "datatype": float,
                "default": 100
            },

            "numintersections": {
                "description": "The number of intersections in the traffic model",
                "datatype": int,
                "default": 4
            },
            
            "decision_vector": {
                "description": "Delay, in seconds, in light schedule based on distance from first intersection",
                "datatype": list,
                "default": [1, 2, 3]
            },
            

            "speed": {
                "description": "Constant speed in meter/second for the cars",
                "datatype": float,
                "default": 5
            },

            "carlength": {
                "description": "Length in meters of each car",
                "datatype": float,
                "default": 4.5
            },
                    
            "reaction": {
                "description": "Reaction time in seconds of cars in queue",
                "datatype": float,
                "default": 0.1
            },

            "transition_probs": {
                "description": "The transition probability of a car end at each point from their current starting point.",
                "datatype": list,
                "default": [0.7, 0.3, 0.3, 0.2, 0.25, 0.1, 0.15]
            },
            
            "pause":{
                "description": "The pause in seconds before move on a green light",
                "datatype": float,
                "default" : 0.1
            },
            
            "car_distance" : {
                "description": "The distance between cars",
                "datatype": float,
                "default" : 0.5
            },
            
            "length_arteries" : {
                "description": "The length in meters of artery roads", 
                "datatype": float,
                "default" : 100
            },
            
            "length_veins" : {
                "description": "The length in meters of vein road", 
                "datatype": float,
                "default" : 100
            },
            
            "redlight_arteries" : {
                "description": "The length of redlight duration of artery roads in each intersection", 
                "datatype": list,
                "default" : [10, 10, 10, 10]
            },
            
            "redlight_veins" : {
                "description": "The length of redlight duration of vein roads in each intersection", 
                "datatype": list,
                "default" : [20, 20, 20, 20]
            }
        }

        self.check_factor_list = {
            "lambdas": self.check_lambdas,
            "numintersections": self.check_numintersections,
            "decision_vector": self.check_decision_vector,
            "speed": self.check_speed,
            "carlength": self.check_carlength,
            "reaction": self.check_reaction,
            "transition_probs": self.check_transition_probs,
            "pause": self.check_pause,
            "car_distance": self.check_car_distance,
            "length_arteries": self.check_length_arteries,
            "length_veins": self.check_length_veins,
            "redlight_arteries": self.check_redlight_arteries,
            "redlight_veins": self.check_redlight_veins
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    def check_lambdas(self):
        return (max(self.factors['lambdas'][3], self.factors['lambdas'][7]) <= min([self.factors['lambdas'][i] for i in [0, 1, 4, 5]]))
    
    def check_runtime(self):
        return self.factors["runtime"] > 0

    def check_numintersections(self):
        return self.factors["numintersections"] > 0

    def check_decision_vector(self):
        for value in self.factors["decision_vector"]:
            if value < 0:
                return False
            else:
                return True

    def check_speed(self):
        return self.factors["speed"] > 0
    
    def check_carlength(self):
        return self.factors["carlength"] > 0
    
    def check_reaction(self):
        return self.factors["reaction"] > 0

    def check_transition_probs(self):
        if any([transition_prob < 0 for transition_prob in self.factors["transition_probs"]]):
            return False
        p16, p17, p21, p23, p41, p43, p47 = self.factors["transition_probs"]
        transition_matrix =  np.array([
                            [0  , 0 , 0  , 0  , 0  , p16, p17, 0],
                            [p21, 0 , p23, 0  , p21, 0  , p23, 0],
                            [0  , 0 , 0  , 0  , 0  , 0  , 0  , 0],
                            [p41, p41, p43, 0 , p41, 0  , p47, 0],
                            [0  , p16, p17, 0 , 0  , 0  , 0  , 0],
                            [p21, 0  , p23, 0 , p21, 0  , p23, 0],
                            [0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
                            [p41, 0  , p47, 0  , p41, p41, p43, 0]])
        prob_sum = np.sum(transition_matrix[:, :], axis=1).tolist()
        del prob_sum[2]
        del prob_sum[5]
        return all([x == 1 for x in prob_sum]) 
    
    def check_pause(self):
        return self.factors["pause"] > 0
    
    def check_car_distance(self):
        return self.factors["car_distance"] > 0
    
    def check_length_arteries(self):
        return self.factors["length_arteries"] > 0
    
    def check_length_veins(self):
        return self.factors["length_veins"] > 0
    
    def check_redlight_arteries(self):
        return self.factors["redlight_arteries"] > 0
    
    def check_redlight_veins(self):
        return self.factors["redlight_veins"] > 0
    
    def check_simulatable_factors(self):                                   
        return True
       
    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "WaitingTime" = average time waiting at a light
        gradients : dict of dicts
            gradient estimates for each response
        """
        
        #Designate separate RNGs for start, end positions and interarrival times
        start_rng = rng_list[0]
        end_rng = rng_list[1]
        arrival_rng = rng_list[2]

        #Initializes variables to start the simulation
        t = 0
        nextcargen = 0
        outbounds = self.factors["runtime"] + 1
        carSimIndex = 0
        nextStart = outbounds
        nextSecArrival = outbounds
        minPrimArrival = 0
        start_prob = [x/sum(self.factors["lambdas"]) for x in self.factors["lambdas"]]
        
        
        #offset of 12 roads
        self.factors["offset"] = [0, 
                                  self.factors["decision_vector"][0], 
                                  self.factors["redlight_veins"][0],
                                  self.factors["decision_vector"][0]+self.factors["redlight_veins"][1],
                                  0,
                                  self.factors["decision_vector"][0],
                                  self.factors["decision_vector"][1],
                                  self.factors["decision_vector"][2],
                                  self.factors["decision_vector"][1]+self.factors["redlight_veins"][2],
                                  self.factors["decision_vector"][2]+self.factors["redlight_veins"][3],
                                  self.factors["decision_vector"][1],
                                  self.factors["decision_vector"][2]]
        print("offset:", self.factors["offset"])
        
        #transition matrix
        p16, p17, p21, p23, p41, p43, p47 = self.factors["transition_probs"]
        transition_matrix =  np.array([
                            [0  , 0 , 0  , 0  , 0  , p16, p17, 0],
                            [p21, 0 , p23, 0  , p21, 0  , p23, 0],
                            [0  , 0 , 0  , 0  , 0  , 0  , 0  , 0],
                            [p41, p41, p43, 0 , p41, 0  , p47, 0],
                            [0  , p16, p17, 0 , 0  , 0  , 0  , 0],
                            [p21, 0  , p23, 0 , p21, 0  , p23, 0],
                            [0  , 0  , 0  , 0  , 0  , 0  , 0  , 0],
                            [p41, 0  , p47, 0  , p41, p41, p43, 0]])
        
        
        
        #Draw out map of all locations in system
        roadmap = np.array([
                ['','N1','N2', ''], 
                ['W1', 'A', 'B', 'E1'], 
                ['W2', 'C', 'D', 'E2'],
                ['', 'S1', 'S2', '']])
        #List each location and the locations that are next accessible   
        graph = {'A': ['N1', 'B', 'C'],
                'B': ['N2', 'E1', 'D'],
                'C': ['A', 'S1', 'W2'],
                'D': ['C', 'B', 'S2'],
                'N1': ['A'],
                'N2': ['B'],
                'E1': [],
                'E2': ['D'],
                'S2': ['D'],
                'S1': ['C'],
                'W2': [],
                'W1': ['A'],
                }

        
        #Lists each location in the system
        points = list(graph.keys())
        
        """
        Find the shortest path between two points # NO lEFT TURN

        Arguments
        ---------
        graph: dictionary
            dictionary with all locations and their connections
        start: string
            name of starting location
        end: string
            name of ending location
        path: list
            list of locations that represent the car's path            

        Returns
        -------
        shortest: list
            list of locations that represent the shortest path from 
            start to finish
        """
        def find_shortest_path(graph, start, end, path=[]):
            path = path + [start]
            #print("current path", path)
            #Path starts and ends at the same point
            if start == end:
                return path
                
            shortest = None
            for node in graph[start]:
                #print("node ", node)
                #if node not in path:
                if sum(x == node for x in path) < 2:
                    if len(path) >= 2: # no left turn
                        direction1 = find_direction(path[-2], path[-1], roadmap)
                        direction2 = find_direction(path[-1], node, roadmap)
                        turn = find_turn(direction1+direction2)
                        #print(turn)
                        if (turn in ['Right', 'Straight']):
                            newpath = find_shortest_path(graph, node, end, path)
                            if newpath:
                                if not shortest or len(newpath) < len(shortest):
                                    shortest = newpath
                    else:
                        newpath = find_shortest_path(graph, node, end, path)
                        if newpath:
                            if not shortest or len(newpath) < len(shortest):
                                shortest = newpath
            return shortest

        """
        Generates shortest path through two random start and end locations

        Returns
        -------
        path: list
            list of locations that car visits
        """
        def generate_path(start):
            path = None
            while path is None:
                end = end_rng.choices(population=range(8), weights=transition_matrix[start])[0]
                path = find_shortest_path(graph,
                                          points[start+self.factors["numintersections"]],
                                          points[end+self.factors["numintersections"]])
            return path

        """
        Takes in road and finds its direction based on the map

        Arguments
        ---------
        start: string
            name of starting location
        end: string
            name of ending location
        roadmap: array
            array of all points in system          

        Returns
        -------
        direction: string
            direction that the road is facing
        """
        def find_direction(start, end, roadmap):
            yloc1, xloc1 = np.where(roadmap == start)
            yloc2, xloc2 = np.where(roadmap == end)
            if xloc1 > xloc2:
                direction = 'West'
            elif xloc1 < xloc2:
                direction = 'East'
            elif yloc1 > yloc2:
                direction = 'North'
            else:
                direction = 'South'
            return direction
        
        """
        Assigns the direction of a turn when given two roads

        Arguments
        ---------
        roadcombo: string
            combined directions of roads           

        Returns
        -------
        turn: string
            direction of turn
        """
        def find_turn(roadcombo):
            turnkey = {'Straight': ['WestWest', 'EastEast', 'SouthSouth', 'NorthNorth'], 
                    'Left': ['NorthWest', 'EastNorth', 'SouthEast', 'WestSouth'], 
                    'Right': ['NorthEast', 'WestNorth', 'SouthWest', 'EastSouth'],
                       'Uturn': ['NorthSouth', 'SouthNorth', 'EastWest', 'WestEast']
                    }
            turn = ''
            for key, values in turnkey.items():
                for value in values:
                    if roadcombo == value:
                        turn = key
            return turn
        
        
        road_pair = [
            ("N1", "A"), ("N2", "B"), ("W1", "A"), ("A", "B"),
            ("C",  "A"), ("D",  "B"), ("A",  "C"), ("B", "D"),
            ("D",  "C"), ("E2", "D"), ("S1", "C"), ("S2", "D"),
            ("A", "N1"), ("B", "N2"), ("B", "E1"), ("C", "W2"),
            ("C", "S1"), ("D", "S2")]
        
        #Generates list of all road objects in the system
        roads = list()    
        roadid = 0
        for (key, value) in road_pair:
                direction = find_direction(key, value, roadmap)
                roads.append(Road(roadid, key, value, direction))
                roads[roadid].nextchange = outbounds
                if direction == 'West' or direction == 'East':
                    roads[roadid].road_length = self.factors["length_veins"]
                else:
                    roads[roadid].road_length = self.factors["length_arteries"]
                roads[roadid].queue.append(0)
                # print('Road', roadid, ':', key, value, direction)
                roadid += 1
        
        # add incoming roads
        # only when startpoints are A, B, C, D have the overflow with overflow queue length issue
        start_points = np.array([key for (key, value) in road_pair])
        for road in roads:
            endpoint = road.endpoint
            if endpoint in {"A", "B", "C", "D"}:
                indices = np.where(endpoint == start_points)[0]
                for i in indices:
                    if i < 12:
                        turn = find_turn(road.direction+roads[i].direction)
                        if turn != 'Left':
                            roads[i].incoming_roads.append(road)

        # check incoming roads
        for road in roads:
            if len(road.incoming_roads) > 0:
                print(road.roadid, " incoming roads:", [r.roadid for r in road.incoming_roads])
        """
        Finds the roads that a car will take on its path 

        Arguments
        ---------
        visits: list
            all locations in a car's path        

        Returns
        -------
        path: list
            list of road objects that the car travels on
        """  
        def find_roads(visits):
            path = list()
            for i in range(len(visits) - 1):
                    for road in roads:
                        if road.startpoint == visits[i] and road.endpoint == visits[i + 1]:
                            path.append(road)
            return path         
           
        
        #Generates list of all intersection objects
        intersections = list()
        for i in range(self.factors["numintersections"]):
            location = points[i]
            intersections.append(Intersection(location, roads))
            if i == 0:
                offset = 0
            else:
                offset = self.factors["decision_vector"][i-1]
            #print('Intersection index', i)
            #schedule = gen_lightschedule(self.factors["interval"], intersections[i], i)
            #intersections[i].schedule = schedule
            #print('Intersection', intersections[i].name, ':',  schedule)
            intersections[i].connect_roads(roads, offset)
        
        """
        Generates light schedule of road 
        """ 
        
        greenlight_arteries = self.factors["redlight_veins"]
        greenlight_veins = self.factors["redlight_arteries"]        
        
        
        for roadid in range(12):
            offset = self.factors["offset"][roadid]
            road = roads[roadid]
            ind = ["A", "B", "C", "D"].index(road.endpoint)
            if road.direction == 'West' or road.direction == 'East':
                interval = [self.factors["redlight_veins"][ind], greenlight_veins[ind]]
            else:
                interval = [greenlight_arteries[ind], self.factors["redlight_arteries"][ind]]
            #print(roadid, road.direction, road.startpoint, road.endpoint, ind, interval)
            for i in range(ceil(self.factors["runtime"] / min(interval)) + 2):
                if i == 0:
                    road.schedule.append(0)
                elif i %2 == 0: #even time index, A: red -> green, V: green -> red
                        road.schedule.append(road.schedule[-1] +interval[1])
                else: #odd time index, A: green -> red, V: red -> green 
                    if i == 1:
                        offsetcalc = (offset % interval[0])
                        if offsetcalc == 0:
                            offsetcalc = interval[0]
                        road.schedule.append(road.schedule[-1] + offsetcalc)
                    else:
                        road.schedule.append(road.schedule[-1] + interval[0])
            print("road", roadid, " schedule: ", road.schedule)
        
        
        
        """
        Finds the next time any intersection light will change signal   

        Arguments
        ---------
        intersections: list
            list of all intersection objects
        t: float
            current time in system

        Returns
        -------
        mintimechange: float
            time that the next light changes
        location: list
            list of locations that a light changes at
        """    
        def find_nextlightchange_road(roads, t):
            mintimechange = self.factors["runtime"]
            #Loops through roads to find a minimum light changing time
            for road in roads[:12]:
                nextchange = min([i for i in road.schedule if i > t])
                if nextchange <= mintimechange:
                    mintimechange = nextchange
            return mintimechange
        
        """
        Updates the intersections with their new light status        

        Arguments
        ---------
        t: float
            current time in system
        intersections: list
            list of all intersection objects
        """      
        def update_road_lights(t, roads):
            if t == 0:
                nextlightlocation = roads[:12]
            else:
                nextlightlocation = []
                for road in roads[:12]:
                    if t in road.schedule:
                        nextlightlocation.append(road)
            for road in nextlightlocation:
                #print('Road', road.roadid, 'was before:', road.status)
                road.update_light(road.schedule, t)
                road.nextchange = min(i for i in road.schedule if i > t)
            status = list()
            nextc = list()
            for road in roads[:12]:
                if road.status == True:
                    status.append("Green")
                else:
                    status.append("Red")
                nextc.append(road.nextchange)
            print("Time:", t,' Road status: ', status, ' next change: ',nextc)
        
        
        """
        Generates list of all car objects as they are created  

        Arguments
        ---------
        initialarrival: float
            time that a car is introduced to the system
        """
        cars = list()
        def gen_car(t):
            # choose start point
            start = start_rng.choices(population=range(8), weights=start_prob)[0]
            initialarrival = t + arrival_rng.expovariate(self.factors["lambdas"][start])
            visits = generate_path(start)
            while visits == None or len(visits) == 1:
                visits = generate_path(start)
            identify = len(cars)
            path = find_roads(visits)
            cars.append(Car(identify, initialarrival, path, visits))
            cars[identify].nextstart = outbounds
            cars[identify].nextSecArrival = outbounds
            # print('Car', identify, ':', visits)
            return initialarrival
        
        
        """
        Finds a car's place in queue and assigns it a new start time      

        Arguments
        ---------
        car: Car
            car object
        raod: Road
            road object
        t: float
            current time in system
        """       
        def find_place_in_queue(car, road, t):
            queueindex = len(road.queue) - 1
            while road.queue[queueindex] == 0 and queueindex > 0:
                queueindex -= 1
            #Car is not the first in its queue
            if queueindex != 0 or road.queue[0] != 0: 
                #Car is second in queue
                if len(road.queue) == queueindex + 1:
                    road.queue.append(car)
                #Car is third or later in queue   duplicated with at first   
                #else:
                #    road.queue[queueindex + 1] = car
                car.placeInQueue = queueindex + 1
                car.nextstart = road.queue[queueindex].nextstart + self.factors["reaction"]
            #Car is the first in its queue
            else: 
                road.queue[queueindex] = car
                car.placeInQueue = queueindex
                #Car is at the end of its path
                if car.locationindex == len(car.path) - 1:
                    car.nextstart = outbounds
                    car.nextSecArrival = outbounds
                #Car still has a road to travel to
                else:
                    #Light is green on the road that the car is on
                    if road.status == True:
                        car.nextstart = t
                    #Light is red on the road that the car is on
                    else:    
                        car.nextstart = road.nextchange
            
        #Lights are turned on and the first car is created
        update_road_lights(0, roads)
        gen_car(nextcargen)
        currentcar = cars[carSimIndex]
        movingcar = cars[0]
        arrivingcar = movingcar
        current_finished = -1
        #Loops through time until runtime is reached
        sumwait = 0
        finishedcars = 0
        cars_wait = {}
        cars_total = {}
        Overflow_total = {}
        Overflow_len_total = {}
        with open("./Cars_detail.csv", mode="w", newline="") as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Print headers.
            output_file.write("Cars,Action,Position,Road,Time\n")
            while t < self.factors["runtime"]:
                #Assigns the next time a light changes
                nextLightTime = find_nextlightchange_road(roads, t)
                #The next event is a car being introduced to the system
                if min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == minPrimArrival:
                    t = minPrimArrival
                    for i in range(12, 18):
                        roads[i].nextchange = t
                    cars[carSimIndex].prevstop = t
                    # print('Car', cars[carSimIndex].identify, 'is arriving first at time', t)
                    action = "Start"
                    csv_writer.writerow([cars[carSimIndex].identify]+[action]+[cars[carSimIndex].visits[cars[carSimIndex].locationindex]]+[cars[carSimIndex].path[cars[carSimIndex].locationindex].roadid]+[t])
                    #A new car is generated
                    nextcargen = gen_car(t)
                    minPrimArrival = nextcargen
                    carSimIndex += 1
                    
                
                    #The arriving car arrives into the system based on its path
                    currentcar = cars[carSimIndex - 1]
                    initroad = currentcar.path[currentcar.locationindex]
                    find_place_in_queue(currentcar, initroad, t)
                       
                #The next event is a light changing    
                elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextLightTime:
                    t = nextLightTime
                    for i in range(12, 18):
                        roads[i].nextchange = t
                    #Roads that change lights at this time are updated
                    update_road_lights(t, roads)
                         
                #The next event is a car starting to move            
                elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextStart:
                    t = nextStart
                    for i in range(12, 18):
                        roads[i].nextchange = t
                    
                    # print('Time:', t, 'Car', movingcar.identify, 'is starting from road', movingcar.path[movingcar.locationindex].roadid, 'at spot', movingcar.placeInQueue)
                    action = "Leave"
                    csv_writer.writerow([movingcar.identify]+[action]+[movingcar.visits[movingcar.locationindex]]+[movingcar.path[movingcar.locationindex].roadid]+[t])
                    #Car is the first in its queue
                    if movingcar.placeInQueue == 0:
                        #Car's next arrival is set
                        #movingcar.nextSecArrival = t + (self.factors["distance"] / self.factors["speed"])
                        # change the distance to by road
                        print("Time:", t, "Car", movingcar.identify, 'is starting from road', movingcar.path[movingcar.locationindex].roadid)
                        movingcar.nextSecArrival = t + self.factors["pause"] +(movingcar.path[movingcar.locationindex].road_length / self.factors["speed"])
                         
                    #Car is not the first in its queue
                    else:
                        #Car's next arrival time is set
                        movingcar.nextSecArrival = t + (self.factors['car_distance']+ self.factors["carlength"] / self.factors["speed"])
                    
                    #Car leaves its current queue and is 'moving'
                    movingcar.path[movingcar.locationindex].queue[movingcar.placeInQueue] = 0
                    movingcar.moving = True
                    movingcar.timewaiting = movingcar.timewaiting + (t - movingcar.prevstop)
                    movingcar.nextstart = outbounds
                    nextStart = outbounds
               
                #The next event is a car arriving within the system 
                elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextSecArrival:
                    t = nextSecArrival
                    for i in range(12, 18):
                        roads[i].nextchange = t
                
                    #Car is first in its queue 
                    if arrivingcar.placeInQueue == 0:
                        #Car changes the road it is traveling on 
                        arrivingcar.update_location()
                        currentroad = arrivingcar.path[arrivingcar.locationindex]
                        #Car is assigned its location and given a new start time
                        find_place_in_queue(arrivingcar, currentroad, t)
                    #Car is not the first in its queue
                    else:
                        #Car moves up in its queue
                        currentroad = arrivingcar.path[arrivingcar.locationindex]
                        currentroad.queue[arrivingcar.placeInQueue] = 0
                        currentroad.queue[arrivingcar.placeInQueue - 1] = arrivingcar
                        arrivingcar.placeInQueue -= 1
                        #Current road has a green light
                        if currentroad.status == True:
                            arrivingcar.nextstart = t
                        #Current road has a red light
                        else:
                            arrivingcar.nextstart = currentroad.nextchange
                    print('Time:', t, 'Car', arrivingcar.identify, 'is arriving at road', arrivingcar.path[arrivingcar.locationindex].roadid, 'at queue position', arrivingcar.placeInQueue)
                    action = 'Arrival'
                    csv_writer.writerow([arrivingcar.identify]+[action]+[arrivingcar.visits[arrivingcar.locationindex]]+[arrivingcar.path[arrivingcar.locationindex].roadid]+[t])

                    #Car is no longer 'moving'
                    movingcar.moving = False
                    arrivingcar.nextSecArrival = outbounds
                    nextSecArrival = outbounds
                    arrivingcar.prevstop = t
                
                carindex = 0
                minSecArrival = outbounds
                minStart = outbounds
                #Finds the next car to start moving and the next car to arrive
                while carindex < len(cars) - 1:
                    testcar = cars[carindex]
                    #Car is elligible to be the next starting car
                    if min(nextStart, testcar.nextstart) == testcar.nextstart and testcar.nextstart != outbounds:
                        minStart = testcar.nextstart
                        movingcar = testcar
                    #Car is elligible to be the next arriving car
                    if min(minSecArrival, testcar.nextSecArrival) == testcar.nextSecArrival and testcar.nextSecArrival != outbounds:
                        minSecArrival = testcar.nextSecArrival
                        arrivingcar = testcar
                    #Next car is tested and the next events are set
                    carindex += 1
                    nextSecArrival = minSecArrival
                    nextStart = minStart
            
                # sumwait = 0
                # finishedcars = 0
                for car in cars:
                    if (car.locationindex == len(car.path) - 1) and (car.finished == False): # car arrived
                        # print('Car', car.identify, 'waiting time:', car.timewaiting)
                        action = "Finish"
                        csv_writer.writerow([car.identify]+[action]+[car.visits[car.locationindex]]+[car.path[car.locationindex].roadid]+[t])
                        car.finished = True
                        cars_wait[car.identify] = car.timewaiting
                        cars_total[car.identify] = t - car.initialarrival
                        # created a list to store which car has arrived. And compare current car identify with the list
                        # if current car.identity < identities in the list, then print A WRANING MESSAGE AND STOP.
                        # if car.identify >= current_finished:
                        #     current_finished = car.identify
                        # else:
                        #     print(' WARN!! Car ', car.identify, ' finished later than Car', current_finished)
                        #     return 
                        sumwait += car.timewaiting
                        finishedcars += 1
    # in the first several interations, there is no finishedcars
                if finishedcars > 0:
                    avgwait = sumwait /  finishedcars
                    # print('Finished cars', finishedcars, ' Average waiting time:', avgwait)
                    # when all finished compute average total time
                    avgtotal = sum(cars_total.values())/len(cars_total)
                else:
                    avgwait = 0
                    avgtotal = 0
                    
                # record queue length of each road and update overflow status
                Overflow_total[t] = 0
                for roadid in range(12):
                    # num of cars in queue
                    cars_in_queue = sum([x != 0 for x in roads[roadid].queue])
                    #queue length = (car_length+ car_distance) * num(cars in queue) - car_distance
                    roads[roadid].queue_hist[t] = max(0, cars_in_queue*(self.factors['car_distance']+ self.factors["carlength"])-self.factors['car_distance'])
                    # if there is overflow status
                    if (roads[roadid].road_length <= roads[roadid].queue_hist[t]):
                        roads[roadid].overflow = True
                        #print("Overflowed road: ", roadid, "at time:", t)
                        Overflow_total[t] = 1
                    else:
                        roads[roadid].overflow = False
                
                # after queue len is calculated, deal with overflow
                # when a road is overflowed, 1. cars in incoming roads cannot get in; 2. calculated oveflow len
                current_overflow_len = list()
                for roadid in range(12):
                    if roads[roadid].overflow == True:
                        # add incoming roads queue length
                        if len(roads[roadid].incoming_roads) > 0:
                            # calculate oeverflow queue lenth
                            overflow_queue = [r.queue_hist[t] for r in roads[roadid].incoming_roads]
                            roads[roadid].overflow_queue[t] = sum(overflow_queue)
                            current_overflow_len.append(sum(overflow_queue))
                            # update start time of cars in queue of incoming roads
                            # start time of the last car in overflowed road
                            last_start = max([x.nextstart for x in roads[roadid].queue if x != 0])
                            for r in roads[roadid].incoming_roads:
                                # if there are cars in queue
                                for car_q in r.queue:
                                    last_start = last_start+self.factors["reaction"]
                                    if car_q != 0:
                                        # first car in queue in the incoming road
                                        if car_q.placeInQueue == 0:
                                            car_q.nextstart = max(car_q.nextstart, last_start)
                                        # second and later car in queue in the incoming road
                                        else:
                                            car_q.nextstart = max(car_q.nextstart, last_start)
                if(len(current_overflow_len) > 0):
                    Overflow_len_total[t] = sum(current_overflow_len)/len(current_overflow_len)
                else:
                    Overflow_len_total[t] = 0 
                
                                    
        # compute ave queue length for each road
        avg_queue_length = 0
        OverflowPercentage = []
        OverflowAveLen = []
        for roadid in range(12):
            # use dictionary to get list of t and queue_len
            # (t - t-1)*queue_len/max(t)
            queue_len = list(roads[roadid].queue_hist.values())
            queue_time = list(roads[roadid].queue_hist.keys())
            time_dif = [queue_time[i + 1] - queue_time[i] for i in range(len(queue_time)-1)]
            avg_queue = sum([x * y for x, y in zip(queue_len[:-1], time_dif)])/t
            avg_queue_length = avg_queue_length+avg_queue
            #overflow queue time of each road
            overflow_ind = [1*(q >= roads[roadid].road_length) for q in queue_len]
            #Overflow_total.append(overflow_ind[:-1])
            overflow_duration = sum([x * y for x, y in zip(overflow_ind[:-1], time_dif)])
            overflow_perc = overflow_duration/t*100
            OverflowPercentage.append(overflow_perc)
            # overflow queue length
            # never overflow
            if overflow_duration == 0:
                overflow_len_dur = 0
            else:
                overflow_len = list(roads[roadid].overflow_queue.values())
                overflow_len_sum = sum([x * y for x, y in zip(overflow_len[:-1], time_dif)])
                overflow_len_dur = overflow_len_sum/overflow_duration
            OverflowAveLen.append(overflow_len_dur)
        
        # total overflow index
        Overflow_total_ind = list(Overflow_total.values())
        Overflow_total_time = list(Overflow_total.keys())
        Overflow_total_len = list(Overflow_len_total.values())
        time_dif = [Overflow_total_time[i + 1] - Overflow_total_time[i] for i in range(len(Overflow_total_time)-1)]
        overflow_system_duration = sum([x * y for x, y in zip(Overflow_total_ind[:-1], time_dif)])
        overflow_system_perc = overflow_system_duration/t*100
        if overflow_system_perc < 51:
            overflow_system_perc_over_51 = False
        else:
            overflow_system_perc_over_51 = True
        #overflow queue length
        if overflow_system_duration == 0:
            OverflowAveLen_system = 0
        else:
            overflow_system_len = sum([x * y for x, y in zip(Overflow_total_len[:-1], time_dif)])
            OverflowAveLen_system = overflow_system_len/overflow_system_duration
        
        
        # average queue
        avg_queue_length = avg_queue_length/12
        
        # how to calculate system overflow len
        
        with open('SummaryTime.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = ['Cars_id', 'WaitingTime', 'TotalTime'])
            writer.writeheader()
            #writer.writerows(cars_wait)
            for key in cars_wait.keys():
                csvfile.write("%s,%s,%s\n"%(key,cars_wait[key],cars_total[key]))
        # Compose responses and gradients.
        responses = {"WaitingTime": avgwait, "SystemTime": avgtotal, "AvgQueueLen": avg_queue_length,
                    "OverflowPercentage":overflow_system_perc, "OverflowPercentageOver51":overflow_system_perc_over_51,
                    "OverflowAveLen":OverflowAveLen_system}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients
            

class MinWaitingTime(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : list
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """

    def __init__(self, name="TRAFFICCONTROL-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None  
        self.model_default_factors = {
            "runtime": 50
            }
        self.model_decision_factors = {"decision_vector"}      
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (1, 1, 1)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 100
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = TrafficLight(self.model_fixed_factors)
        self.lower_bounds = (0,)*3
        self.upper_bounds = (min(self.model.factors["redlight_arteries"]+self.model.factors["redlight_veins"]),)*3
    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "decision_vector": vector                               
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["decision_vector"])                               #change this to decision variable
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["WaitingTime"],)       
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,)*3,)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic
            constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        box_feasible = super().check_deterministic_constraints(x)



    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([rand_sol_rng.uniform(0, min(self.model.factors["redlight_arteries"]+self.model.factors["redlight_veins"])) for _ in range(self.dim)])
        return x