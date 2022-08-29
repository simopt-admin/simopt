# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:51:30 2022

@author: wesle
"""
import numpy as np
import math
from base import Model, Problem

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
    def connect_roads(self, roads):
        for Road in roads:
            if Road.endpoint == self.name:
                direction = Road.direction
                if direction == 'East' or direction == 'West':
                    self.horizontalroads.append(Road)
                    Road.status = True                    
                else:
                    self.verticalroads.append(Road)
                    Road.status = False

"""
Defines the Car object class
"""        
class Car:
    def __init__(self, carid, arrival, path):
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
        self.name = "TRAFFICLIGHT"
        self.n_rngs = 1
        self.n_responses = 1
        self.specifications = {
            "lambda": {
                "description": "Rate parameter of interarrival \
                                time distribution.",
                "datatype": float,
                "default": 0.5
            },

            "runtime": {
                "description": "Total time that the simulation runs", 
                "datatype": float,
                "default": 50
            },

            "numintersections": {
                "description": "Number of intersections",
                "datatype": int,
                "default": 4
            },

            "interval": {
                "description": "Interval between light changes",
                "datatype": float,
                "default": 5
            },

            "offset": {
                "description": "Delay in light schedule based on distance from first intersection",
                "datatype": list,
                "default": [0, 0, 0, 0]
            },

            "speed": {
                "description": "Constant speed for the cars",
                "datatype": float,
                "default": 2.5
            },

            "distance": {
                "description": "Distance of travel between roads",
                "datatype": float,
                "default": 5
            },

            "carlength": {
                "description": "Length of each car",
                "datatype": float,
                "default": 1
            },
                    
            "reaction": {
                "description": "Reaction time of cars in queue",
                "datatype": float,
                "default": 0.1
            }
        }

        self.check_factor_list = {
            "lambda": self.check_lambda,
            "runtime": self.check_runtime,
            "numintersections": self.check_numintersections,
            "interval": self.check_interval,
            "offset": self.check_offset,
            "speed": self.check_speed,
            "distance": self.check_distance,
            "carlength": self.check_carlength,
            "reaction": self.check_reaction
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    def check_lambda(self):
        return self.factors["lambda"] > 0

    def check_runtime(self):
        return self.factors["runtime"] > 0

    def check_numintersections(self):
        return self.factors["numintersections"] > 0

    def check_interval(self):
        return self.factors["interval"] > 0

    def check_offset(self):
        for value in self.factors["offset"]:
            if value < 0:
                return False
            else:
                return True

    def check_speed(self):
        return self.factors["speed"] > 0
    
    def check_distance(self):
        return self.factors["distance"] > 0
    
    def check_carlength(self):
        return self.factors["carlength"] > 0
    
    def check_reaction(self):
        return self.factors["reaction"] > 0

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
        outbounds = self.factors["T"] + 1
        carSimIndex = 0
        nextStart = outbounds
        nextSecArrival = outbounds
        minPrimArrival = 0

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
                'S1': ['C'],
                'S2': ['D'],
                'W1': ['A'],
                'W2': []
                }
        #Lists each location in the system
        points = list(graph.keys())

        """
        Find the shortest path between two points

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
            #Path starts and ends at the same point
            if start == end:
                return path
                
            shortest = None
            for node in graph[start]:
                if node not in path:
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
        def generate_path():
            start = start_rng.randint(self.factors["numintersections"], 11)
            end = end_rng.randint(self.factors["numintersections"], 11)
            path = find_shortest_path(graph, points[start], points[end])
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
                    'Right': ['NorthEast', 'WestNorth', 'SouthWest', 'EastSouth']
                    }
            turn = ''
            for key, values in turnkey.items():
                for value in values:
                    if roadcombo == value:
                        turn = key
            return turn
        
        #Generates list of all road objects in the system
        roads = list()    
        roadid = 0
        for key, value in graph.items():
            for value in graph[key]:
                direction = find_direction(key, value, roadmap)
                roads.append(Road(roadid, key, value, direction))
                roads[roadid].queue.append(0)
                print('Road', roadid, ':', key, value, direction)
                roadid += 1
                
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
        
        """
        Generates light schedule based on a given interval

        Arguments
        ---------
        interval: float
            time between light changes
        intersection: Intersection
            intersection object
        index: int
            index that links an intersection to an offset value

        Returns
        -------
        schedule: list
            times that a light changes
        """ 
        def gen_lightschedule(interval, intersection, index):
            schedule = list()
            '''
            startrow = np.array(np.where(roadmap == 'A'))[0]
            startcol = np.array(np.where(roadmap == 'A'))[1]
            endrow = np.array(np.where(roadmap == location))[0]
            endcol = np.array(np.where(roadmap == location))[1]
            distance = int((endrow - startrow) + (endcol - startcol))
            '''
            intersection.offset = self.factors["offset"][index]
            offsetcalc = (intersection.offset % interval)
            if offsetcalc == 0:
                offsetcalc = interval
            for i in range(math.ceil(self.factors["T"] / interval) + 1):
                if i == 0:
                    schedule.append(0)
                else:
                    schedule.append(offsetcalc + (interval * (i -1)))
            return schedule
        
        
        #Generates list of all intersection objects
        intersections = list()
        for i in range(self.factors["numintersections"]):
            location = points[i]
            intersections.append(Intersection(location, roads))
            schedule = gen_lightschedule(self.factors["interval"], intersections[i], i)
            intersections[i].schedule = schedule
            print('Intersection', intersections[i].name, ':',  schedule)
            intersections[i].connect_roads(roads)
        
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
        def find_nextlightchange(intersections, t):
            mintimechange = self.factors["T"]
            location = []
            #Loops through intersections to find a minimum light changing time
            for intersection in intersections:
                index = 0
                while index < len(intersection.schedule) and t >= intersection.schedule[index]:
                    index += 1
                if index != len(intersection.schedule) and intersection.schedule[index] <= mintimechange:
                    mintimechange = intersection.schedule[index]
                    location.append(intersection)
            #Loops through chosen intersections and updates their roads
            for intersection in location:
                for road in intersection.horizontalroads:
                    road.nextstart = mintimechange
                for road in intersection.verticalroads:
                    road.nextstart = mintimechange
                
            return mintimechange, location
        
        """
        Updates the intersections with their new light status        

        Arguments
        ---------
        t: float
            current time in system
        intersections: list
            list of all intersection objects
        """      
        def update_intersections(t, intersections):    
            print("I AM CHANGING A LIGHT AT TIME:", t)
            if t == 0:
                nextlightlocation = intersections
            else:
                nextlightlocation = []
                for intersection in intersections:
                    if t in intersection.schedule:
                        nextlightlocation.append(intersection)
            for intersection in nextlightlocation:
                for road in intersection.horizontalroads:
                    road.update_light(intersection.schedule, t)
                    print('Road', road.roadid, 'is now:', road.status)
                    road.nextchange = t + self.factors["interval"]
                for road in intersection.verticalroads:
                    road.update_light(intersection.schedule, t)
                    print('Road', road.roadid, 'is now:', road.status)
                    road.nextchange = t + self.factors["interval"]
        
        """
        Generates list of all car objects as they are created  

        Arguments
        ---------
        initialarrival: float
            time that a car is introduced to the system
        """
        cars = list()
        def gen_car(initialarrival):
            visits = generate_path()
            while visits == None or len(visits) == 1:
                visits = generate_path()
            identify = len(cars)
            path = find_roads(visits)
            cars.append(Car(identify, initialarrival, path))
            cars[identify].nextstart = outbounds
            cars[identify].nextSecArrival = outbounds
            print('Car', identify, ':', visits)
            
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
                #Car is third or later in queue     
                else:
                    road.queue[queueindex + 1] = car
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
        update_intersections(0, intersections)
        gen_car(nextcargen)
        currentcar = cars[carSimIndex]
        movingcar = cars[0]
        arrivingcar = movingcar
        
        #Loops through time until runtime is reached
        while t < self.factors["T"]:
            #Assigns the next time a light changes
            nextLightTime = find_nextlightchange(intersections, t)[0] 
            #The next event is a car being introduced to the system
            if min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == minPrimArrival:
                t = minPrimArrival
                cars[carSimIndex].prevstop = t
                print('Car', cars[carSimIndex].identify, 'is arriving first at time', t)
                #A new car is generated
                nextcargen = t + arrival_rng.expovariate(self.factors["lambda"])
                minPrimArrival = nextcargen
                carSimIndex += 1
                gen_car(nextcargen)
                
                #The arriving car arrives into the system based on its path
                currentcar = cars[carSimIndex - 1]
                initroad = currentcar.path[currentcar.locationindex]
                find_place_in_queue(currentcar, initroad, t)
                       
            #The next event is a light changing    
            elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextLightTime:
                t = nextLightTime
                #Intersections that change lights at this time are updated
                update_intersections(t, intersections)
                         
            #The next event is a car starting to move            
            elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextStart:
                t = nextStart
                print('Time:', t, 'Car', movingcar.identify, 'is starting from road', movingcar.path[movingcar.locationindex].roadid, 'at spot', movingcar.placeInQueue)
                #Car is the first in its queue
                if movingcar.placeInQueue == 0:
                    #Car's next arrival is set
                    movingcar.nextSecArrival = t + (self.factors["distance"] / self.factors["speed"])
                #Car is not the first in its queue
                else:
                    #Car's next arrival time is set
                    movingcar.nextSecArrival = t + (self.factors["carlength"] / self.factors["speed"])
                    
                #Car leaves its current queue and is 'moving'
                movingcar.path[movingcar.locationindex].queue[movingcar.placeInQueue] = 0
                movingcar.moving = True
                movingcar.timewaiting = movingcar.timewaiting + (t - movingcar.prevstop)
                movingcar.nextstart = outbounds
                nextStart = outbounds
               
            #The next event is a car arriving within the system 
            elif min(minPrimArrival, nextLightTime, nextStart, nextSecArrival, nextcargen) == nextSecArrival:
                t = nextSecArrival    
                
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
                print('Time:', t, 'Car', arrivingcar.identify, 'is arriving at road', arrivingcar.path[arrivingcar.locationindex].roadid, 'at spot', arrivingcar.placeInQueue)
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
        
            sumwait = 0
            finishedcars = 0
            for car in cars:
                if car.locationindex == len(car.path) - 1 and car.identify > 100:
                    print('Car', car.identify, 'waiting time:', car.timewaiting)
                    sumwait += car.timewaiting
                    finishedcars += 1

            avgwait = sumwait /  finishedcars
            print(avgwait)

        # Compose responses and gradients.
        responses = {"WaitingTime": avgwait}
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
        self.dim = 1
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None  
        self.model_default_factors = {}
        self.model_decision_factors = {"offset"}      
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (0, 0, 0, 0)
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
        self.lower_bounds = (0,)*self.model.factors["numintersections"]
        self.upper_bounds = (self.model.factors["interval"],)*self.model.factors["numintersections"]
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
            "offset": vector                               
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
        vector = tuple(factor_dict["offset"])                               #change this to decision variable
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
        det_objectives_gradients = ((0,)*self.factors["numintersections"],)
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
        x = tuple([rand_sol_rng.uniform(0, self.model.factors["interval"]) for _ in self.dim])
        return x