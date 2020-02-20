import csv
import numpy as np
import copy
import random
import heapq
import time
import math
import argparse
import statistics

class Map:
    def __init__(self, industrial, commercial, residential, width, height, map):
        self.industrial = industrial
        self.commercial = commercial
        self.residential = residential
        self.width = width
        self.height = height
        self.scenic = []
        self.free_space = [] # empty tiles
        self.valid_cnt = self.createMap(map)

    def createMap(self, map):
        self.info = {}  # store the information of the map
        invalid_cnt = 0
        for h in range(self.height):
            for w in range(self.width):
                self.info.update({(h,w):map[h][w]})
                if map[h][w] == 'X':
                    invalid_cnt += 1
                if map[h][w] == 'S':
                    self.scenic.append((h,w))
                # save the empty tile positions
                if map[h][w] != 'S' and map[h][w] != 'X':
                    self.free_space.append((h,w))
        return self.height * self.width - invalid_cnt

    def checkMap(self, position):   # if the position is valid, return difficulty, otherwise return 0
        information = self.info[position]
        if information == 'X':
            return 0
        elif information == 'S':
            return 1
        else:
            return 2 + int(information)

def findNeighbors(position, map, distance):
    w = map.width
    h = map.height
    neighbors = []
    X, Y = position[0], position[1]
    for x in range(X-distance, X+distance+1):
        if x < 0: continue
        if x >= h: break
        for y in range(Y-distance+abs(X-x), Y+distance-abs(X-x)+1):
            if y < 0: continue
            if y >= w: break
            if x == X and y == Y: continue
            neighbors.append((x,y))
    return neighbors

def checkSameMap(pop1, pop2):
    for items in pop1.zones:    # check each different kind of zones
        if set(pop1.zones[items]) != set(pop2.zones[items]):
            return False
    return True

class Population:
    def __init__(self):
        self.zones = {'industrial':[],'commercial':[],'residential':[]}
        self.industrial = 0
        self.commercial = 0
        self.residential = 0
        self.score = 0
        self.extra_zones = [] # extra zones that we can place: ['name1', 'name2', 'namex']
        self.places_to_build = None # tile locations where we can build: empty tiles and scenic tiles

    def checkOverlap(self, position):   # find the positions already build a zone
        for items in self.zones:
            if position in self.zones[items]:
                return True
        return False

    def addZone(self, name, position, map): # if successfully added, return True, otherwise return False
        diff = map.checkMap(position)
        present_zone, valid_zone = checkZoneInfo(self,name,map)
        if diff == 0 or self.checkOverlap(position) or present_zone >= valid_zone:
            return False
        previous_list = self.zones[name]
        previous_list.append(position)
        self.zones.update({name:previous_list})
        return True

    def removeZone(self, name, position, map):
        if not self.checkOverlap(position): # no zone here, can't remove
            return False
        self.zones[name].remove(position)
        return True

    def calculateScore(self, map):  # whenever you make an action, add or remove, you need to calculate score
        self.score = 0
        self.industrial = len(self.zones['industrial'])
        self.commercial = len(self.zones['commercial'])
        self.residential = len(self.zones['residential'])
        for pos in self.zones['industrial']:
            diff = map.checkMap(pos)
            self.score -= diff
            neighbors = findNeighbors(pos,map,2)
            for items in neighbors:
                if map.info[items] == 'X': self.score -= 10 # Industrial zones within 2 tiles take a penalty of -10
                elif items in self.zones['industrial']: self.score += 2   # each industrial tile within 2 squares, bonus of 2
        for pos in self.zones['commercial']:
            diff = map.checkMap(pos)
            self.score -= diff
            neighbors_in_2 = findNeighbors(pos,map,2)
            neighbors_in_3 = findNeighbors(pos,map,3)
            for items in neighbors_in_2:
                if map.info[items] == 'X': self.score -= 20 # Commercial and residential zones within 2 tiles take a penalty of -20
                elif items in self.zones['commercial']: self.score -= 4   #  For each commercial site with 2 squares, penalty of 4
            for items in neighbors_in_3:
                if items in self.zones['residential']: self.score += 4  # For each residential tile within 3 squares, bonus of 4 points
        for pos in self.zones['residential']:
            diff = map.checkMap(pos)
            self.score -= diff
            neighbors_in_2 = findNeighbors(pos,map,2)
            neighbors_in_3 = findNeighbors(pos,map,3)
            for items in neighbors_in_2:
                if map.info[items] == 'X': self.score -= 20 # Commercial and residential zones within 2 tiles take a penalty of -20
                elif items in map.scenic and not self.checkOverlap(items): self.score += 10 # Residential zones within 2 tiles, bonus of 10
            for items in neighbors_in_3:
                if items in self.zones['industrial']: self.score -= 5  # For each industrial site within 3 squares, penalty of 5
                elif items in self.zones['commercial']: self.score += 4  # for each commercial site with 3 squares, bonus of 4

    def mergeMap(self, map):
        show_map = map.info.copy()
        for items in self.zones:
            for pos in self.zones[items]:
                show_map.update({pos:items})
        return show_map

class priorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0
    def push(self, priority, val):  # if successfully pushed, return True
        for maps in self.queue:
            if maps[0] < priority: continue
            elif maps[0] == priority:
                if checkSameMap(val,maps[2]):   # if it's the same map, don't add it
                    return False
            else: break
        heapq.heappush(self.queue,(priority,self.index,val))
        self.index += 1
        return True
    def get(self, k=1):   # k maps, if k > 0: the largest k, if k < 0: the smallest k.
        if k >= 0:
            return heapq.nlargest(k,self.queue)
        if k < 0:
            return heapq.nsmallest(0-k,self.queue)
    def sizeq(self):
        return len(self.queue)
    def pop(self, k):  # delete k smallest in queue
        while k > 0:
            heapq.heappop(self.queue)
            k -= 1

def randomGenerate(map, K, queue):  # need to modify later: if map.valid_cnt < total zones
    while K > 0:
        # time.sleep(0.0001)
        population = Population()
        total_count = 0
        generate_i = random.randint(0,map.industrial)
        for i in range(generate_i):
            if total_count == map.valid_cnt: break
            x = random.randint(0,map.height-1)
            y = random.randint(0,map.width-1)
            while not population.addZone('industrial',(x,y),map):
                x = random.randint(0,map.height-1)
                y = random.randint(0,map.width-1)
                continue
            total_count += 1
        generate_c = random.randint(0,map.commercial)
        for c in range(generate_c):
            if total_count == map.valid_cnt: break
            x = random.randint(0,map.height-1)
            y = random.randint(0,map.width-1)
            while not population.addZone('commercial',(x,y),map):
                x = random.randint(0,map.height-1)
                y = random.randint(0,map.width-1)
                continue
            total_count += 1
        generate_r = random.randint(0,map.residential)
        for r in range(generate_r):
            if total_count == map.valid_cnt: break
            x = random.randint(0,map.height-1)
            y = random.randint(0,map.width-1)
            while not population.addZone('residential',(x,y),map):
                x = random.randint(0,map.height-1)
                y = random.randint(0,map.width-1)
                continue
            total_count += 1
        population.calculateScore(map)
        if queue.push(population.score,population):
            K -= 1
        else: continue  # not success, then generate new one

# genetic algorithm
def checkZoneInfo(population, zone, map):
    if zone == 'industrial':
        return len(population.zones[zone]), map.industrial
    if zone == 'commercial':
        return len(population.zones[zone]), map.commercial
    if zone == 'residential':
        return len(population.zones[zone]), map.residential
    return 0, 0
def mutation(population, map, zone='all', mutation_rate = 0.2):
    mutation_prob = mutation_rate
    if random.randint(0,99) >= 100 * mutation_prob:  # do not mutate
        population.calculateScore(map)
        return population
    # otherwise, mutate
    zone_names = ['industrial', 'commercial', 'residential']
    if zone == 'all':
        zone = zone_names[random.randint(0,2)]
    if len(population.zones[zone]) == 0 and checkZoneInfo(population,zone,map)[1] != 0: # is empty
        while True:
            rand_x = random.randint(0,map.height-1)
            rand_y = random.randint(0,map.width-1)
            if population.addZone(name=zone,position=(rand_x,rand_y),map=map):
                break
    elif len(population.zones[zone]) != 0 and checkZoneInfo(population,zone,map)[0] != checkZoneInfo(population,zone,map)[1]:
        pos = population.zones[zone][random.randint(0, len(population.zones[zone]) - 1)]
        while True:
            rand_x = random.randint(0,map.height-1)
            rand_y = random.randint(0,map.width-1)
            if population.addZone(name=zone,position=(rand_x,rand_y),map=map):
                break
        if random.randint(0,99) >= 100 * mutation_prob: # only move the zone to another place
            population.removeZone(zone,pos,map)
        else:   # add a zone, but need to check if it's acceptable
            present_zone, valid_zone = checkZoneInfo(population,zone,map)
            if present_zone > valid_zone: population.removeZone(zone,pos,map)  # can't add

    population.calculateScore(map)
    return population

def crossover(pop1, pop2, queue, map, mutation_rate):
    children = [Population(), Population()]
    zone_names = ['industrial','commercial','residential']
    zone_select = random.randint(0,2)
    zone = zone_names[zone_select]  # select one zone to crossover

    children[0] = copy.deepcopy(pop1)
    children[1] = copy.deepcopy(pop2)
    if set(pop1.zones[zone]) == set(pop2.zones[zone]):  # exactly the same arrangement for this zone, including empty
        mutation(children[0],zone=zone,map=map,mutation_rate=mutation_rate)
        mutation(children[1],zone=zone,map=map,mutation_rate=mutation_rate)
        queue.push(children[0].score,children[0])
        queue.push(children[1].score,children[1])
        return
    else:
        if len(pop1.zones[zone]) == 0:
            newpos = pop2.zones[zone][random.randint(0,len(pop2.zones[zone])-1)]
            if children[0].addZone(zone,newpos,map):
                children[1].removeZone(zone, newpos, map)
        elif len(pop2.zones[zone]) == 0:
            newpos = pop1.zones[zone][random.randint(0,len(pop1.zones[zone])-1)]
            if children[1].addZone(zone,newpos,map):
                children[0].removeZone(zone, newpos, map)
        else:
            newpos1 = pop2.zones[zone][random.randint(0, len(pop2.zones[zone]) - 1)]
            newpos2 = pop1.zones[zone][random.randint(0, len(pop1.zones[zone]) - 1)]
            if children[0].checkOverlap(newpos1) or children[1].checkOverlap(newpos2):
                mutation(children[0], zone=zone,map=map,mutation_rate=mutation_rate)
                mutation(children[1], zone=zone,map=map,mutation_rate=mutation_rate)
            else:
                children[0].addZone(zone,newpos1,map)
                children[1].removeZone(zone,newpos1,map)
                children[0].removeZone(zone,newpos2,map)
                children[1].addZone(zone,newpos2,map)
    mutation(children[0],map=map,mutation_rate=mutation_rate)
    mutation(children[1],map=map,mutation_rate=mutation_rate)
    queue.push(children[0].score, children[0])
    queue.push(children[1].score, children[1])

def culling(lowest_k2, queue):
    queue.pop(lowest_k2)

def elitism(highest_k1, queue, top_k1):
    heap_k1 = queue.get(highest_k1)
    heapq.heapify(heap_k1)
    list_k1 = list(heapq.merge(heap_k1,top_k1))
    heapq.heapify(list_k1)
    new_k1 = heapq.nlargest(highest_k1,list_k1)
    heapq.heapify(new_k1)
    return new_k1

def geneticAlgorithm(queue, map, K, highest_k1, lowest_k2, mutation_rate):
    times = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 666]
    current = times.pop(0)
    randomGenerate(map, K, queue)
    start_time = time.time()

    top_k1 = queue.get(highest_k1)
    time_elapsed = 0
    best_score = -999
    best_time = 0
    while time_elapsed < 10:
        size_q = queue.sizeq()
        states = queue.get(size_q)
        queue.pop(size_q)   # clear up queue
        for id,state in enumerate(states):
            mate_id = random.randint(0,highest_k1-1)
            mate = top_k1[mate_id]
            crossover(state[2],mate[2],queue,map,mutation_rate=mutation_rate)
        heapq.heapify(top_k1)
        top_k1 = elitism(highest_k1,queue,top_k1)
        if queue.sizeq() > K:
            culling(queue.sizeq() - K,queue)

        # printMap(heapq.nlargest(1,top_k1)[0],map)
        time_elapsed = time.time() - start_time
        if heapq.nlargest(1, top_k1)[0][0] > best_score:
            best_score = heapq.nlargest(1, top_k1)[0][0]
            best_time = time_elapsed
        # if time_elapsed >= current:
        #     print("t: %d"%current)
        #     printMap(heapq.nlargest(1, top_k1)[0], map)
        #     current = times.pop(0)
    # printMap(heapq.nlargest(1, top_k1)[0], map)
    print(time_elapsed)
    printFile(heapq.nlargest(1, top_k1)[0],best_time,best_score,map)


def printFile(population, best_time, best_score, map):
    fp = open("ga_result.txt","w")
    show_map = population[2].mergeMap(map)
    w, h= map.width, map.height
    fp.write(str(best_score)+"\n")
    fp.write(str(best_time)+"\n")
    for x in range(h):
        printline = ""
        for y in range(w):
            info = show_map[(x,y)]
            if info.__len__() == 1 and info != 'X' and info != 'S':
                info = '.'
            if info.__len__() > 1:
                info = info[0]
            printline += info + " "
        fp.write(printline+"\n")
    fp.close()

def printMap(populationMap,map):    # print the graph and the score of one map      (from the queue!)
    show_map = populationMap[2].mergeMap(map)
    w = map.width
    h = map.height
    for x in range(h):
        printline = ""
        for y in range(w):
            info = show_map[(x,y)]
            if info.__len__() == 1 and info != 'X' and info != 'S':
                info = '.'
            if info.__len__() > 1:
                info = info[0]
            printline += info + " "
        print(printline)
    print(populationMap[0])  # the score of this map

def setup(filename):
    fp = open(filename,"r")

    reader = csv.reader(fp)
    input_file = []
    for item in reader:
        input_file.append(item)
    industrial, commercial, residential = int(input_file[0][0]), int(input_file[1][0]), int(input_file[2][0])
    width = len(input_file[3]) # the width of the map
    height = len(input_file[3:]) # the height of the map
    urbanMap = Map(industrial, commercial, residential, width, height, input_file[3:])
    fp.close()
    return urbanMap


########################Simulated Annealing Start##########################################################################################################################################################################################
class SimulatedAnnealing:
    def __init__(self, input_file, initial_temperature, run_time, step):
        self.actions = ['move', 'add', 'remove']
        self.initial_temperature = initial_temperature # set temperature
        self.restart_count = 0 # number of times it will restart
        self.time_step = step
        self.run_time = run_time # how long the algorithm will run
        self.start_time = None
        self.best_population_time = None # how long it took to get the best map
        self.best_population = None # the best map
        self.map = setup(input_file) # create new map

    # run simulated annealing algorithm
    def run(self):
        # store the program's start time
        self.start_time = time.time()
        while (time.time() - self.start_time) < self.run_time:
            # create a new population
            n = 0
            population = self.randomGenerateAnnealing()
            temperature = self.initial_temperature
            recent_scores = []
            best_score = None
            # restart if we haven't made progress in a while or we have ran out of time
            while len(recent_scores) < 500 and (time.time()-self.start_time) < self.run_time:
                valid_action = False
                # pick a random action to make
                action = self.actions[random.randint(0,len(self.actions)-1)]
                ### ACTION -> MOVE A ZONE
                if action == 'move':
                    # get all of the zones that are on the map
                    zones = self.getZones(population)
                    # make sure that there are zones to place and places to move it
                    if len(zones) > 0 and len(population.places_to_build) > 0:
                        population = self.moveZoneAnnealing(population, zones, temperature)
                        valid_action = True
                ### ACTION -> ADD A ZONE
                elif action == 'add':
                    if len(population.extra_zones) > 0 and len(population.places_to_build) > 0:
                        population = self.addZoneAnnealing(population, temperature)
                        valid_action = True
                ### ACTION -> REMOVE A ZONE
                elif action == 'remove':
                    # get all of the zones that are placed on the map
                    zones = self.getZones(population)
                    if len(zones) > 0:
                        population = self.removeZoneAnnealing(population, zones, temperature)
                        valid_action = True
                    # only update the temperature if we performed a valid action
                if valid_action == True:
                    if best_score == None or population.score > best_score:
                        best_score = population.score
                        recent_scores = []
                    else:
                        recent_scores.append(population.score)
                    n += self.time_step
                    prev_temperature = temperature
                    temperature = (0.9**n)*(prev_temperature) # geometric


            # after running, update the best population
            if self.best_population == None or self.best_population.score < population.score:
                self.best_population_time = time.time() - self.start_time
                self.best_population = None
                self.best_population = population
            self.restart_count += 1
        text = self.printMapAnnealing(self.best_population)
        self.saveMapToFile(self.best_population, text)

    def moveZoneAnnealing(self, population, zones, temperature):
        # copy the current population
        updated_population = copy.deepcopy(population)
        # get the zone that we are going to move
        zone_index = random.randint(0,len(zones)-1)
        zone_to_move = zones[zone_index]
        # get the new location for the zone
        pos_index = random.randint(0,len(updated_population.places_to_build)-1)
        new_position = updated_population.places_to_build.pop(pos_index)
        # move the zone
        updated_population.removeZone(zone_to_move['name'], zone_to_move['position'], self.map)
        updated_population.addZone(zone_to_move['name'], new_position, self.map)
        # update the places where we can build by adding the position where the zone was previously located
        updated_population.places_to_build.append(zone_to_move['position'])
        # calculate the difference in scores
        updated_population.calculateScore(self.map)
        population.calculateScore(self.map)
        dE = updated_population.score - population.score
        # decide whether or not to update the current population
        if self.makeDecision(dE, temperature) == True:
            population = None
            population = updated_population
        else:
            updated_population = None
        return population

    def addZoneAnnealing(self, population, temperature):
        # copy the current population
        updated_population = copy.deepcopy(population)
        # get a zone to add
        zone_index = random.randint(0, len(updated_population.extra_zones)-1)
        zone_to_add = updated_population.extra_zones.pop(zone_index)
        # find an empty space to add zone
        position_index = random.randint(0, len(updated_population.places_to_build)-1)
        zone_position = updated_population.places_to_build.pop(position_index)
        # add zone to position
        updated_population.addZone(zone_to_add, zone_position, self.map)
        population.calculateScore(self.map)
        updated_population.calculateScore(self.map)
        dE = updated_population.score - population.score
        if self.makeDecision(dE, temperature) == True:
            population = None
            population = updated_population
        else:
            updated_population = None
        return population

    def removeZoneAnnealing(self, population, zones, temperature):
        # copy the current population
        updated_population = copy.deepcopy(population)
        # get a random zone to remove
        zone_index = random.randint(0, len(zones)-1)
        zone_to_remove = zones[zone_index]
        # remove zone position
        updated_population.removeZone(zone_to_remove['name'], zone_to_remove['position'], self.map)
        # add zone to extra_zones
        updated_population.extra_zones.append(zone_to_remove['name'])
        # calculate map score
        population.calculateScore(self.map)
        updated_population.calculateScore(self.map)
        dE = updated_population.score - population.score
        if self.makeDecision(dE, temperature) == True:
            population = None
            population = updated_population
        else:
            updated_population = None
        return population

    # create a new random population [for simulated annealing]
    def randomGenerateAnnealing(self):
        # creat new populaion
        population = Population()
        # tile locations where we can build zones
        population.places_to_build = copy.deepcopy(self.map.free_space) + copy.deepcopy(self.map.scenic)
        map_size = self.map.height * self.map.width
        # select a random number of zones to place on the map
        industrial_count = random.randint(0, self.map.industrial)
        commercial_count = random.randint(0, self.map.commercial)
        residential_count = random.randint(0, self.map.residential)
        zones = []
        # don't try to place a zone if prof does not want us to!
        if industrial_count > 0:
            zones.append({'name': 'industrial', 'count': industrial_count})
        if commercial_count > 0:
            zones.append({'name': 'commercial', 'count': commercial_count})
        if residential_count > 0:
            zones.append({'name': 'residential', 'count': residential_count})
        # store extra zones
        for count in range(self.map.industrial - industrial_count):
            population.extra_zones.append('industrial')
        for count in range(self.map.commercial - commercial_count):
            population.extra_zones.append('commercial')
        for count in range(self.map.residential - residential_count):
            population.extra_zones.append('residential')
        # calculate the number of empty tiles (plus scenic tiles) we have left
        while len(population.places_to_build) > 0 and len(zones) > 0:
            # select a random zone to place
            zone_index = random.randint(0, len(zones)-1)
            zone_to_add = zones[zone_index]
            # get a random position to place zone
            position_index  = random.randint(0, len(population.places_to_build)-1)
            zone_position = population.places_to_build.pop(position_index)
            # add the zone to the map
            population.addZone(zone_to_add['name'], zone_position, self.map)
            # decrement the number of zones to place
            zones[zone_index]['count'] -= 1
            # get rid of this zone if there arent any left to place
            if zones[zone_index]['count'] == 0:
                del(zones[zone_index])
        # calculate the score for this population
        population.calculateScore(self.map)
        return population

    def makeDecision(self, dE, temperature):
        # accept the new state since it gives us a better score
        if dE > 0:
            return True
        # accept the new state (worse) with some probability
        elif temperature > 0:
            # the probability that we will take a worse step
            if np.exp(dE/temperature) >= random.random():
                return True
        return False

    # restructure the format of the zones for easy access
    def getZones(self, population):
        zones = []
        for name in population.zones:
            for position in population.zones[name]:
                item = {'name': name, 'position': position}
                zones.append(item)
        return zones

    def printMapAnnealing(self, population):
        population.calculateScore(self.map)
        outputString = ''
        for position in self.map.info:
            if position in population.zones['industrial']:
                outputString += 'i '
            elif position in population.zones['commercial']:
                outputString += 'c '
            elif position in population.zones['residential']:
                outputString += 'r '
            else:
                if self.map.info[position] == 'X' or self.map.info[position] == 'S':
                    outputString += self.map.info[position] + ' '
                else:
                    outputString += '.' + ' '
            if position[1] == self.map.width - 1:
                outputString += '\n'
        outputString += '\n\n'
        return outputString

    def saveMapToFile(self, population, outputString):
        fp = open("hc_result.txt", "w")
        fp.write('Score: {}\n'.format(population.score)) # write the best score to file
        fp.write('Time: {}\n'.format(self.best_population_time)) # write the time of best map to file
        fp.write(outputString) # write the map to a file
        fp.write('Restarts: {}\n'.format(self.restart_count))
        fp.close()


# get the name of the input file for the map and the algorithm used
def getInput():
    my_parser = argparse.ArgumentParser(description='Please add some command line inputs... if you do not, I won"t know how to behave')
    my_parser.add_argument('input_file', help='map')
    my_parser.add_argument('algorithm', help='algorithm')
    args = my_parser.parse_args()
    return args.input_file, args.algorithm

if __name__ == "__main__":
    # get the map and algorithm
    input_file, algorithm = getInput()
    if algorithm.upper() == 'GA':
        urbanMap = setup(input_file)
        mapQueue = priorityQueue()
        geneticAlgorithm(mapQueue,urbanMap,300,30,10,0.4)
    elif algorithm.upper() == 'HC':
        SA = SimulatedAnnealing(input_file, 50, 10, 1)
        SA.run()
