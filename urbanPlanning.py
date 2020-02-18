import csv
import numpy as np
import copy
import random
import heapq
import time

class Map:
    def __init__(self, industrial, commercial, residential, width, height, map):
        self.industrial = industrial
        self.commercial = commercial
        self.residential = residential
        self.width = width
        self.height = height
        self.scenic = []
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

    def checkOverlap(self, position):   # find the positions already build a zone
        for items in self.zones:
            if position in self.zones[items]:
                return True
        return False

    def addZone(self, name, position, map): # if successfully added, return True, otherwise return False
        diff = map.checkMap(position)
        if diff == 0 or self.checkOverlap(position):
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
        time.sleep(0.0001)
        population = Population()
        total_count = 0
        if map.valid_cnt >= map.industrial + map.commercial + map.residential:
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
def mutation(population, map, zone='all', mutation_rate = 0.2):
    mutation_prob = mutation_rate
    if random.randint(0,99) >= 100 * mutation_prob:  # do not mutate
        population.calculateScore(map)
        return population
    # otherwise, mutate
    zone_names = ['industrial', 'commercial', 'residential']
    if zone == 'all':
        zone = zone_names[random.randint(0,2)]
    if len(population.zones[zone]) == 0: # is empty
        while True:
            rand_x = random.randint(0,map.height-1)
            rand_y = random.randint(0,map.width-1)
            if population.addZone(name=zone,position=(rand_x,rand_y),map=map):
                break
    else:
        pos = population.zones[zone][random.randint(0, len(population.zones[zone]) - 1)]
        while True:
            rand_x = random.randint(0,map.height-1)
            rand_y = random.randint(0,map.width-1)
            if population.addZone(name=zone,position=(rand_x,rand_y),map=map):
                break
        population.removeZone(zone,pos,map)
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
        if time_elapsed >= current:
            print("t: %d"%current)
            printMap(heapq.nlargest(1, top_k1)[0], map)
            current = times.pop(0)
    printMap(heapq.nlargest(1, top_k1)[0], map)
    print(time_elapsed)


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


if __name__ == "__main__":
    urbanMap = setup("urban 2.txt")
    mapQueue = priorityQueue()

    geneticAlgorithm(mapQueue,urbanMap,400,30,10,0.8)

