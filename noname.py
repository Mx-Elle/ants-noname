#matthew&barry
import heapq
import random
import sys
from enum import Enum
import numpy as np
import numpy.typing as npt
from board import Entity, neighbors

AntMove = tuple[tuple[int, int], tuple[int, int]]
def valid_nbr(row: int, col: int, walls: npt.NDArray[np.int_]) -> list[tuple[int, int]]:
    all = neighbors((row, col), walls.shape)
    validl = []
    for n in all:
        if not walls[n]:
            validl.append(n)
    return validl

def heuristic(a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]) -> int:
    dr = min(abs(a[0] - b[0]), shape[0] - abs(a[0] - b[0]))
    dc = min(abs(a[1] - b[1]), shape[1] - abs(a[1] - b[1]))
    return dr + dc

def astar(start_coord: tuple[int, int], end_coord: tuple[int, int], shape: tuple[int, int], walls: npt.NDArray[np.int_]) -> list[tuple[int, int]] | None:
    frontier = []
    came = {start_coord: None}
    cost = {start_coord: 0}
    heapq.heappush(frontier, (0, random.randint(0, sys.maxsize), start_coord))
    
    while frontier:
        _, _, curr = heapq.heappop(frontier)
        
        if curr == end_coord:
            route = []
            temp = end_coord
            while temp != start_coord:
                route.append(temp)
                temp = came[temp]
            route.reverse()
            return route

        for nbs in neighbors(curr, shape):
            if walls[nbs]:
                continue
            
            curr_cost = cost[curr] + 1
            
            if nbs not in cost or curr_cost < cost[nbs]:
                cost[nbs] = curr_cost
                h = heuristic(nbs, end_coord, shape)
                heapq.heappush(frontier, (curr_cost + h, random.randint(0, sys.maxsize), nbs))
                came[nbs] = curr
                
    return None

class RandomBot:
    def __init__(self,walls: npt.NDArray[np.int_],harvest_radius: int,vision_radius: int,battle_radius: int,max_turns: int,time_per_turn: float,) -> None:
        self.walls = walls
        self.shape = walls.shape
        self.collect_radius = harvest_radius
        self.vision_radius = vision_radius
        self.battle_radius = battle_radius
        self.max_turns = max_turns
        self.time_per_turn = time_per_turn
        self.infer_enemy_hill = set()

    @property
    def name(self):
        return "noname666"

    def move_ants(self,vision: set[tuple[tuple[int, int], Entity]],stored_food: int,) -> set[AntMove]:
        out = set()
        fants = {coord for coord, kind in vision if kind == Entity.FRIENDLY_ANT}#friendly ants
        fhills = {coord for coord, kind in vision if kind == Entity.FRIENDLY_HILL}#friendly hills
        food_seen = {coord for coord, kind in vision if kind == Entity.FOOD}
        eants = {coord for coord, kind in vision if kind == Entity.ENEMY_ANT}
        
        #early in game, calculate coordinates of enemy base through symetry
        if not self.infer_enemy_hill and fhills:
            for r, c in fhills:
                self.infer_enemy_hill.add((self.shape[0] - r - 1, self.shape[1] - c - 1))

        #update infered hills
        for ant_loc in fants:
            if ant_loc in self.infer_enemy_hill:
                self.infer_enemy_hill.remove(ant_loc)
        
        #rush if opponent has one hill left and we have advantage on amount
        if len(self.infer_enemy_hill) == 1 and len(fants) > len(eants) * 2:
            target_hill = list(self.infer_enemy_hill)[0]
            claimed = set(fhills)
            for ant in fants:
                path = astar(ant, target_hill, self.shape, self.walls)
                if path and len(path) > 0:
                    next_pos = path[0]
                    if next_pos not in claimed:
                        claimed.add(next_pos)
                        out.add((ant, next_pos))
                        continue
                row, col = ant
                nlist = valid_nbr(row, col, self.walls)
                valid_moves = [n for n in nlist if n not in claimed]
                if valid_moves:
                    next_pos = random.choice(valid_moves)
                    claimed.add(next_pos)
                    out.add((ant, next_pos))
                else:
                    claimed.add(ant)
            return out
        
        #set friendly bsae as claimed so that new ants are forced to move out
        claimed = set(fhills)
        
        #task distribution
        for ant in fants:
            bestp = None#best path
            
            #food within sight+enemy base=targets
            poss_tar = list(food_seen) + list(self.infer_enemy_hill)
            
            #sort target by distance. A star prioritizes closest targets
            poss_tar.sort(key=lambda t: heuristic(ant, t, self.shape))
            #list.sort(key=lambda single target:distance from ant to target)
            
            #limit to 2 targets to prevent timeout
            for target in poss_tar[:1]:
                path = astar(ant, target, self.shape, self.walls)
                if path and len(path) > 0:
                    bestp = path
                    break

            if bestp:
                next = bestp[0]
                #check is claimed by other ants
                if next not in claimed:
                    claimed.add(next)
                    out.add((ant, next))
                    continue
            
            #fallback
            row, col = ant
            nlist = valid_nbr(row, col, self.walls)
            valid_moves = []
            for nbr in nlist:
                if nbr not in claimed:
                    valid_moves.append(nbr)
                    
            if valid_moves:
                next = random.choice(valid_moves)
                claimed.add(next)
                out.add((ant, next))
            else:
                #nowhere to go. stop
                claimed.add(ant)
                
        return out