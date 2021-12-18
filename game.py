from enum import Enum
import random
import itertools
from typing import Union
from copy import deepcopy


class Block(Enum):
    EMPTY = 0
    WALL = 1
    CAT = 2

    def string(self) -> str:
        if self == Block.EMPTY:
            return "."
        elif self == Block.WALL:
            return "X"
        else:
            return "O"


Coord = tuple[int, int]


class Result(Enum):
    GAMING = 0
    WIN = 1
    LOSE = 2
    EXIT = 3


class GameCore:
    def __init__(self, height: int, width: int, wall_count: int) -> None:
        self.height = height
        self.width = width
        self.cat = int(height / 2), int(width / 2)
        self.result = Result.GAMING

        state = [[Block.EMPTY for _ in range(width)] for _ in range(height)]
        coords = list(itertools.product(range(width), range(height)))
        for i, j in random.sample(coords, wall_count):
            state[i][j] = Block.WALL
        self.state = state

        self.set_block(self.cat, Block.CAT)

    def print_generic(self, format) -> None:
        height_num_width = len(str(self.height - 1))
        print(" " * (height_num_width), end="")
        for j in range(self.width):
            print(f"{j: 2}", end="")
        print()
        for i in range(self.height):
            print(f"{i:<{height_num_width}} ", end="")
            if i % 2 == 1:
                print(" ", end="")
            for j in range(self.width):
                coord = (i, j)
                print(format(coord), end=" ")
            print()

    def print_state(self) -> None:
        self.print_generic(lambda coord: self.get_block(coord).string())

    def set_block(self, coord: Coord, block: Block) -> None:
        self.state[coord[0]][coord[1]] = block

    def get_block(self, coord: Coord) -> Block:
        return self.state[coord[0]][coord[1]]

    def is_valid_coord(self, coord: Coord) -> bool:
        return 0 <= coord[0] < self.height and 0 <= coord[1] < self.width

    def is_edge_coord(self, coord: Coord) -> bool:
        i, j = coord
        return i == 0 or i == self.height - 1 or j == 0 or j == self.width - 1

    def neighbors(self, coord: Coord) -> list[Coord]:
        i, j = coord
        delta = 1 if i % 2 == 0 else 0

        left = i, j - 1
        right = i, j + 1
        top_left = i - 1, j - delta
        top_right = i - 1, j + 1 - delta
        bottom_left = i + 1, j - delta
        bottom_right = i + 1, j + 1 - delta

        return list(filter(self.is_valid_coord, [left, top_left, top_right, right, bottom_right, bottom_left]))

    def cat_neighbors(self) -> list[Coord]:
        return self.neighbors(self.cat)

    def step_cat(self) -> None:
        distances = self.distances_to_edge()
        neighbors = filter(
            lambda coord: distances[coord[0]][coord[1]] is not None, self.cat_neighbors())
        next = min(
            neighbors, key=lambda coord: distances[coord[0]][coord[1]], default=None)

        if next is None:
            self.result = Result.WIN
        else:
            self.set_block(next, Block.CAT)
            self.set_block(self.cat, Block.EMPTY)
            self.cat = next

        if self.is_edge_coord(self.cat):
            self.result = Result.LOSE
            return

    def put_wall(self, coord: Coord) -> bool:
        if self.get_block(coord) == Block.EMPTY:
            self.set_block(coord, Block.WALL)
            return True
        return False

    def step(self) -> None:
        while True:
            inputs = input(">").split()
            if len(inputs) == 1:
                cmd = inputs[0]
                if cmd == "q":
                    self.result = Result.EXIT
                    break
                elif cmd == "p":
                    self.step_cat()
                    break
            elif len(inputs) == 2:
                i, j = (int(x) for x in inputs)
                if self.put_wall((i, j)):
                    self.step_cat()
                    break

    def distances_to_edge(self) -> list[list[Union[int, None]]]:
        queue: list[Coord] = []
        index = 0
        distances = [[None for _ in range(self.height)]
                     for _ in range(self.width)]

        def may_update(coord: Coord, new_dist: int = 0):
            if self.get_block(coord) == Block.WALL:
                return
            dist = distances[coord[0]][coord[1]]
            if dist is None or dist > new_dist:
                distances[coord[0]][coord[1]] = new_dist
                queue.append(coord)

        for j in range(self.width):
            may_update((0, j))
            may_update((self.height - 1, j))
        for i in range(self.height):
            may_update((i, 0))
            may_update((i, self.width - 1))

        while index < len(queue):
            coord = queue[index]
            index += 1
            dist = distances[coord[0]][coord[1]]
            for neighbor in self.neighbors(coord):
                may_update(neighbor, dist + 1)

        return distances

    def print_distances(self) -> None:
        distances = self.distances_to_edge()

        def format(coord: Coord) -> str:
            dist = distances[coord[0]][coord[1]]
            if dist is None:
                return "X"
            else:
                return str(dist)

        self.print_generic(format)


class Game:
    def __init__(self, height: int, width: int, wall_count: int) -> None:
        self.game = GameCore(height=height, width=width, wall_count=wall_count)
        self.histories = []

    def step_input(self) -> None:
        game_updated = False
        game_copy = deepcopy(self.game)

        while True:
            inputs = input(">").split()
            if len(inputs) == 1:
                cmd = inputs[0]
                if cmd == "q":
                    self.game.result = Result.EXIT
                    break
                elif cmd == "p":
                    self.game.step_cat()
                    game_updated = True
                    break
                elif cmd == "b":
                    if len(self.histories) > 0:
                        self.game = self.histories.pop()
                        break
                    else:
                        print("no history to go back")
                else:
                    print("bad command")
            elif len(inputs) == 2:
                i, j = (int(x) for x in inputs)
                if self.game.put_wall((i, j)):
                    self.game.step_cat()
                    game_updated = True
                    break
                else:
                    print("cannot put wall there")
            else:
                print("bad input")

        if game_updated:
            self.histories.append(game_copy)

    def run(self) -> None:
        while self.game.result == Result.GAMING:
            self.game.print_state()
            self.step_input()

        print(f"\n\n\nGAME OVER: {self.game.result.name}\n")
        self.game.print_state()
