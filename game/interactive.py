from .core import *
from copy import deepcopy


class InteractiveGame:
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
