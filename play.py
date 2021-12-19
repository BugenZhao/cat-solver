from game import InteractiveGame as Game


def play():
    game = Game(height=11, width=11, wall_count=60)
    game.run()
