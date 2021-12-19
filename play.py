from game import InteractiveGame as Game


def play():
    game = Game(height=11, width=11, wall_count=8)
    game.run()


if __name__ == '__main__':
    play()
