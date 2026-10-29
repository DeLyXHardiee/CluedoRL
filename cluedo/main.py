from GameEngine import CluedoGameEngine
from Actor import CluedoActor
import time

for i in range(100):
    game = CluedoGameEngine(4, seed=i)
    game.start([CluedoActor() for _ in range (4)])
    game.play()