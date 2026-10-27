from GameEngine import CluedoGameEngine
from Actor import CluedoActor
import time
game = CluedoGameEngine()

player1 = CluedoActor()
player2 = CluedoActor()
player3 = CluedoActor()
player4 = CluedoActor()
#player5 = CluedoActor()
#player6 = CluedoActor()
game.players.append(player1)
game.players.append(player2)
game.players.append(player3)
game.players.append(player4)
#game.players.append(player5)
#game.players.append(player6)

#time how long it takes to play this loop
start_time = time.time()
for i in range(1000000):
    game.start_game()
end_time = time.time()
print("Total time for 1000000 games:", end_time - start_time)