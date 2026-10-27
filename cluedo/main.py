from GameEngine import CluedoGameEngine
from Actor import CluedoActor
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


for i in range(100):
    game.start_game()

