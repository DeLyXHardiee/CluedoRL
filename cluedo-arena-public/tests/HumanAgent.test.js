import test from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../cluedo_game_engine/src/models/Game.js';
import { HumanAgent } from '../cluedo_game_engine/src/models/HumanAgent.js';
import { SUSPECTS, WEAPONS, ROOMS } from '../cluedo_game_engine/src/config/gameConstants.js';

test('play mode creates a human player and waits for their action', async () => {
  const game = new Game('play');
  await game.initialize();

  assert.ok(game.humanPlayer instanceof HumanAgent);
  assert.equal(game.getGameSummary().hasHumanPlayer, true);
  assert.equal(game.getGameSummary().solution, null);

  const result = await game.processTurn();
  assert.deepEqual(result, { awaitingHuman: true });

  await game.processHumanAction({
    type: 'suggestion',
    suspect: SUSPECTS[0],
    weapon: WEAPONS[0],
    room: ROOMS[0]
  });

  assert.equal(game.currentTurn, 1);
  assert.equal(game.activeAgentIndex, 1);
  assert.equal(game.humanPlayer.location, ROOMS[0]);
  assert.equal(game.getGameSummary().humanMemory.turnHistory.length, 1);
});

test('accusation results record the active turn', async () => {
  const game = new Game('play');
  await game.initialize();
  game.currentTurn = 12;

  game.checkAccusation(
    game.humanPlayer,
    SUSPECTS[0],
    WEAPONS[0],
    ROOMS[0]
  );

  const accusationResult = game.gameLog.find(
    event => event.type === 'accusationResult'
  );
  assert.equal(accusationResult.type, 'accusationResult');
  assert.equal(accusationResult.turn, 12);
});
