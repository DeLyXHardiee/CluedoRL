import test from 'node:test';
import assert from 'node:assert/strict';
import { Memory } from '../cluedo_game_engine/src/models/Memory.js';
import { considerAccusation, makeSuggestion } from '../cluedo_game_engine/src/services/heuristicBot.js';
import { SUSPECTS, WEAPONS, ROOMS } from '../cluedo_game_engine/src/config/gameConstants.js';

test('addKnownCard updates sets correctly', () => {
  const memory = new Memory('agent1');
  memory.eliminatedCards.add('Candlestick');
  memory.addKnownCard('Candlestick');
  assert.ok(memory.knownCards.has('Candlestick'));
  assert.ok(!memory.eliminatedCards.has('Candlestick'));
});

test('doesPlayerNotHaveCard reflects negative constraints', () => {
  const memory = new Memory('agent2');
  memory.playerNegativeConstraints.set('Alice', new Set(['Rope']));
  assert.ok(memory.doesPlayerNotHaveCard('Alice', 'Rope'));
  assert.ok(!memory.doesPlayerNotHaveCard('Alice', 'Knife'));
});

test('reset clears memory state', () => {
  const memory = new Memory('agent3');
  memory.addKnownCard('Lead Pipe');
  memory.eliminatedCards.add('Revolver');
  memory.memoryHistory.push({ turnNumber: 1 });
  memory.reset();
  assert.equal(memory.knownCards.size, 1);
  assert.ok(memory.knownCards.has('Lead Pipe'));
  assert.equal(memory.eliminatedCards.size, 0);
  assert.equal(memory.playerNegativeConstraints.size, 0);
  assert.equal(memory.memoryHistory.length, 0);
});

test('updateMemory consumes structured turn events', async () => {
  const memory = new Memory('agent4');

  await memory.updateMemory([
    { type: 'suggestion', suspect: 'Miss Scarlet', weapon: 'Rope', room: 'Study' },
    { type: 'cardShown', from: 'agent2', to: 'agent4', card: 'Rope' }
  ]);

  assert.ok(memory.eliminatedCards.has('Rope'));
  assert.deepEqual(memory.formatMemoryForAgent().knownInformation.knownCardsByPlayer, {
    agent4: [],
    agent2: ['Rope']
  });
});

test('updateMemory records solution cards from an unchallenged suggestion', async () => {
  const memory = new Memory('agent5');

  await memory.updateMemory([
    { type: 'suggestion', suspect: 'Miss Scarlet', weapon: 'Rope', room: 'Study' },
    {
      type: 'challenge',
      canChallenge: false,
      challengingAgent: null,
      canInferUnchallengedSolution: true,
      suggestingAgentCards: ['Rope']
    }
  ]);

  assert.deepEqual(
    [...memory.confirmedSolutionCards].sort(),
    ['Miss Scarlet', 'Study'].sort()
  );
  assert.ok(memory.eliminatedCards.has('Rope'));
});

test('observers do not infer solution cards from another agent’s unchallenged suggestion', async () => {
  const memory = new Memory('observer');

  await memory.updateMemory([
    { type: 'suggestion', suspect: 'Miss Scarlet', weapon: 'Rope', room: 'Study' },
    { type: 'challenge', canChallenge: false, canInferUnchallengedSolution: false }
  ]);

  assert.equal(memory.confirmedSolutionCards.size, 0);
  assert.equal(memory.eliminatedCards.size, 0);
});

test('certain heuristic accuses when an unchallenged suggestion identifies all solution cards', async () => {
  const memory = new Memory('agent6');
  await memory.updateMemory([
    { type: 'suggestion', suspect: 'Miss Scarlet', weapon: 'Rope', room: 'Study' },
    {
      type: 'challenge',
      canChallenge: false,
      canInferUnchallengedSolution: true,
      suggestingAgentCards: []
    }
  ]);

  const decision = considerAccusation({
    name: 'Red Agent',
    policy: 'cand-random-certain',
    cards: new Set(),
    memory
  });

  assert.equal(decision.shouldAccuse, true);
  assert.deepEqual(decision.accusation, {
    suspect: 'Miss Scarlet',
    weapon: 'Rope',
    room: 'Study'
  });
});

test('candidate heuristic avoids recently suggested unresolved cards', () => {
  const memory = new Memory('agent7');
  memory.memoryHistory = SUSPECTS.slice(0, 3).map((suspect, turnNumber) => ({
    turnNumber,
    suggestion: { suspect, weapon: WEAPONS[0], room: ROOMS[0] }
  }));

  const suggestion = makeSuggestion({
    policy: 'cand-random-certain',
    cards: new Set(),
    memory,
    location: ROOMS[0]
  }, {
    availableSuspects: SUSPECTS,
    availableWeapons: WEAPONS,
    availableRooms: ROOMS
  });

  assert.ok(!SUSPECTS.slice(0, 3).includes(suggestion.suspect));
});

test('first heuristic selects the first card not known to be outside the envelope', () => {
  const memory = new Memory('agent8');
  memory.eliminatedCards.add(SUSPECTS[1]);
  memory.eliminatedCards.add(WEAPONS[1]);
  memory.eliminatedCards.add(ROOMS[1]);

  const agent = {
    policy: 'first-random-certain',
    cards: new Set([SUSPECTS[0], WEAPONS[0], ROOMS[0]]),
    memory,
    location: ROOMS[8]
  };
  const suggestion = makeSuggestion(agent, {
    availableSuspects: SUSPECTS,
    availableWeapons: WEAPONS,
    availableRooms: ROOMS
  });

  assert.deepEqual(suggestion, {
    suspect: SUSPECTS[2],
    weapon: WEAPONS[2],
    room: ROOMS[2],
    reasoning: 'Heuristic strategy "first-random-certain" selected a first suggestion.'
  });
});

test('candidate heuristic chooses an unresolved room instead of its current location', () => {
  const memory = new Memory('agent9');
  const agent = {
    policy: 'cand-random-certain',
    cards: new Set(ROOMS.slice(0, -1)),
    memory,
    location: ROOMS[0]
  };

  const suggestion = makeSuggestion(agent, {
    availableSuspects: SUSPECTS,
    availableWeapons: WEAPONS,
    availableRooms: ROOMS
  });

  assert.equal(suggestion.room, ROOMS.at(-1));
});
