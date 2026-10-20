import { SUSPECTS, WEAPONS, ROOMS } from '../config/gameConstants.js';

const STRATEGIES = {
  'random-random-certain': { suggest: 'random', reveal: 'random', accuse: 'certain' },
  'first-random-certain': { suggest: 'first', reveal: 'random', accuse: 'certain' },
  'first-smart-certain': { suggest: 'first', reveal: 'smart', accuse: 'certain' },
  'cand-random-certain': { suggest: 'cand', reveal: 'random', accuse: 'certain' },
  'focused-random-certain': { suggest: 'focused', reveal: 'random', accuse: 'certain' },
  'bluff-random-certain': { suggest: 'bluff', reveal: 'random', accuse: 'certain' },
  'cand-smart-certain': { suggest: 'cand', reveal: 'smart', accuse: 'certain' },
  'cand-random-gamble2': { suggest: 'cand', reveal: 'random', accuse: 'gamble2' },
  'cand-random-gamble3': { suggest: 'cand', reveal: 'random', accuse: 'gamble3' },
  'focused-smart-gamble2': { suggest: 'focused', reveal: 'smart', accuse: 'gamble2' },
  'focused-smart-gamble3': { suggest: 'focused', reveal: 'smart', accuse: 'gamble3' }
};

const STRATEGY_ALIASES = {
  random: 'random-random-certain',
  first: 'first-random-certain',
  candidates: 'cand-random-certain',
  focused: 'focused-random-certain',
  bluff: 'bluff-random-certain'
};

let activeStrategyName = 'focused-random-certain';
const shownLog = new Map();

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function resolveStrategy(strategyName = activeStrategyName) {
  return STRATEGIES[strategyName] || STRATEGIES[activeStrategyName];
}

function resolveStrategyName(strategyName) {
  const normalizedName = strategyName?.toLowerCase();
  const resolvedName = STRATEGY_ALIASES[normalizedName] || normalizedName;
  return STRATEGIES[resolvedName] ? resolvedName : activeStrategyName;
}

function envelopeCandidates(agent, cards) {
  return cards.filter(card =>
    !agent.cards.has(card) && !agent.memory.eliminatedCards.has(card)
  );
}

function recentSuggestions(agent, categoryIndex) {
  return new Set(
    (agent.memory.memoryHistory || [])
      .slice(-3)
      .map(entry => entry.suggestion?.[categoryIndex])
      .filter(Boolean)
  );
}

function suggestionCard(agent, cards, mode, categoryIndex, bluffCategory = null) {
  const candidates = envelopeCandidates(agent, cards);
  const category = ['suspect', 'weapon', 'room'][categoryIndex];
  const unexploredCandidates = candidates.filter(card =>
    !recentSuggestions(agent, category).has(card)
  );
  const availableCandidates = unexploredCandidates.length ? unexploredCandidates : candidates;

  if (mode === 'random') return randomItem(cards);
  if (mode === 'first') return candidates[0] || cards[0];
  if (mode === 'bluff' && categoryIndex === bluffCategory) {
    const ownCards = cards.filter(card => agent.cards.has(card));
    if (ownCards.length) return randomItem(ownCards);
  }
  if (mode === 'focused' && candidates.length === 1) {
    const ownCards = cards.filter(card => agent.cards.has(card));
    return ownCards.length ? randomItem(ownCards) : candidates[0];
  }
  return randomItem(availableCandidates.length ? availableCandidates : cards);
}

function makeSuggestionForMode(agent, gameState, mode, strategyName) {
  const bluffCategory = mode === 'bluff'
    ? [SUSPECTS, WEAPONS, ROOMS]
      .map((cards, index) => envelopeCandidates(agent, cards).length > 1 ? index : -1)
      .filter(index => index >= 0)
      .filter(() => Math.random() < 0.3)[0] ?? null
    : null;
  return {
    suspect: suggestionCard(agent, gameState.availableSuspects, mode, 0, bluffCategory),
    weapon: suggestionCard(agent, gameState.availableWeapons, mode, 1, bluffCategory),
    room: suggestionCard(agent, gameState.availableRooms, mode, 2, bluffCategory),
    reasoning: `Heuristic strategy "${strategyName}" selected a ${mode} suggestion.`
  };
}

function evaluateChallengeForMode(agent, cards, mode, askingAgentName, strategyName) {
  const agentLog = shownLog.get(agent.name) || new Map();
  const cardToShow = mode === 'smart'
    ? cards.find(card => agentLog.get(card)?.has(askingAgentName)) ||
      cards.reduce((best, card) =>
        (agentLog.get(card)?.size || 0) > (agentLog.get(best)?.size || 0) ? card : best
      )
    : randomItem(cards);

  if (cardToShow && askingAgentName) {
    if (!agentLog.has(cardToShow)) agentLog.set(cardToShow, new Set());
    agentLog.get(cardToShow).add(askingAgentName);
    shownLog.set(agent.name, agentLog);
  }

  return {
    cardToShow: cardToShow || null,
    reasoning: cardToShow
      ? `Heuristic strategy "${strategyName}" selected a ${mode} reveal.`
      : 'No matching card is available.'
  };
}

function accusationCandidates(agent) {
  const confirmed = agent.memory.confirmedSolutionCards;
  if (confirmed && confirmed.size > 0) {
    return [
      SUSPECTS.filter(card => confirmed.has(card)),
      WEAPONS.filter(card => confirmed.has(card)),
      ROOMS.filter(card => confirmed.has(card))
    ].map((category, index) => category.length ? category : envelopeCandidates(agent, [SUSPECTS, WEAPONS, ROOMS][index]));
  }

  return [
    envelopeCandidates(agent, SUSPECTS),
    envelopeCandidates(agent, WEAPONS),
    envelopeCandidates(agent, ROOMS)
  ];
}

function considerAccusationForMode(agent, mode, strategyName) {
  const candidates = accusationCandidates(agent);
  const combinations = candidates.reduce((total, category) => total * category.length, 1);
  const maxCombinations = mode === 'gamble2' ? 2 : mode === 'gamble3' ? 3 : 1;
  const shouldAccuse = candidates.every(category => category.length === 1) ||
    (mode !== 'certain' && combinations <= maxCombinations);
  const accusation = shouldAccuse
    ? candidates.map(category => randomItem(category))
    : [null, null, null];

  return {
    shouldAccuse,
    accusation: {
      suspect: accusation[0],
      weapon: accusation[1],
      room: accusation[2]
    },
    confidence: {
      suspect: candidates[0].length === 1 ? 1 : 0,
      weapon: candidates[1].length === 1 ? 1 : 0,
      room: candidates[2].length === 1 ? 1 : 0
    },
    reasoning: shouldAccuse
      ? `Heuristic strategy "${strategyName}" made an accusation.`
      : 'The solution is not uniquely determined and is too broad for a gamble.'
  };
}

export function listStrategies() {
  return Object.keys(STRATEGIES);
}

export function setStrategy(strategyName) {
  const normalizedName = strategyName?.toLowerCase();
  const resolvedName = STRATEGY_ALIASES[normalizedName] || normalizedName;
  if (!STRATEGIES[resolvedName]) {
    throw new Error(`Unknown heuristic strategy "${strategyName}". Available strategies: ${listStrategies().join(', ')}`);
  }
  activeStrategyName = resolvedName;
}

export function getStrategy() {
  return activeStrategyName;
}

export function configureFromEnvironment() {
  setStrategy(process.env.HEURISTIC_STRATEGY || activeStrategyName);
}

export function makeSuggestion(agent, gameState) {
  const strategyName = resolveStrategyName(agent.policy);
  return makeSuggestionForMode(agent, gameState, resolveStrategy(strategyName).suggest, strategyName);
}

export function evaluateChallenge(agent, cards, askingAgentName) {
  const strategyName = resolveStrategyName(agent.policy);
  return evaluateChallengeForMode(agent, cards, resolveStrategy(strategyName).reveal, askingAgentName, strategyName);
}

export function considerAccusation(agent) {
  const strategyName = resolveStrategyName(agent.policy);
  return considerAccusationForMode(agent, resolveStrategy(strategyName).accuse, strategyName);
}

export async function updateMemory(agent, memory, turnEvents) {
  await memory.updateMemory(turnEvents);

  return {
    summary: 'Deterministic memory update completed.'
  };
}
