export class Memory {
  constructor(agentId) {
    this.agentId = agentId;
    this.knownCards = new Set();
    this.playerKnownCards = new Map([[agentId, new Set()]]);
    this.eliminatedCards = new Set();
    this.confirmedSolutionCards = new Set();
    this.playerNegativeConstraints = new Map();
    this.memoryHistory = [];
    this.lastUpdated = Date.now();
  }

  addKnownCard(card) {
    this.knownCards.add(card);
    this.recordKnownCardForPlayer(this.agentId, card);
    this.eliminatedCards.delete(card);
  }

  recordKnownCardForPlayer(playerName, card) {
    if (!this.playerKnownCards.has(playerName)) {
      this.playerKnownCards.set(playerName, new Set());
    }
    this.playerKnownCards.get(playerName).add(card);
  }

  eliminateCard(card) {
    this.eliminatedCards.add(card);
  }

  doesPlayerNotHaveCard(playerName, cardName) {
    const constraints = this.playerNegativeConstraints.get(playerName);
    return constraints ? constraints.has(cardName) : false;
  }

  formatMemoryForAgent() {
    const playerNames = [...new Set([
      ...(this.game?.agents?.map(agent => agent.name) || []),
      ...this.playerKnownCards.keys(),
      this.agentId
    ])];
    const knownCardsByPlayer = Object.fromEntries(
      playerNames.map(playerName => [
        playerName,
        Array.from(this.playerKnownCards.get(playerName) || [])
      ])
    );

    return {
      knownInformation: {
        myCards: Array.from(this.knownCards),
        eliminatedCards: Array.from(this.eliminatedCards),
        confirmedSolutionCards: Array.from(this.confirmedSolutionCards),
        knownCardsByPlayer
      },
      turnHistory: this.memoryHistory.map(entry => {
        let eventString = `Turn ${entry.turnNumber || '?'}:`;
        if (entry.suggestion) {
          eventString += ` ${entry.activeAgent || 'Agent'} suggested ${entry.suggestion.suspect}, ${entry.suggestion.weapon}, ${entry.suggestion.room}.`;
        }
        if (entry.challengeResult?.canChallenge) {
          eventString += ` ${entry.challengeResult.challengingAgent} showed a card.`;
        } else if (entry.suggestion) {
          eventString += ' No challenge.';
        }
        return eventString;
      })
    };
  }

  recordTurnEvents(turnNumber, activeAgent, suggestion, challengeResult) {
    this.memoryHistory.push({
      turnNumber: turnNumber || this.memoryHistory.length + 1,
      timestamp: Date.now(),
      activeAgent,
      suggestion: suggestion ? {
        suspect: suggestion.suspect,
        weapon: suggestion.weapon,
        room: suggestion.room
      } : null,
      challengeResult: challengeResult ? {
        canChallenge: challengeResult.canChallenge,
        challengingAgent: challengeResult.challengingAgent
      } : null
    });
    this.lastUpdated = Date.now();
  }

  async updateMemory(turnEvents) {
    let suggestionCards = [];

    for (const event of turnEvents) {
      if (event.type === 'suggestion') {
        suggestionCards = [event.suspect, event.weapon, event.room];
      } else if (event.type === 'cardShown' && event.card) {
        if (event.from === this.agentId) {
          this.addKnownCard(event.card);
        } else if (event.to === this.agentId) {
          this.eliminateCard(event.card);
          this.recordKnownCardForPlayer(event.from, event.card);
        }
      } else if (
        event.type === 'challenge' &&
        !event.canChallenge &&
        event.canInferUnchallengedSolution &&
        suggestionCards.length === 3
      ) {
        const suggestingAgentCards = new Set(event.suggestingAgentCards || []);
        suggestingAgentCards.forEach(card => {
          if (!this.knownCards.has(card)) this.eliminateCard(card);
        });
        suggestionCards
          .filter(card => !suggestingAgentCards.has(card))
          .forEach(card => this.confirmedSolutionCards.add(card));
      }
    }
  }

  reset() {
    this.memoryHistory = [];
    this.playerKnownCards = new Map([[this.agentId, new Set(this.knownCards)]]);
    this.eliminatedCards.clear();
    this.confirmedSolutionCards.clear();
    this.playerNegativeConstraints.clear();
    this.lastUpdated = Date.now();
  }

  maintain() {
    if (this.memoryHistory.length > 100) {
      this.memoryHistory = this.memoryHistory.slice(-100);
    }
    this.lastUpdated = Date.now();
  }
}
