import { Memory } from './Memory.js';
import { PolicyService } from '../services/policyService.js';
import { logger } from '../utils/logger.js';

/**
 * Represents an AI agent in the Cluedo game.
 * 
 * Each Agent:
 * - Holds its own cards
 * - Maintains a memory of game events
 * - Uses a deterministic strategy to make decisions
 * - Can make suggestions, respond to challenges, and consider accusations
 */
export class Agent {
  /**
   * Creates a new Agent instance.
   * 
   * @param {string} name - The agent's name/identifier
   * @param {Array<string>} cards - Cards in the agent's hand
   * @param {string} policy - The strategy to use for this agent
   * @param {Object} gameRef - Reference to the game instance
   */
  constructor(name, cards, policy, gameRef) {
    this.name = name;
    this.cards = new Set(cards); // Ensure cards is a Set
    this.policy = policy;
    this.hasLost = false;
    this.game = gameRef;

    // Initialize memory system
    this.memory = new Memory(name);
    this.memory.game = gameRef;
    
    // Initialize with known cards
    cards.forEach(card => this.memory.addKnownCard(card));
    
    // Validate memory system integrity
    if (typeof this.memory.formatMemoryForAgent !== 'function') {
      throw new Error('Memory system corrupted - missing format method');
    }
  }

  /**
   * Makes a strategic suggestion during the agent's turn.
   * 
   * @param {Object} gameState - Current game state information
   * @returns {Promise<Object>} Suggestion with suspect, weapon, room and reasoning
   */
  async makeSuggestion(gameState) {
    try {
      // Ensure gameState has required properties
      const validGameState = {
        currentTurn: gameState.currentTurn || 0,
        availableSuspects: gameState.availableSuspects || [],
        availableWeapons: gameState.availableWeapons || [],
        availableRooms: gameState.availableRooms || [],
        recentHistory: gameState.recentHistory || [],
        activePlayers: gameState.activePlayers || 0
      };

      // Validate cards is a Set
      if (!(this.cards instanceof Set)) {
        this.cards = new Set(this.cards);
      }

      const result = await PolicyService.makeSuggestion(this, validGameState);
      
      // Return only the suggestion part (or error if present)
      if (result.error) {
          return { 
              suspect: result.suspect, 
              weapon: result.weapon, 
              room: result.room, 
              reasoning: result.reasoning, 
              error: result.error 
          };
      } else {
          return { 
              suspect: result.suspect, 
              weapon: result.weapon, 
              room: result.room, 
              reasoning: result.reasoning 
          };
      }
      
    } catch (error) {
      logger.error(`${this.name} agent-level error during makeSuggestion: ${error.message}`, { error });
      // Return fallback suggestion
      return {
        suspect: gameState.availableSuspects ? gameState.availableSuspects[0] : 'Miss Scarlet',
        weapon: gameState.availableWeapons ? gameState.availableWeapons[0] : 'Candlestick',
        room: this.location || 'Lounge', // Use current location or default
        reasoning: `Agent error occurred, using fallback suggestion: ${error.message}`,
        error: error.message
      };
    }
  }

  /**
   * Evaluates a suggestion from another agent and determines which card to show.
   * 
   * When this agent has one or more cards mentioned in another agent's suggestion,
   * this method decides which card to strategically reveal to disprove the suggestion.
   * 
   * @param {Object} suggestion - The suggestion to evaluate (suspect, weapon, room)
   * @returns {Promise<Object>} Challenge result with canChallenge flag and cardToShow
   */
  async evaluateChallenge(suggestion, matchingCards, askingAgentName = null) {
    // Ensure matchingCards is an array of strings, not a Set or single value
    const cardsArray = matchingCards ? 
      (Array.isArray(matchingCards) ? matchingCards : [matchingCards]) : 
      [];
    
    try {
      const result = await PolicyService.evaluateChallenge(this, suggestion, cardsArray, askingAgentName);
      
      // Update memory with challenge result if successful and card shown
      if (result.cardToShow && !result.error) {
        this.memory.addKnownCard(result.cardToShow);
      }
      
      // Return only the challenge result part (or error)
      return { cardToShow: result.cardToShow, reasoning: result.reasoning, error: result.error || null };

    } catch (error) {
      logger.error(`Agent-level challenge evaluation failed for ${this.name}: ${error.message}`, { error });
      // If evaluation fails, don't show any card
      return { cardToShow: null, reasoning: `Agent error in challenge evaluation: ${error.message}`, error: error.message };
    }
  }

  /**
   * Determines whether to make an accusation and with what confidence.
   * 
   * Based on all information gathered through suggestions, challenges,
   * and memory, this method decides if the agent should make a final
   * accusation about the solution.
   * 
   * @returns {Promise<Object>} Decision object containing:
   *                           - shouldAccuse: Boolean
   *                           - accusation: {suspect, weapon, room}
   *                           - confidence: Confidence levels for each element
   *                           - reasoning: Explanation of decision
   */
  async considerAccusation(suggestion, challengeResult) {
    try {
      // Ensure game reference exists
      if (!this.game) {
        logger.error(`Game reference missing for ${this.name}`);
        throw new Error('Game reference not set');
      }
      
      logger.debug(`Formatting memory for ${this.name} accusation consideration...`);
      const memoryState = await this.memory.formatMemoryForAgent();
      logger.debug(`Memory formatted for ${this.name}.`);

      const gameState = {
        currentTurn: this.game.currentTurn,
        knownCards: this.cards,
        memory: memoryState,
        currentSuggestion: suggestion,
        currentChallengeResult: challengeResult
      };
      
      logger.debug(`Calling PolicyService.considerAccusation for ${this.name}...`);
      const result = await PolicyService.considerAccusation(this, gameState);
      
      // Return only the decision part (or error)
      return { 
          shouldAccuse: result.shouldAccuse, 
          accusation: result.accusation, 
          confidence: result.confidence, 
          reasoning: result.reasoning, 
          error: result.error || null 
      };

    } catch (error) {
      logger.error(`Agent-level accusation consideration failed for ${this.name}: ${error.message}`, { error });
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        confidence: { suspect: 0, weapon: 0, room: 0 },
        reasoning: `Agent error in accusation consideration: ${error.message}`,
        error: error.message
      };
    }
  }
  

  /**
   * Updates the agent's memory with events from the most recent turn.
   * 
   * @param {Array} turnEvents - Array of events from the most recent turn
   * @returns {Promise<{memory: Memory}>} Updated memory
   */
  async updateMemory(turnEvents) {
    try {
      const updateResult = await PolicyService.updateMemory(this, this.memory, turnEvents);

      return { 
          summary: updateResult.summary, 
          error: updateResult.error || null
      };

    } catch (error) {
      logger.error(`Agent-level error during memory update for ${this.name}: ${error.message}`, { error });
      // Return empty/error results on error
      return { 
          summary: `(Agent error in memory update: ${error.message})`, 
          error: error.message 
      };
    }
  }

  /**
   * Marks this agent as having lost the game.
   */
  setLost() {
    this.hasLost = true;
  }

  /**
   * Gets a formatted representation of this agent's memory state.
   *
   * @returns {Promise<Object>} The agent's current memory and policy
   */
  async getMemoryState() {
    const memoryState = await this.memory.formatMemoryForAgent();
    return {
      ...memoryState,
      policy: this.policy
    };
  }

  /**
   * Processes this agent's turn.
   * Performs maintenance operations on memory.
   * 
   * @returns {Promise<void>}
   */
  async processTurn() {
    // Perform memory maintenance (cleanup, optimization)
    this.memory.maintain();
  }

  /**
   * Moves the agent to a new location (room).
   * For now, this is a placeholder. In a real game, this would involve
   * dice rolls, pathfinding, or player choice.
   * It ensures the agent has *some* location assigned.
   *
   * @param {Array<string>} availableRooms - List of all possible rooms.
   */
  move(availableRooms) {
    if (!this.location) {
      this.location = availableRooms[Math.floor(Math.random() * availableRooms.length)];
      logger.debug(`${this.name} assigned starting location: ${this.location}`);
    } else {
      const destinations = availableRooms.filter(room => room !== this.location);
      if (destinations.length > 0) {
        this.location = destinations[Math.floor(Math.random() * destinations.length)];
      }
      logger.debug(`${this.name} moved to location: ${this.location}`);
    }
  }
} 