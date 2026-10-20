import * as heuristicBot from './heuristicBot.js';

export const POLICY_STRATEGIES = heuristicBot.listStrategies();

export class PolicyService {
  static #agentStrategies = null;

  static setStrategy(strategyName) {
    heuristicBot.setStrategy(strategyName);
  }

  static getStrategy() {
    return heuristicBot.getStrategy();
  }

  static setAgentStrategies(strategyNames) {
    if (!Array.isArray(strategyNames) || strategyNames.length === 0) {
      throw new Error('At least one agent strategy must be provided.');
    }
    strategyNames.forEach(strategyName => heuristicBot.setStrategy(strategyName));
    this.#agentStrategies = strategyNames.map(strategyName => {
      heuristicBot.setStrategy(strategyName);
      return heuristicBot.getStrategy();
    });
  }

  static getAgentStrategies(agentCount) {
    if (this.#agentStrategies) {
      if (this.#agentStrategies.length !== agentCount) {
        throw new Error(`Expected ${agentCount} agent strategies, received ${this.#agentStrategies.length}.`);
      }
      return [...this.#agentStrategies];
    }
    return Array(agentCount).fill(this.getStrategy());
  }

  static async makeSuggestion(agent, gameState) {
    return heuristicBot.makeSuggestion(agent, gameState);
  }

  static async evaluateChallenge(agent, suggestion, cards, askingAgentName = null) {
    return heuristicBot.evaluateChallenge(agent, cards, askingAgentName);
  }

  static async considerAccusation(agent) {
    return heuristicBot.considerAccusation(agent);
  }

  static async updateMemory(agent, memory, turnEvents) {
    return heuristicBot.updateMemory(agent, memory, turnEvents);
  }
}
