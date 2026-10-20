import { Agent } from './Agent.js';

export class HumanAgent extends Agent {
  constructor(name, cards, gameRef) {
    super(name, cards, 'human', gameRef);
    this.isHuman = true;
  }
}
