# Cluedo AI

A deterministic multi-agent simulation of the classic deduction game Cluedo
(Clue). Agents use configurable heuristic strategies for suggestions, card
reveals, accusations, and memory updates.

## Features

- Deterministic and probabilistic heuristic strategies
- Per-agent card and deduction memory
- Interactive web visualization
- Automated games and result tracking
- Optional Python utilities for data processing and evaluation

## Project structure

- `cluedo_game_engine/`: Node.js game engine
  - `src/models/`: game, agent, memory, and result models
  - `src/services/heuristicBot.js`: heuristic strategy implementations
  - `src/services/policyService.js`: policy interface used by agents
  - `public/`: web visualization
- `scripts/`: Python data-processing and evaluation utilities
- `tests/`: JavaScript and Python tests

## Requirements

- Node.js 14+
- npm 6+
- Python 3.8+ for the Python utilities

No API keys, external model services, or environment file are required.

## Quick start

Install the Node.js dependencies:

```bash
cd cluedo_game_engine
npm install
```

Start the web application:

```bash
npm start
```

Open `http://localhost:3000`.

## Heuristic strategies

The available strategies are combinations of suggestion, reveal, and
accusation policies:

```text
random-random-certain
first-random-certain
cand-random-certain
focused-random-certain
bluff-random-certain
cand-smart-certain
cand-random-gamble2
cand-random-gamble3
focused-smart-gamble2
```

`first-random-certain` selects the first suspect, weapon, and room in the
configured card order that the agent has not eliminated from the envelope. It
therefore skips cards in its own hand and cards revealed to it.

Select a strategy for an automated run:

```bash
node src/index.js --heuristic-strategy=focused-smart-gamble2 --run-games --num-games=3
```

The same option can be used when starting the web application:

```bash
npm start -- --heuristic-strategy=bluff-random-certain
```

To assign a different strategy to each agent, provide six comma-separated
strategies in this order:

```text
Red, Blue, Green, Yellow, Purple, Orange
```

For example:

```bash
node src/index.js \
  --agent-strategies=focused-smart-gamble2,cand-random-certain,bluff-random-certain,random-random-certain,cand-smart-certain,focused-random-certain \
  --run-games \
  --num-games=3
```

Each strategy is assigned to the corresponding colored agent. The option must
contain exactly six valid strategy names.

The same assignment is stored by default in
`cluedo_game_engine/config/agentStrategies.json`:

```json
{
  "strategies": [
    "focused-smart-gamble2",
    "cand-random-certain",
    "bluff-random-certain",
    "random-random-certain",
    "cand-smart-certain",
    "focused-random-certain"
  ]
}
```

Edit that file and run `npm start` or `npm run run-games:heuristic`. The
`--agent-strategies` command-line option overrides the file for one run.

The strategy interface is defined by
`cluedo_game_engine/src/services/policyService.js`. A future non-heuristic
policy can implement the same four operations without changing the game loop:

1. Make a suggestion
2. Choose a card to reveal
3. Decide whether to accuse
4. Update memory

## Running tests

JavaScript tests:

```bash
cd cluedo_game_engine
npm test
```

Python tests:

```bash
pytest
```

## Running the standalone Python pilot

The standalone strategy simulator can be run with:

```bash
python cluedo_pilot.py
```

## Contributing

Contributions are welcome. Please submit a pull request with tests for
behavioral changes.
