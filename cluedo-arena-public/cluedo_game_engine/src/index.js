import { Game } from './models/Game.js';
import { startServer } from './server.js';
import { PolicyService } from './services/policyService.js';
import { readFile } from 'fs/promises';

async function runAutomatedGames(numGames) {
  console.log(`Starting ${numGames} automated games...`);
  
  for (let i = 0; i < numGames; i++) {
    console.log(`\n=== Starting Game ${i + 1}/${numGames} ===\n`);
    const game = new Game('spectate');
    await game.initialize();
    
    while (!game.isGameOver()) {
      await game.processTurn();
    }
    
    console.log(`Game ${i + 1} completed`);
  }
}

async function main() {
  
  const runGamesArg = process.argv.find(arg => arg === '--run-games');
  const numGamesArg = process.argv.find(arg => arg.startsWith('--num-games='));
  const heuristicStrategyArg = process.argv.find(arg => arg.startsWith('--heuristic-strategy='));
  const agentStrategiesArg = process.argv.find(arg => arg.startsWith('--agent-strategies='));

  try {
    if (agentStrategiesArg) {
      PolicyService.setAgentStrategies(agentStrategiesArg.split('=')[1].split(','));
    } else if (heuristicStrategyArg) {
      PolicyService.setAgentStrategies(Array(6).fill(heuristicStrategyArg.split('=')[1]));
    } else {
      const configText = await readFile(new URL('../config/agentStrategies.json', import.meta.url), 'utf8');
      const config = JSON.parse(configText);
      PolicyService.setAgentStrategies(config.strategies);
    }
  } catch (error) {
    console.error(`Error setting heuristic strategy: ${error.message}`);
    process.exit(1);
  }

  if (runGamesArg) {
    // Extract number of games from arguments or default to 1
    const numGames = numGamesArg 
      ? parseInt(numGamesArg.split('=')[1], 10) 
      : 1;
      
    await runAutomatedGames(numGames);
  } else {
    // Default behavior - start server
    try {
      await startServer();
      console.log('Server running - Open http://localhost:3000 to play');
    } catch (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }
}

// Only run main if not in test mode
if (process.env.NODE_ENV !== 'test') {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error);
  process.exit(1);
}); 