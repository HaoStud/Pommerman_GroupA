'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from tqdm import tqdm


def main():
    '''Simple function to bootstrap a game.
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    #print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        #agents.DockerAgent("pommerman/mcts-agent", port=12345),
        agents.MCTSAgent(),
        agents.SimpleAgent(),
        agents.MCTSAgent(),
        #agents.SimpleAgent(),
        agents.SimpleAgent(),

        #agents.RandomAgent(),
    ]
    env = pommerman.make('PommeRadio-v2', agent_list)

    rewards = np.zeros(len(agent_list))
    num_episodes = 10

    # Run the episodes just like OpenAI Gym
    for i_episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        while not done:
        #for i in range(100):
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            #print("state: ", state[0])
        rewards += reward
        print('Episode {} finished. Rewards: {}'.format(i_episode, reward))
    print("Rewards after {} episodes: {}".format(num_episodes, rewards))
    env.close()


if __name__ == '__main__':
    main()
