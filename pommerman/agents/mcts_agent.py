import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
import numpy as np
from pommerman import characters
from pommerman import utility
import random
from pommerman.envs.v0 import Pomme
from copy import deepcopy
import time



"""
This code is a modified version of 
https://github.com/tambetm/pommerman-baselines/blob/master/mcts_selfplay/mcts_selfplay_agent.py
"""



c_param = 1.4


class Util(object):
    @staticmethod
    def argmax_tiebreaking_axis1(Q, action_mask):
        # find the best action with random tie-breaking
        mask = ((Q == np.max(Q, axis=1, keepdims=True)) * action_mask).astype(bool)
        return np.array([np.random.choice(np.flatnonzero(m)) for m in mask])

    @staticmethod
    def any_lst_equal(lst, values):
        '''Checks if list are equal'''
        return any([lst == v for v in values])


class Model(object):
    def __init__(self, env, num_agents, num_actions, debug=False):
        self.env = env
        self.obs = self.env.get_observations()
        self.last_obs = None
        self.agents_pos_rec = [[] for _ in range(num_agents)]
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.debug = debug
        self.own_agent_id = None
        self.done = None

    def add_own_agent_id(self, agent_id):
        self.own_agent_id = agent_id

    @staticmethod
    def get_next_position(position, direction):
        '''Returns the next position coordinates'''
        x, y = position
        if direction == constants.Action.Right:
            return (x, y + 1)
        elif direction == constants.Action.Left:
            return (x, y - 1)
        elif direction == constants.Action.Down:
            return (x + 1, y)
        elif direction == constants.Action.Up:
            return (x - 1, y)
        elif direction == constants.Action.Stop:
            return (x, y)
        raise constants.InvalidAction("We did not receive a valid direction.")

    @staticmethod
    def position_on_board(board, position):
        '''Determines if a position is on the board'''
        x, y = position
        return all([len(board) > x, len(board[0]) > y, x >= 0, y >= 0])

    @staticmethod
    def position_is_passable(board, position, can_kick=True):
        '''Determines if a position can be passed'''
        #return all([
        #    any([
        #        position_is_agent(board, position),
        #        position_is_powerup(board, position),
        #        position_is_passage(board, position)
        #    ]), not position_is_enemy(board, position, enemies)
        #])
        powerups = [constants.Item.ExtraBomb.value, constants.Item.IncrRange.value, constants.Item.Kick.value]
        return any([board[position] in powerups,
                    board[position] == constants.Item.Passage.value,
                    all([board[position] == constants.Item.Bomb.value, can_kick])])

    @staticmethod
    def get_valid_actions(board, position, ammo=1, can_kick=True):
        """
        Filters out useless actions.
        :param board:
        :param position:
        :param ammo:
        :param can_kick:
        :return: List of legal actions
        """
        def filter_invalid_directions(board, my_position, directions, can_kick):
            ret = []
            for direction in directions:
                position = Model.get_next_position(my_position, direction)
                if Model.position_on_board(board, position) and Model.position_is_passable(board, position, can_kick):
                    ret.append(direction.value)
            return ret
        directions = [constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
        valid_actions = filter_invalid_directions(board, position, directions, can_kick)
        valid_actions.append(constants.Action.Stop.value)
        if ammo > 0:
            valid_actions.append(constants.Action.Bomb.value)
        return sorted(valid_actions)

    def get_action_mask(self):
        """
        Creates a mask with shape (NUM_AGENTS, NUM_ACTIONS) where valid actions of all agents are marked with 1s
        and invalid (i.e. useless) actions are marked with 0s.
        :param obs: Dictionary with observations of our agent.
        """
        observations = self.obs

        #  Determine action mask only for alive agents
        agent_values = list(np.arange(constants.Item.Agent0.value, constants.Item.Agent0.value + self.num_agents))
        action_mask = np.zeros((self.num_agents, self.num_actions)).astype(int)
        for a, obs in zip(agent_values, observations):
            if a in obs['alive']:
                agent_mask = obs['board'] == a
                if np.any(agent_mask):  # if agent is found in view  # TODO: use v0
                    pos = (np.where(agent_mask)[0][0],
                           np.where(agent_mask)[1][0])
                    if pos == obs['position']:
                        ammo = obs['ammo']
                        can_kick = obs['can_kick']
                    else:
                        ammo = 1  # Assume that enemy agents have some ammunition
                        can_kick = True  # Assume that enemy agents can kick bombs
                    valid_actions = Model.get_valid_actions(obs['board'], pos, ammo, can_kick)
                    action_mask[a-10, valid_actions] = 1
                else:
                    action_mask[a-10, :] = 1
            else:
                action_mask[a-10, :] = 1  # TODO: why every action allowed when dead?
        return action_mask

    def get_rewards(self):
        if self.env._game_type in [constants.GameType.TeamRadio, constants.GameType.Team]:
            return self._get_team_rewards()
        else:
            raise Exception('Game type not implemented: ', self.env._game_type)

    def _get_team_rewards(self):
        obs = self.obs

        step_count = self.env._step_count
        max_steps = self.env._max_steps
        agents = self.env._agents

        # print("getting reward")
        # print(obs)
        if self.debug:
            print('Step count:', step_count)

        alive_agents = [num for num, agent in enumerate(agents) if agent.is_alive]

        r_list = [0, 0, 0, 0]
        if alive_agents == [0, 2]:
            if self.debug:
                print('Team [0, 2] wins.')
            r_list = [1, -1, 1, -1]
        elif alive_agents == [0]:
            if self.debug:
                print('Team [0, 2] wins. agent 0 alive')
            #r_list = [1, -1, 0.5, -1]
            r_list = [1, -1, 0, -1]
        elif alive_agents == [2]:
            if self.debug:
                print('Team [0, 2] wins. agent 2 alive')
            #r_list = [0.5, -1, 1, -1]
            r_list = [0, -1, 1, -1]
        elif alive_agents == [1, 3]:
            if self.debug:
                print('Team [1, 3] wins.')
            r_list = [-1, 1, -1, 1]
        elif alive_agents == [1]:
            if self.debug:
                print('Team [1, 3] wins. agent 1 alive')
            #r_list = [-1, 1, -1, 0.5]
            r_list = [-1, 1, -1, 0]
        elif alive_agents == [3]:
            if self.debug:
                print('Team [1, 3] wins. agent 3 alive')
            #r_list = [-1, 0.5, -1, 1]
            r_list = [-1, 0, -1, 1]

        # if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        #     # Team [0, 2] wins.
        #     return [1, -1, 1, -1]
        # elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
        #     # Team [1, 3] wins.
        #     return [-1, 1, -1, 1]

        elif step_count >= max_steps:
            if self.debug:
                print('Game over by max steps')
            # return [-1] * 4
            r_list = [-1, -1, -1, -1]
            # for reward shaping: tie game , all get 0
            #r_list = [0, 0, 0, 0]
        elif len(alive_agents) == 0:
            if self.debug:
                print('Everyone is dead')
            # return [-1] * 4
            r_list = [-1, -1, -1, -1]
            # for reward shaping: tie game , all get 0
            #r_list = [0, 0, 0, 0]
        #elif self.last_obs is not None:
        #    if self.debug:
        #        print('Reward shaping')
        #    # return [0] * 4
        #    for i in range(len(r_list)):
        #        if self.last_obs[i]['can_kick'] == False and obs[i]['can_kick'] == True:
        #            r_list[i] += 0.02
        #        if self.last_obs[i]['ammo'] < obs[i]['ammo'] :
        #            r_list[i] += 0.01 # * obs[i]['ammo']
        #        if self.last_obs[i]['blast_strength'] < obs[i]['blast_strength'] :
        #            r_list[i] += 0.01 #* obs[i]['blast_strength']

        #    if self.debug:
        #        print('Reward after powerups:', r_list)

        #    #Going to a cell not in a 121-length FIFO queue gets 0.001
        #    curr_agents_pos = [ag['position'] for ag in obs]
        #    for i, p in enumerate(self.agents_pos_rec):
        #        if curr_agents_pos[i] not in p:
        #            #print("get readr")
        #            r_list[i] += 0.001
        #        if len(p) == 121:
        #            #print('del p0')
        #            p.pop(0)
        #        p.append(curr_agents_pos[i])

        #    # if 3 agents are alive: 012, 023, 013, 123
        #    ## team [0,2] get 0.5 and team [1,3] get -0.5 : 012, 023,
        #    if Util.any_lst_equal(alive_agents, [[0, 1, 2], [0, 2, 3] ]):
        #        if len(self.last_obs[0]['alive']) == 4:
        #            r_list[0] += 0.5
        #            r_list[2] += 0.5

        #            r_list[1] += -0.5
        #            r_list[3] += -0.5
        #    ## team [1,3] get 0.5 and team [0,2] get -0.5 : 013, 123,
        #    elif  Util.any_lst_equal(alive_agents, [[0, 1, 3], [1, 2, 3] ]):
        #        if len(self.last_obs[0]['alive']) == 4:
        #            r_list[0] += -0.5
        #            r_list[2] += -0.5

        #            r_list[1] += 0.5
        #            r_list[3] += 0.5

        #    # if 2 agents are alive : 01, 03 , 12, 23, each team one agent dead,
        #    elif Util.any_lst_equal(self.last_obs[0]['alive'],[[10,11,12],  [10, 12, 13]]):
        #        if len(alive_agents) == 2:

        #            r_list[0] += 0.5
        #            r_list[2] += 0.5

        #            r_list[1] += -0.5
        #            r_list[3] += -0.5
        #    elif Util.any_lst_equal(self.last_obs[0]['alive'],[[10,11,13],  [11, 12, 13]]):
        #        if len(alive_agents) == 2:

        #            r_list[0] += -0.5
        #            r_list[2] += -0.5

        #            r_list[1] += 0.5
        #            r_list[3] += 0.5
        #    if self.debug:
        #        print('Reward team checks:', r_list)
        #else:
        #    if self.debug:
        #        print('First observation')
        else:
            for num, agent in enumerate(agents):
                if agent.is_alive:
                    r_list[num] = 1
                else:
                    r_list[num] = -1
        if not self.done:
            r_list = [0, 0, 0, 0]

        if self.last_obs is not None:
            if self.debug:
                print('Reward shaping')
            # return [0] * 4
            for i in range(len(r_list)):
                if self.last_obs[i]['can_kick'] == False and obs[i]['can_kick'] == True:
                    r_list[i] += 0.5
                if self.last_obs[i]['ammo'] < obs[i]['ammo'] :
                    r_list[i] += 0.3 # * obs[i]['ammo']
                if self.last_obs[i]['blast_strength'] < obs[i]['blast_strength'] :
                    r_list[i] += 0.4 #* obs[i]['blast_strength']
            #alive_agents = [num for num, agent in enumerate(agents) if agent.is_alive]
        return r_list

    def closest_nonvisible(self, visible_mask, agent_position):
        board = self.env._board.copy()

        # invert board
        board[visible_mask] = constants.Item.Fog.value
        # location should be non-visible and a passage
        possible_locations = np.where((board != constants.Item.Fog.value)
                                      & (board == constants.Item.Passage.value))
        possible_locations = np.array(list(zip(possible_locations[0], possible_locations[1])))

        # find smallest distance
        return possible_locations[
            np.argmin(
                np.linalg.norm(possible_locations - agent_position, axis=1)
            )
        ]

    def remove_agent(self, agent_id, agent_position):
        if self.last_obs is None:
            return

        #prev_val = self.last_obs[agent_id]['board'][agent_position]
        #if prev_val >= 10:
            # TODO: what do we input when we don't know the previous value?

        # simple
        prev_val = constants.Item.Passage.value

        self.env._board[agent_position] = prev_val

    # wrapper function around env.step(), stores last obs
    def step(self, actions):
        if type(actions) is not list:
            actions = actions.tolist()

        self.last_obs = self.obs
        # (actions always converted to list, otherwise type error in v2)
        obs, rewards, done, info = self.env.step(actions)
        self.obs = obs

        # Modify 'done' variable: Change it such that done = True as soon as our agent is dead.
        new_done = any([self.own_agent_id + 10 not in obs[0]['alive'], done])
        self.done = new_done

        # Use custom reward function:
        new_rewards = self.get_rewards()

        #print("new_rewards: ", new_rewards)
        #if new_done:
        #    print("obs['alive']", obs[0]['alive'])
        #    print("new_final_rewards: ", new_rewards)
        #print("")

        return obs, new_rewards, new_done, info

    # TODO: move all env related functions here (save state / load state)


class MCTSNode(object):
    def __init__(self, num_agents, num_actions, action_mask=None):
        self.num_agents = num_agents
        self.num_actions = num_actions
        # values for 6 actions
        # Q: (Total simulation reward / Total number of visits) of every child node for every agent
        self.Q = np.zeros((self.num_agents, self.num_actions))
        # W: Total simulation reward of every child node for every agent
        self.W = np.zeros((self.num_agents, self.num_actions))
        # N: Total number of visits of every child node for every agent
        self.N = np.zeros((self.num_agents, self.num_actions), dtype=np.uint32)
        if action_mask is not None:
            self.action_mask = action_mask
        else:
            self.action_mask = np.ones(self.Q.shape)

    def actions(self):
        """
        Returns the action for each agent based on the UCT function value
        """
        # U: Exploration component of UCT function
        U = c_param * np.sqrt(np.sum(self.N, axis=1, keepdims=True)) / (1 + self.N)
        #U = (c_param * np.sqrt(np.sum(self.N, axis=1, keepdims=True)) / (1 + self.N)) * self.action_mask
        U[self.action_mask == 0] = - np.inf
        #print("Q + U: ", self.Q + U)
        return Util.argmax_tiebreaking_axis1(self.Q + U, self.action_mask)

    def update(self, actions, rewards):
        """
        Updates the statistics of every visited child node for each agent
        """
        assert len(actions) == len(rewards)
        self.W[range(self.num_agents), actions] += rewards
        self.N[range(self.num_agents), actions] += 1
        idx = (self.N != 0)
        self.Q[idx] = self.W[idx] / self.N[idx]

    def probs(self, temperature=1):
        """
        If temperature != 0, returns a distribution over actions for each agent, where
        actions that were performed more often have higher probability.
        If temperature == 0, the probability for the most frequent action is set to 1
        and the probabilities of all other actions are set to 0.
        """
        if temperature == 0:
            p = np.zeros(self.N.shape)
            idx = Util.argmax_tiebreaking_axis1(self.N, self.action_mask)
            p[range(self.num_agents), idx] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt, axis=1, keepdims=True)


class MCTSAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.debug = False

        self.num_agents = 4
        self.num_actions = len(constants.Action)
        self.tree = {}

    def get_agent_id(self, obs):
        agent_value = obs['board'][obs['position']]
        agent_id = agent_value - 10
        return agent_id

    #def get_game_type(self, type_idx):
    #    # types = pommerman.REGISTRY
    #    types = [constants.GameType.FFA, constants.GameType.Team,
    #             constants.GameType.TeamRadio, constants.GameType.OneVsOne]
    #    type_strings = ['TODO', 'TODO', 'PommeRadio-v2', 'OneVsOne-v0']
    #    game_type = types[type_idx - 1]
    #    type_str = type_strings[type_idx - 1]
    #    return game_type, type_str

    #def get_num_agents(self, type_idx):
    #    if type_idx == 4:
    #        return 2
    #    else:
    #        return 4

    # TODO: move this into model
    def update_internal_env(self, obs):
        if obs['step_count'] == 0:
            # Create fully observable internal environment of type 'PommeTeam-v0':
            self.env = pommerman.make('PommeTeam-v0', [SimpleAgent() for _ in range(self.num_agents)])
            self.env.reset()
            self.model = Model(self.env, self.num_agents, self.num_actions, debug=self.debug)

        # Retrieve game_type:
        #self.game_type, type_str = self.get_game_type(obs['game_type'])
        # Retrieve number of agents:
        #if self.num_agents is None:
        #self.num_agents = self.get_num_agents(obs['game_type'])
        # Retrieve id of our agent:
        self.agent_id = self.get_agent_id(obs)
        # Create agent_list:
        #agents = [SimpleAgent() for _ in range(self.num_agents)]
        # Create environment based on game_type and agent_list, then reset it:

        # Add own agent_id to Model:
        if self.model.own_agent_id is None:
            self.model.add_own_agent_id(self.agent_id)

        # Synchronize internal environment with partial observation.
        #print("internal env before: ", self.env._board)
        #print("")
        #print("obs: ", obs['board'])
        #print("")

        # Get visible area of the board:
        visible_mask = obs['board'] != constants.Item.Fog.value
        # Get positions of visible area of the board:
        visible_locations = list(zip(np.where(visible_mask)[0], np.where(visible_mask)[1]))

        #print(self.env._board)
        #print(obs['board'])

        # Set env._board to visible part of obs['board']:
        self.env._board[visible_mask] = obs['board'][visible_mask]

        # Update env._agents:
        for agent_id, agent in enumerate(self.env._agents):
            #print("position of agent {}: {}".format(agent_id+10, agent.position))
            #print("agent {} is alive (internal): {}".format(agent_id+10, agent.is_alive))
            # Start with default parameters
            ammo = 1
            is_alive = True
            blast_strength = constants.DEFAULT_BLAST_STRENGTH
            can_kick = False
            # Set is_alive to False if agent is already dead:
            if agent_id + 10 not in obs['alive']:
                is_alive = False
            #print("agent {} is alive (real): {}".format(agent_id + 10, is_alive))

            # Update position of visible agents:
            agent_mask = obs['board'] == agent_id + 10
            if np.any(agent_mask):  # True if agent is visible. Set agent's start_position to observed position
                # make sure agent is removed from nonvisible area
                if agent.position not in visible_locations:
                    self.model.remove_agent(agent_id, agent.position)

                position = (np.where(agent_mask)[0][0],
                            np.where(agent_mask)[1][0])
                agent.set_start_position(position)
                # Set ammo, blast_strength, can_kick for our own agent:
                if agent_id == self.agent_id:
                    blast_strength = obs['blast_strength']
                    can_kick = obs['can_kick']
                    ammo = obs['ammo']
            elif is_alive:  # If agent is not visible, set his start_position to his current hypothetic position
                # careful: agent should not be placed in visible field!
                if agent.position not in visible_locations:
                    # real world says agent is alive, but he died in the virtual world
                    #if self.env._board[agent.position] != agent_id + 10:
                    #    pass
                    agent.set_start_position(agent.position)
                else:
                    # get closest location to agent position in non-visible area
                    closest_nonvisible = self.model.closest_nonvisible(visible_mask, agent.position)
                    # workaround: put agent into visible mask,
                    # so the next agent can no longer be placed into the same position
                    visible_mask[closest_nonvisible] = True
                    agent.set_start_position(closest_nonvisible)
            elif not is_alive:
                # clean up field
                self.model.remove_agent(agent_id, agent.position)

            #print("start_position of agent {}: {}".format(agent_id+10, agent.start_position))
            agent.reset(ammo, is_alive, blast_strength, can_kick)

        #agent_positions = [agent.position for agent in self.env._agents if agent.is_alive]
        #uniques, double_index = np.unique(agent_positions, return_inverse=True, axis=0)
        #if len(uniques) != len(agent_positions):
        #    import pdb; pdb.set_trace()

        # Update env._bombs in visible area:
        # Remove existing bombs in visible area:
        for bomb in list(self.env._bombs):
            if bomb.position in visible_locations:
                self.env._bombs.remove(bomb)
        # Insert bombs from obs['bombs']:
        bomb_positions = list(zip(np.where(obs['bomb_life'])[0], np.where(obs['bomb_life'])[1]))
        for b in range(len(bomb_positions)):
            pos = bomb_positions[b]
            self.env._bombs.append(characters.Bomb(
                bomber=characters.Bomber(agent_id=random.choice([i for i in range(self.num_agents)])),
                position=pos,
                life=int(obs['bomb_life'][pos]),
                blast_strength=int(obs['bomb_blast_strength'][pos]),
                moving_direction=int(obs['bomb_moving_direction'][pos])))

        # Update env._items in visible area:
        # Remove existing items in visible area:
        for loc in list(self.env._items.keys()):
            if loc in visible_locations:
                self.env._items.pop(loc)
        # Get position of wooden fields in visible area:
        wood_positions = list(zip(np.where(obs['board'] == 2)[0], np.where(obs['board'] == 2)[1]))
        # Determine number of items to place behind the wooden fields. Use ratio num_items/num_wood
        # for 2vs2 game_type specified in constants.py.
        num_items = int(5. / 9. * len(wood_positions))
        # Randomly place num_items items behind wooden fields, where the items to place are also chosen randomly.
        item_types = [constants.Item.ExtraBomb.value, constants.Item.IncrRange.value, constants.Item.Kick.value]
        item_positions = random.sample(wood_positions, num_items)  # sample without replacement
        for pos in item_positions:
            self.env._items[pos] = random.choice(item_types)  # sample with replacement

        # Update env._flames in visible area:
        # Remove existing flames in visible area:
        for flame in list(self.env._flames):
            if flame.position in visible_locations:
                self.env._flames.remove(flame)
        # Insert flames from obs['flame_life']:
        flame_positions = list(zip(np.where(obs['flame_life'])[0], np.where(obs['flame_life'])[1]))
        for f in range(len(flame_positions)):
            pos = flame_positions[f]
            self.env._flames.append(characters.Flame(
                position=pos,
                life=int(obs['flame_life'][pos])))

        # Update self.env._step_count:
        self.env._step_count = int(obs['step_count'])

        #print("internal env after: ", self.env._board)

    def search(self, observation, temperature=1, duration=0.1):
        """
        Performs MCTS search.
        :param observation: Dictionary of partial observations for our agent.
        :param temperature: Whether to return a distribution over actions or a one-hot-coded vector
                            for each agent
        :param duration: Maximum allowed duration for executing 'act' method.
        """
        t_start = time.time()
        #print("obs board: ", observation['board'])

        # Update internal environment from agent's observation:
        self.update_internal_env(observation)

        # Create action_mask that highlights useful and useless moves, given the initial observation
        action_mask = self.model.get_action_mask()

        # Save initial state of internal environment as game tree root from where all iterations will start
        root = self.env.get_json_info()
        #print("Updated env board: ", self.env._board)
        #print("")

        # Set initial game state to return to after each simulation
        self.env._init_game_state = root
        del root['intended_actions']
        root = str(root)
        # Insert root into game tree
        self.tree[root] = MCTSNode(self.num_agents, self.num_actions, action_mask=action_mask)

        # Create array of possible actions for all agents
        av_actions = np.expand_dims(np.arange(self.num_actions), axis=0).repeat(self.num_agents, 0)

        i = 0
        while time.time() - t_start < duration:
            # set game state to initial game state
            obs = self.env.reset()
            # serialize game state
            state = self.env.get_json_info()
            # remove 'intended_actions' as they are not reset by env.reset(),
            # and a state already contained in the game tree would thus not
            # be recognized.
            del state['intended_actions']
            state = str(state)

            trace = []  # list of triplets of type (node, actions, rewards) visited during a rollout
            done = False

            dead_agents = np.array([i for i in range(10, 14) if i not in observation['alive']]).astype(int)
            # fetch rewards so we know which agents are alive. Reward of alive agents is 0 when
            # the game is still running.
            #rewards = np.array(self.env._get_rewards())
            # rewards now come from our model and not env (forward model)
            #rewards = np.array(self.model.get_rewards())  # [rewardAgent0, ..., rewardAgentN]
            while not done:
                if state in self.tree:  # game tree traversal
                    node = self.tree[state]
                    # choose actions of all agents based on Q + U
                    actions = node.actions()
                    # use Stop action for all dead agents to reduce tree size
                    actions[dead_agents-10] = constants.Action.Stop.value
                    #actions[rewards != 0] = constants.Action.Stop.value
                    # step environment forward
                    obs, rewards, done, info = self.model.step(actions)
                    rewards = np.array(rewards, dtype=np.float32)
                    trace.append((node, actions, rewards))
                    # fetch next state
                    state = self.env.get_json_info()
                    del state['intended_actions']
                    state = str(state)
                else:  # Rollout + Expansion
                    # Rollout:
                    # Choose and remember actions of rollout starting node:
                    #start_actions = np.array([np.random.choice(av_actions[r]) for r in range(av_actions.shape[0])])
                    start_action_mask = self.model.get_action_mask()
                    normalized_action_mask = start_action_mask / np.sum(start_action_mask,
                                                                        axis=1, keepdims=True)
                    start_actions = np.array([np.random.choice(av_actions[r], p=normalized_action_mask[r])
                                              for r in range(av_actions.shape[0])])
                    # use Stop action for all dead agents to reduce tree size
                    start_actions[dead_agents - 10] = constants.Action.Stop.value
                    #start_actions[rewards != 0] = constants.Action.Stop.value
                    actions = start_actions
                    while not done:
                        # step environment forward
                        obs, rewards, done, info = self.model.step(actions)
                        rewards = np.array(rewards, dtype=np.float32)
                        # Choose random actions for every agent:
                        actions = np.array([np.random.choice(av_actions[r]) for r in range(av_actions.shape[0])])
                        # use Stop action for all dead agents to reduce tree size
                        actions[dead_agents - 10] = constants.Action.Stop.value
                        #actions[rewards != 0] = constants.Action.Stop.value
                    # Add rollout starting node to game tree
                    node = MCTSNode(self.num_agents, self.num_actions, action_mask=start_action_mask)
                    self.tree[state] = node
                    # Append (rollout starting node, start_actions, final_rewards) to trace:
                    trace.append((node, start_actions, rewards))

            # Update tree nodes with rollout results
            for node, actions, rews in reversed(trace):
                # use the reward of the last timestep where it was non-null
                rewards[rews != 0] = rews[rews != 0]
                node.update(actions, rewards)
            i += 1

        # Return to initial game state
        self.env.set_json_info()
        self.env._init_game_state = None
        if self.debug:
            print("root.Q: ", self.tree[root].Q)
            print("root.W: ", self.tree[root].W)
            print("root.N: ", self.tree[root].N)
        # return action probabilities
        return self.tree[root].probs(temperature), i

    def act(self, obs, action_space):
        #t_0 = time.time()

        # Get distribution over next actions for each agent:
        pi, num_iters = self.search(obs, temperature=0, duration=0.084)
        #print("probs: ", pi)
        # Choose most promising action for each agent:
        actions = [np.random.choice(self.num_actions, p=pi[r]) for r in range(pi.shape[0])]
        if self.debug:
            print("actions: ", actions)
        # Step internal environment with chosen actions of all agents:
        _, _, _, _ = self.model.step(actions)
        # Set new state of internal environment as its initial game state:
        new_state = self.env.get_json_info()
        self.env._init_game_state = new_state

        #self.env.render()
        #duration = time.time() - t_0
        #print("duration: {}, iterations: {}".format(duration, num_iters))

        # Return chosen action of our agent to real environment:
        return actions[self.agent_id]
