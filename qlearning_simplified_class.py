import time
import numpy as np
import random
import matplotlib.pyplot as plt
# from transitions import Machine
from transitions.extensions import GraphMachine as Machine

class Connection(object):
    pass

class QlearningSimplified(object):

    def __init__(self, epsilon=0.3, total_episodes=5000, max_steps=1000, alpha=0.005, gamma=0.95, disable_graphs=False):
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.disable_graphs = disable_graphs

    # Function to choose the next action
    def choose_action(self, state, actions, Qmatrix):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0,len(actions)-1)
        else:
            #choose random action between the max ones
            action=np.random.choice(np.where(Qmatrix[state, :] == Qmatrix[state, :].max())[0])
        return action

    # Function to learn the Q-value
    def update(self, state, state2, reward, action, Qmatrix):
        predict = Qmatrix[state, action]
        maxQ = np.amax(Qmatrix[state2, :]) #find maximum value for the new state
        target = reward + self.gamma * maxQ
        Qmatrix[state, action] = Qmatrix[state, action] + self.alpha * (target - predict)

    def run(self):
        conn = Connection()

        states = ['start', 'SYN_sent', 'established', 'FIN_wait_1', 'FIN_wait_2', 'time_wait','closed']

        actions = [
            # client
            'active_open/send_SYN', 'rcv_SYN,ACK/snd_ACK', 'close/snd_FIN', 'rcv_ACK/x', 'rcv_FIN/snd_ACK', 'timeout=2MSL/x',
                 ]
        # actions are in the format event/response
        transitions= [
            # client transactions, green arrows
            {'trigger' : actions[0], 'source' : 'start', 'dest' : 'SYN_sent'},
            {'trigger' : actions[1], 'source' : 'SYN_sent', 'dest' : 'established'},
            {'trigger' : actions[2], 'source' : 'established', 'dest' : 'FIN_wait_1'},
            {'trigger' : actions[3], 'source' : 'FIN_wait_1', 'dest' : 'FIN_wait_2'},
            {'trigger' : actions[4], 'source' : 'FIN_wait_2', 'dest' : 'time_wait'},
            {'trigger' : actions[5], 'source' : 'time_wait', 'dest' : 'closed'},
        ]

        machine = Machine(model=conn, states=states, transitions=transitions, initial='start', ignore_invalid_triggers=True, auto_transitions=True, use_pygraphviz=True)

        #machine.get_graph().draw('my_state_diagram.png', prog='dot')

        # Q-learning algorithm

        # Initializing the Q-matrix
        if self.disable_graphs == False:
            print("N states: ", len(states))
            print("N actions: ", len(actions))
        Q = np.zeros((len(states), len(actions)))

        start_time = time.time()

        x = range(0, self.total_episodes)
        y_timesteps = []
        y_reward = []

        # Starting the SARSA learning
        for episode in range(self.total_episodes):
            if self.disable_graphs == False:
                print("Episode", episode)
            t = 0
            conn.state = 'start'
            state1 = states.index(conn.state)
            done = False
            reward_per_episode = 0

            while t < self.max_steps:
                #Getting the next state

                action1 = self.choose_action(state1, actions, Q)
                conn.trigger(actions[action1])
                state2 = states.index(conn.state)
                #print("From state", state1, "to state", state2)
                tmp_reward = -1

                if state1 == 5 and state2 == 6:
                    #print("Connection closed correctly")
                    tmp_reward = 1000
                    done = True
                if state1 == 1 and state2 == 2:
                    #print("Connection estabilished")
                    tmp_reward = 10

                #print("Action1:", action1, ". Action2:", action2)

                #Learning the Q-value
                self.update(state1, state2, tmp_reward, action1, Q)

                state1 = state2

                #Updating the respective vaLues
                t += 1
                reward_per_episode += tmp_reward

                #If at the end of learning process
                if done:
                    break
            y_timesteps.append(t-1)
            y_reward.append(reward_per_episode)


        #Visualizing the Q-matrix
        if self.disable_graphs == False:
            print(actions)
            print(Q)

            print("--- %s seconds ---" % (time.time() - start_time))

            plt.plot(x, y_reward)
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.title('Rewards per episode')

            plt.show()

            plt.plot(x, y_timesteps)
            plt.xlabel('Episodes')
            plt.ylabel('Timestep to end of the episode')
            plt.title('Timesteps per episode')

            plt.show()


        #conn.to_closed()
        conn.state = 'start'
        if self.disable_graphs == False:
            print("Restarting... returning to state: " + conn.state)
        t = 0
        finalPolicy = []
        finalReward = 0
        optimal = [0, 1, 2, 3, 4, 5]
        while t < 10:
            state = states.index(conn.state)
            #print("[DEBUG] state:", state)
            max_action = np.argmax(Q[state, :])
            if self.disable_graphs == False:
                print("Action to perform is", actions[max_action])
            previous_state = conn.state
            conn.trigger(actions[max_action])
            state1 = states.index(previous_state)
            state2 = states.index(conn.state)
            tmp_reward = -1

            if state1 == 5 and state2 == 6:
                # print("Connection closed correctly")
                tmp_reward = 1000
                done = True
            if state1 == 1 and state2 == 2:
                # print("Connection established")
                tmp_reward = 10
            finalReward += tmp_reward

            if self.disable_graphs == False:
                print("New state", conn.state)
            if previous_state == 'time_wait' and conn.state == 'closed':
                break
            t += 1

        print("Length final policy is", len(finalPolicy))
        print("Final policy is", finalPolicy)
        if len(finalPolicy) == 6 and np.array_equal(finalPolicy, optimal):
            return True, finalReward
        else:
            return False, finalReward

if __name__ == '__main__':
    optimalPolicy, obtainedReward = QlearningSimplified(total_episodes=20).run()
    if optimalPolicy:
        print("Optimal policy was found with reward", obtainedReward)
    else:
        print("No optimal policy reached with reward", obtainedReward)
