import time
import numpy as np
import random
import matplotlib.pyplot as plt
# from transitions import Machine
from transitions.extensions import GraphMachine as Machine

def start_1(p):
    if p < 0.1:
        return 1 # 10% send SYN
    else:
        return 0 # 90% do nothing x

def syn_received_3(p):
    if p < 0.1:
        return 0 # 10% do nothing x
    else:
        return 2 # 90% send ACK

def syn_sent_4(p):
    if p < 0.1:
        return 0 # 10% do nothing x
    elif p < 0.2:
        return 1 # 10% send SYN
    else:
        return 3 # 80% send SYN/ACK

def connection_established_6(p):
    if p < 0.3:
        return 4 # 30% send FIN
    else:
        return 0 # 70% do nothing x

def last_ack_9(p):
    if p < 0.1:
        return 0 # 10% do nothing x
    else:
        return 2 # 90% send ACK

def fin_wait_10(p):
    if p < 0.1:
        return 4 # 10% send FIN
    else:
        return 2 # 90% send ACK

def fin_wait_received_ack_11(p):
    if p < 0.1:
        return 0 # 10% do nothing x
    else:
        return 4 # 90% send FIN

def closing_15(p):
    if p < 0.1:
        return 0 # 10% do nothing x
    else:
        return 2 # 90% send ACK

class Server(object):
    # server_actions = ["x", "SYN", "ACK", "SYN/ACK", "FIN"]

    def server_action(self, current_state):
        # state 0 is closed/listen, the last one
        # state 1 is start state

        prob = np.random.uniform(0, 1)

        switcher = {
            1: start_1(prob),
            2: 0, # 100% do nothing x
            3: syn_received_3(prob),
            4: syn_sent_4(prob),
            5: 0, # 100% do nothing x
            6: connection_established_6(prob),
            7: 0, # 100% do nothing x
            8: 0, # 100% do nothing x
            9: last_ack_9(prob),
            10: fin_wait_10(prob),
            11: fin_wait_received_ack_11(prob),
            12: 0, # 100% do nothing x
            13: 0, # 100% do nothing x
            14: 0, # 100% do nothing x
            15: closing_15(prob)
        }

        act = switcher.get(current_state)

        if act == None:
            act = -1

        return act

class Connection(object):
    pass

class SarsaLambdaFull(object):
    def __init__(self, epsilon=0.3, total_episodes=5000, max_steps=1000, alpha=0.005, gamma=0.95, lam = 0.9, disable_graphs=False):
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
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

    # Function to update the Q-value matrix and the Eligibility matrix
    def update(self, state, state2, reward, action, action2, states, actions, Qmatrix, Ematrix):
        predict = Qmatrix[state, action]
        target = reward + self.gamma * Qmatrix[state2, action2]
        delta = target - predict
        Ematrix[state, action] = Ematrix[state, action] + 1
        # for all s, a
        for s in range(len(states)):
            for a in range(len(actions)):
                Qmatrix[s,a] = Qmatrix[s,a] + self.alpha * delta * Ematrix[s,a]
                Ematrix[s,a] = self.gamma * self.lam * Ematrix[s,a]

    def run(self):
        conn = Connection()

        states = ['closed_listen_0', 'start_1', 'closed_listen_rcvd_SYN_2', 'SYN_rcvd_3', 'SYN_sent_4', 'SYN_sent_rcvd_SYN_ACK_5',
                  'established_6', 'established_rcvd_FIN_7', 'close_wait_8', 'last_ACK_9', 'FIN_wait_10',
                  'FIN_wait_2_rcvd_ACK_11', 'FIN_wait_rcvd_ACK_FIN_12', 'time_wait_13', 'FIN_wait_rcvd_FIN_14', 'closing_15']

        actions = ["x", "SYN", "ACK", "SYN/ACK", "FIN"]
        server_actions = ["server_x", "server_SYN", "server_ACK", "server_SYN/ACK", "server_FIN"]
        # actions are in the format event/response
        transitions= [
            # x transactions keep the state, do I have to define those triggers?
            # client actions
            {'trigger' : actions[1], 'source' : states[1], 'dest' : states[4]},
            {'trigger' : actions[2], 'source' : states[5], 'dest' : states[6]},
            {'trigger' : actions[3], 'source' : states[2], 'dest' : states[3]},
            {'trigger' : actions[4], 'source' : states[3], 'dest' : states[10]},
            {'trigger' : actions[4], 'source' : states[6], 'dest' : states[10]},
            {'trigger' : actions[2], 'source' : states[7], 'dest' : states[8]},
            {'trigger' : actions[4], 'source' : states[8], 'dest' : states[9]},
            {'trigger' : actions[2], 'source' : states[12], 'dest' : states[13]},
            {'trigger' : actions[0], 'source' : states[13], 'dest' : states[0]},
            {'trigger' : actions[2], 'source' : states[14], 'dest' : states[15]},
            # server actions
            {'trigger' : server_actions[1], 'source' : states[1], 'dest' : states[2]},
            {'trigger' : server_actions[2], 'source' : states[4], 'dest' : states[2]},
            {'trigger' : server_actions[3], 'source' : states[4], 'dest' : states[5]},
            {'trigger' : server_actions[2], 'source' : states[3], 'dest' : states[6]},
            {'trigger' : server_actions[4], 'source' : states[6], 'dest' : states[7]},
            {'trigger' : server_actions[2], 'source' : states[9], 'dest' : states[0]},
            {'trigger' : server_actions[2], 'source' : states[10], 'dest' : states[11]},
            {'trigger' : server_actions[4], 'source' : states[10], 'dest' : states[14]},
            {'trigger' : server_actions[4], 'source' : states[11], 'dest' : states[12]},
            {'trigger' : server_actions[2], 'source' : states[15], 'dest' : states[13]},
            {'trigger' : server_actions[0], 'source' : states[13], 'dest' : states[0]}
        ]

        machine = Machine(model=conn, states=states, transitions=transitions, initial='start', ignore_invalid_triggers=True, auto_transitions=True, use_pygraphviz=True)

        machine.get_graph().draw('client_server_diagram.png', prog='dot')

        # SARSA(lambda) algorithm

        # Initializing the Q-matrix
        if self.disable_graphs == False:
            print("N states: ", len(states))
            print("N actions: ", len(actions))
        Q = np.zeros((len(states), len(actions)))
        E = np.zeros((len(states), len(actions)))  # trace for state action pairs

        start_time = time.time()

        x = range(0, self.total_episodes)
        y_timesteps = []
        y_reward = []

        x_global = []
        y_global_reward = []

        serv = Server()

        # Starting Q-learning training
        for episode in range(self.total_episodes):
            if self.disable_graphs == False:
                print("Episode", episode)
            t = 0
            conn.state = states[1]
            state1 = 1
            # first server perform an action, then client chooses
            print("From state", 1)
            done = False
            reward_per_episode = 0

            act = serv.server_action(state1)
            print("Server does action", server_actions[act])
            conn.trigger(server_actions[act])

            state1 = states.index(conn.state) # retrieve current state
            print("Goes to state1", state1)

            action1 = self.choose_action(state1, actions, Q)

            while t < self.max_steps:

                conn.trigger(actions[action1])
                print("Client does action", actions[action1])
                state2 = states.index(conn.state)
                print("Goes to state2", state2)
                tmp_reward = -1

                if state2 == 0:
                    #print("Connection closed correctly")
                    tmp_reward = 1000
                if state1 != 6 and state2 == 6: # anche state1 == 5?
                    #print("Connection estabilished")
                    tmp_reward = 10
                if state2 == 0:
                    done = True

                #Choosing the next action
                action2 = self.choose_action(state2, actions, Q)

                #Learning the Q-value
                self.update(state1, state2, tmp_reward, action1, action2, states, actions, Q, E)


                act = serv.server_action(state2)
                print("Server does action", server_actions[act])
                conn.trigger(server_actions[act])


                state1 = states.index(conn.state)
                print("Goes to state1", state1)

                # choose action based on the new state
                action1 = self.choose_action(state1, actions, Q)

                #Updating the respective vaLues
                t += 1
                reward_per_episode += tmp_reward

                #If at the end of learning process
                if done:
                    break

            y_timesteps.append(t-1)
            y_reward.append(reward_per_episode)

            if episode % 20 == 0:
                conn.state = states[1]
                if self.disable_graphs == False:
                    print("Restarting... returning to state: " + conn.state)
                t = 0
                finPolicy = []
                finReward = 0
                while t < 10:
                    state = states.index(conn.state)
                    print("State", state)
                    act = serv.server_action(state)
                    print("Server does action", server_actions[act])
                    conn.trigger(server_actions[act])
                    state = states.index(conn.state)
                    print("Goes to state", state)
                    max_action = np.argmax(Q[state, :])
                    finPolicy.append(max_action)
                    if self.disable_graphs == False:
                        print("Action to perform is", actions[max_action])
                    previous_state = conn.state
                    conn.trigger(actions[max_action])
                    print("End in state", conn.state)
                    state1 = states.index(previous_state)
                    state2 = states.index(conn.state)
                    tmp_reward = -1
                    if state2 == 0:
                        #print("Connection closed correctly")
                        tmp_reward = 1000
                    if state1 != 6 and state2 == 6:
                        #print("Connection estabilished")
                        tmp_reward = 10
                    finReward += tmp_reward
                    if self.disable_graphs == False:
                        print("New state", conn.state)
                    if conn.state == states[0]:
                        break
                    t += 1

                x_global.append(episode)
                y_global_reward.append(finReward)

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

        conn.state = states[1]
        if self.disable_graphs == False:
            print("Restarting... returning to state: " + conn.state)
        t = 0
        finalPolicy = []
        finalReward = 0
        optimal = [1, 2, 4, 2, 0] # client actions. How can i evaluate the policy if that depends on server actions?
        optimal_path = [1, 4, 5, 6, 10, 11, 12, 13, 0]
        sub_optimal_path1 = [1, 4, 5, 6, 10, 14, 15, 13, 0]
        sub_optimal_path2 = [1, 2, 3, 6, 10, 14, 15, 13, 0]
        sub_optimal_path3 = [1, 2, 3, 6, 10, 11, 12, 13, 0]

        while t < 10:
            state = states.index(conn.state)
            print("State", state)
            act = serv.server_action(state)
            print("Server does action", server_actions[act])
            conn.trigger(server_actions[act])
            state = states.index(conn.state)
            print("Goes to state", state)
            max_action = np.argmax(Q[state, :])
            finalPolicy.append(max_action)
            if self.disable_graphs == False:
                print("Action to perform is", actions[max_action])
            previous_state = conn.state
            conn.trigger(actions[max_action])
            print("End in state", conn.state)
            state1 = states.index(previous_state)
            state2 = states.index(conn.state)
            tmp_reward = -1
            if state2 == 0: # o state 0?
                #print("Connection closed correctly")
                tmp_reward = 1000
            if state1 != 6 and state2 == 6: # anche state1 == 5?
                #print("Connection estabilished")
                tmp_reward = 10
            finalReward += tmp_reward
            if self.disable_graphs == False:
                print("New state", conn.state)
            if conn.state == states[0]:
                break
            t += 1

        print("Length final policy is", len(finalPolicy))
        print("Final policy is", finalPolicy)
        print("Final reward is", finalReward)
        return x_global, y_global_reward


if __name__ == '__main__':
    x, y_reward = SarsaLambdaFull(total_episodes=2000, lam=0.3, disable_graphs=False).run()
    print("End of episodes, showing graph...")
    plt.plot(x, y_reward, label="Sarsa full")
    plt.xlabel('Episodes')
    plt.ylabel('Final policy reward')
    plt.title('FULL: Final policy over number of episodes chosen.')
    plt.legend()
    plt.show()
