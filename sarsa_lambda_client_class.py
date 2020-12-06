import csv
import pathlib
import time
from datetime import datetime

import numpy as np
import random
import matplotlib.pyplot as plt
from transitions.extensions import GraphMachine as Machine

from server import Server
from utilities import Connection, get_states, get_client_actions, get_server_actions, get_transitions, \
    follow_final_policy, compute_reward


class SarsaLambdaFull(object):
    def __init__(self, algorithm="sarsa_lambda", epsilon=0.2, total_episodes=500, max_steps=100, alpha=0.05, gamma=0.9, lam=0.9, follow_policy=False, disable_graphs=False):
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.disable_graphs = disable_graphs
        self.algorithm = algorithm
        self.follow_policy = follow_policy
        if lam is not None:
            self.lam = lam

    # Function to choose the next action
    def choose_action(self, state, actions, Qmatrix):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, len(actions) - 1)
        else:
            # choose random action between the max ones
            action = np.random.choice(np.where(Qmatrix[state, :] == Qmatrix[state, :].max())[0])
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
                Qmatrix[s, a] = Qmatrix[s, a] + self.alpha * delta * Ematrix[s, a]
                Ematrix[s, a] = self.gamma * self.lam * Ematrix[s, a]

    def run(self):
        conn = Connection()
        states = get_states()
        actions = get_client_actions()
        server_actions = get_server_actions()
        transitions = get_transitions(states, actions, server_actions)

        machine = Machine(model=conn, states=states, transitions=transitions, initial='start', ignore_invalid_triggers=True, auto_transitions=True, use_pygraphviz=True)

        # machine.get_graph().draw('client_server_diagram.png', prog='dot')

        current_date = datetime.now()

        log_dir = 'output/log'
        pathlib.Path(log_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5 YY_mm_dd_HH_MM_SS'
        log_filename = current_date.strftime(log_dir + '/' + 'log_' + '%Y_%m_%d_%H_%M_%S' + '.log')

        log_date_filename = 'output/log_date.log'

        output_Q_params_dir = 'output/output_Q_parameters'
        pathlib.Path(output_Q_params_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5
        output_Q_filename = current_date.strftime(
            output_Q_params_dir + '/' + 'output_Q_' + '%Y_%m_%d_%H_%M_%S' + '.csv')
        output_parameters_filename = current_date.strftime(
            output_Q_params_dir + '/' + 'output_parameters_' + '%Y_%m_%d_%H_%M_%S' + '.csv')

        output_dir = 'output/output_csv'
        pathlib.Path(output_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5
        output_filename = current_date.strftime(
            output_dir + '/' + 'output_' + self.algorithm + '_' + '%Y_%m_%d_%H_%M_%S' + '.csv')

        with open(log_date_filename, mode='a') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            output_writer.writerow([current_date.strftime('%Y_%m_%d_%H_%M_%S'), self.algorithm])

        # Write parameters in output_parameters_filename
        with open(output_parameters_filename, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            output_writer.writerow(['algorithm_used', self.algorithm])
            output_writer.writerow(['epsilon', self.epsilon])
            output_writer.writerow(['max_steps', self.max_steps])
            output_writer.writerow(['total_episodes', self.total_episodes])
            output_writer.writerow(['alpha', self.alpha])
            output_writer.writerow(['gamma', self.gamma])

            if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
                output_writer.writerow(['lambda', self.lam])

        # SARSA(lambda) algorithm

        # Initializing the Q-matrix
        if not self.disable_graphs:
            print("N states: ", len(states))
            print("N actions: ", len(actions))
        Q = np.zeros((len(states), len(actions)))
        E = np.zeros((len(states), len(actions)))  # trace for state action pairs

        start_time = time.time()

        x = range(0, self.total_episodes)
        y_timesteps = []
        y_reward = []
        y_cum_reward = []

        x_global = []
        y_global_reward = []

        serv = Server()

        # Write into output_filename the header: Episodes, Reward, CumReward, Timesteps
        with open(output_filename, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(['Episodes', 'Reward', 'CumReward', 'Timesteps'])

        cum_reward = 0
        # Starting Q-learning training
        for episode in range(self.total_episodes):
            print("Episode", episode)
            t = 0
            conn.state = states[1]
            state1 = states.index(conn.state)  # retrieve current state
            # first server perform an action, then client chooses
            print("\tSTARTING FROM STATE", state1)
            done = False
            reward_per_episode = 0

            act = serv.server_action(state1)
            print("\tSERVER ACTION", server_actions[act])
            conn.trigger(server_actions[act])

            state1 = states.index(conn.state)  # retrieve current state
            action1 = self.choose_action(state1, actions, Q)

            while t < self.max_steps:
                state1 = states.index(conn.state)  # retrieve current state
                print("\t\tSTATE1", state1)
                if state1 == 0:
                    print("[DEBUG] state1 is 0")
                    break

                conn.trigger(actions[action1])
                print("\tCLIENT ACTION", actions[action1])
                state2 = states.index(conn.state)
                print("\t\tSTATE2", state2)

                act = serv.server_action(state2)
                print("\tSERVER ACTION", server_actions[act])
                conn.trigger(server_actions[act])
                new_state = states.index(conn.state)
                if new_state != state2:
                    print("\t[DEBUG]: Server changed state from ", state2, "to", new_state)
                    state2 = new_state

                tmp_reward, done = compute_reward(state1, state2, action1)

                # Choosing the next action
                action2 = self.choose_action(state2, actions, Q)

                # Learning the Q-value
                self.update(state1, state2, tmp_reward, action1, action2, states, actions, Q, E)

                # Update log file
                with open(log_filename, "a") as write_file:
                    write_file.write("\nTimestep " + str(t) + " finished.")
                    write_file.write(" Temporary reward: " + str(tmp_reward))
                    write_file.write(" Previous state: " + str(state1))
                    write_file.write(" Current state: " + str(state2))
                    write_file.write(" Performed action: " + str(action1))
                    if self.algorithm != 'qlearning':
                        write_file.write(" Next action: " + str(action2))


                state1 = state2
                action1 = action2

                # Updating the respective vaLues
                t += 1
                reward_per_episode += tmp_reward
                print("\t[DEBUG]: TMP REWARD", tmp_reward)
                print("\t[DEBUG]: REW PER EP", reward_per_episode)

                # If at the end of learning process
                if done:
                    break

            y_timesteps.append(t - 1)
            y_reward.append(reward_per_episode)
            cum_reward += reward_per_episode
            y_cum_reward.append(cum_reward)

            with open(log_filename, "a") as write_file:
                write_file.write("\nEpisode " + str(episode) + " finished.\n")
            with open(output_filename, mode="a") as output_file:
                output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow([episode, reward_per_episode, cum_reward, t - 1])  # Episode or episode+1?

            if self.follow_policy and episode % 20 == 0:
                finPolicy, finReward = follow_final_policy(Q)
                x_global.append(episode)
                y_global_reward.append(finReward)

        # Print and save the Q-matrix inside output_Q_data.csv file
        print("Q MATRIX:")
        print(Q)
        header = ['Q']  # For correct output structure
        for i in actions:
            header.append(i)

        with open(output_Q_filename, "w") as output_Q_file:
            output_Q_writer = csv.writer(output_Q_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            output_Q_writer.writerow(header)
            for index, stat in enumerate(states):
                row = [stat]
                for val in Q[index]:
                    row.append("%.4f" % val)
                output_Q_writer.writerow(row)

        with open(log_filename, "a") as write_file:
            write_file.write("\nTotal time of %s seconds." % (time.time() - start_time))

        # Visualizing the Q-matrix
        if not self.disable_graphs:
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

        optimal = [1, 2, 4, 2, 0]  # client actions. How can i evaluate the policy if that depends on server actions?
        optimal_path = [1, 4, 5, 6, 10, 11, 12, 13, 0]
        sub_optimal_path1 = [1, 4, 5, 6, 10, 14, 15, 13, 0]
        sub_optimal_path2 = [1, 2, 3, 6, 10, 14, 15, 13, 0]
        sub_optimal_path3 = [1, 2, 3, 6, 10, 11, 12, 13, 0]

        finalPolicy, finalReward = follow_final_policy(Q)

        print("Length final policy is", len(finalPolicy))
        print("Final policy is", finalPolicy)
        print("Final reward is", finalReward)
        return x_global, y_global_reward


if __name__ == '__main__':
    x_results, y_rew = SarsaLambdaFull(total_episodes=100, disable_graphs=True).run()
    # print("End of episodes, showing graph...")
    # plt.plot(x_results, y_rew, label="Sarsa lambda full")
    # plt.xlabel('Episodes')
    # plt.ylabel('Final policy reward')
    # plt.title('FULL: Final policy over number of episodes chosen.')
    # plt.legend()
    # plt.show()
    print("DONE.")
