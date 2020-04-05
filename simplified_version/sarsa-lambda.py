class Connection(object):
    pass


conn = Connection()

#states = ['closed', 'listen', 'SYN_rcvd', 'SYN_sent', 'established', 'FIN_wait_1', 'FIN_wait_2', 'closing', 'time_wait', 'close_wait', 'last_ACK']
states = ['start', 'SYN_sent', 'established', 'FIN_wait_1', 'FIN_wait_2', 'time_wait','closed']

actions = [
    # client
    'active_open/send_SYN', 'rcv_SYN,ACK/snd_ACK', 'close/snd_FIN', 'rcv_ACK/x', 'rcv_FIN/snd_ACK', 'timeout=2MSL/x',
    # server
#    'passive_open/x', 'rcv_SYN/send_SYN,ACK', 'close/snd_FIN',
    # purple
#    'send/send_SYN', 'close/x',
         ]
# trigger is the action, source is the state s, dest is the next state s'
# actions are in the format event/response
transitions= [
    # client transactions, green arrows
    {'trigger' : actions[0], 'source' : 'start', 'dest' : 'SYN_sent'},
    {'trigger' : actions[1], 'source' : 'SYN_sent', 'dest' : 'established'},
    {'trigger' : actions[2], 'source' : 'established', 'dest' : 'FIN_wait_1'},
    {'trigger' : actions[3], 'source' : 'FIN_wait_1', 'dest' : 'FIN_wait_2'},
    {'trigger' : actions[4], 'source' : 'FIN_wait_2', 'dest' : 'time_wait'},
    {'trigger' : actions[5], 'source' : 'time_wait', 'dest' : 'closed'},
    # server transactions, red arrows
#    {'trigger' : actions[6], 'source' : 'start', 'dest' : 'listen'},
#    {'trigger' : actions[7], 'source' : 'listen', 'dest' : 'SYN_rcvd'},
#    {'trigger' : actions[3], 'source' : 'SYN_rcvd', 'dest' : 'established'},
#    {'trigger' : actions[4], 'source' : 'established', 'dest' : 'close_wait'},
#    {'trigger' : actions[8], 'source' : 'close_wait', 'dest' : 'last_ACK'},
#    {'trigger' : actions[3], 'source' : 'last_ACK', 'dest' : 'closed'},
    # purple arrows
#    {'trigger' : actions[9], 'source' : 'listen', 'dest' : 'SYN_sent'},
#    {'trigger' : actions[10], 'source' : 'SYN_sent', 'dest' : 'closed'},
#    {'trigger' : actions[7], 'source' : 'SYN_sent', 'dest' : 'SYN_rcvd'},
#    {'trigger' : actions[8], 'source' : 'SYN_rcvd', 'dest' : 'FIN_wait_1'},
#    {'trigger' : actions[4], 'source' : 'FIN_wait_1', 'dest' : 'closing'},
#    {'trigger' : actions[3], 'source' : 'closing', 'dest' : 'time_wait'}
]

# from transitions import Machine
from transitions.extensions import GraphMachine as Machine

machine = Machine(model=conn, states=states, transitions=transitions, initial='start', ignore_invalid_triggers=True, auto_transitions=True, use_pygraphviz=True)

#machine.get_graph().draw('my_state_diagram.png', prog='dot')

###

# SARSA algorithm
import numpy as np
import random
import time
import matplotlib.pyplot as plt


# Defining the different parameters
epsilon = 0.3 # small exploration, big exploitation
total_episodes = 5000
max_steps = 1000
alpha = 0.005 # smaller than before
gamma = 0.95
# new paramter lambda for sarsa(lambda)
lam = 0.9

# Initializing the Q-matrix
print("N states: ", len(states))
print("N actions: ", len(actions))
Q = np.zeros((len(states), len(actions)))
E = np.zeros((len(states), len(actions)))  # trace for state action pairs

# Function to choose the next action
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = random.randint(0,len(actions)-1)
    else:
        #actions2 = np.argmax(Q[state, :]) #they might be more than one
        #action = actions2[random.randint(0,len(actions2)-1)]
        #choose random action between the max ones
        action=np.random.choice(np.where(Q[state, :] == Q[state, :].max())[0])
    return action

# Function to update the Q-value matrix and the Eligibility matrix
def update(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    delta = target - predict
    E[state, action] = E[state, action] + 1 #perché +1?
    # for all s, a
    for s in range(len(states)):
        for a in range(len(actions)):
            Q[s,a] = Q[s,a] + alpha * delta * E[s,a]
            E[s,a] = gamma * lam * E[s,a]

# Training the learning agent

start_time = time.time()

x = range(0, total_episodes)
y_timesteps = []
y_reward = []

# Starting the SARSA(lambda) learning
for episode in range(total_episodes):
    print("Episode", episode)
    t = 0
    conn.state = 'start'
    state1 = states.index(conn.state)
    action1 = choose_action(state1)
    done = False
    reward_per_episode = 0

    while t < max_steps:
        #Getting the next state

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

        #Choosing the next action
        action2 = choose_action(state2)

        #print("Action1:", action1, ". Action2:", action2)

        # In SARSA(lambda) the update function is different
        # Updates both the Q and the E matrix
        update(state1, state2, tmp_reward, action1, action2)

        state1 = state2
        action1 = action2

        #Updating the respective vaLues
        t += 1
        reward_per_episode += tmp_reward

        #If at the end of learning process
        if done:
            break
    y_timesteps.append(t-1)
    y_reward.append(reward_per_episode)



#Visualizing the Q-matrix
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
print("Restarting... returning to state: " + conn.state)
t = 0

while t < 10:
    state = states.index(conn.state)
    #print("[DEBUG] state:", state)
    max_action = np.argmax(Q[state, :])
    print("Action to perform is", actions[max_action])
    previous_state = conn.state
    conn.trigger(actions[max_action])
    print("New state", conn.state)
    if previous_state == 'time_wait' and conn.state == 'closed':
        break
    t += 1

print("End")

# time: 22.515238761901855 seconds
