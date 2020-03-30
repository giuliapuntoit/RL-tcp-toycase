class Connection(object):
    pass

conn = Connection()

states = ['closed', 'listen', 'SYN_rcvd', 'SYN_sent', 'established', 'FIN_wait_1', 'FIN_wait_2', 'closing', 'time_wait', 'close_wait', 'last_ACK']
actions = [
    # client
    'active_open/send_SYN', 'rcv_SYN,ACK/snd_ACK', 'close/snd_FIN', 'rcv_ACK/x', 'rcv_FIN/snd_ACK', 'timeout=2MSL/x',
    # server
    'passive_open/x', 'rcv_SYN/send_SYN,ACK', 'close/snd_FIN',
    # purple
    'send/send_SYN', #'close/x',
         ]
# trigger is the action, source is the state s, dest is the next state s'
# actions are in the format event/response
transitions= [
    # client transactions, green arrows
    {'trigger' : actions[0], 'source' : 'closed', 'dest' : 'SYN_sent'},
    {'trigger' : actions[1], 'source' : 'SYN_sent', 'dest' : 'established'},
    {'trigger' : actions[2], 'source' : 'established', 'dest' : 'FIN_wait_1'},
    {'trigger' : actions[3], 'source' : 'FIN_wait_1', 'dest' : 'FIN_wait_2'},
    {'trigger' : actions[4], 'source' : 'FIN_wait_2', 'dest' : 'time_wait'},
    {'trigger' : actions[5], 'source' : 'time_wait', 'dest' : 'closed'},
    # server transactions, red arrows
    {'trigger' : actions[6], 'source' : 'closed', 'dest' : 'listen'},
    {'trigger' : actions[7], 'source' : 'listen', 'dest' : 'SYN_rcvd'},
    {'trigger' : actions[3], 'source' : 'SYN_rcvd', 'dest' : 'established'},
    {'trigger' : actions[4], 'source' : 'established', 'dest' : 'close_wait'},
    {'trigger' : actions[8], 'source' : 'close_wait', 'dest' : 'last_ACK'},
    {'trigger' : actions[3], 'source' : 'last_ACK', 'dest' : 'closed'},
    # purple arrows
    {'trigger' : actions[9], 'source' : 'listen', 'dest' : 'SYN_sent'},
    #{'trigger' : actions[10], 'source' : 'SYN_sent', 'dest' : 'closed'},
    {'trigger' : actions[7], 'source' : 'SYN_sent', 'dest' : 'SYN_rcvd'},
    {'trigger' : actions[8], 'source' : 'SYN_rcvd', 'dest' : 'FIN_wait_1'},
    {'trigger' : actions[4], 'source' : 'FIN_wait_1', 'dest' : 'closing'},
    {'trigger' : actions[3], 'source' : 'closing', 'dest' : 'time_wait'}
]

# from transitions import Machine
from transitions.extensions import GraphMachine as Machine

machine = Machine(model=conn, states=states, transitions=transitions, initial='closed', ignore_invalid_triggers=True, auto_transitions=True, use_pygraphviz=True)
# ignore_invalid_triggers=True ignore exceptions?
# auto_transitions=False

#machine.get_graph().draw('my_state_diagram.png', prog='dot')

print("Let's simulate a typical connection, following green arrows:")
print("Initial state: " + conn.state) # state of the connection is closed

conn.trigger('active_open/send_SYN')
print(conn.state)

conn.trigger('rcv_ACK/x') # til now if the action is wrong i remain in the same state
print(conn.state)

conn.trigger('rcv_SYN,ACK/snd_ACK')
print(conn.state)

conn.trigger('close/snd_FIN')
print(conn.state)

conn.trigger('rcv_ACK/x')
print(conn.state)

conn.trigger('rcv_FIN/snd_ACK')
print(conn.state)

conn.trigger('timeout=2MSL/x')
print("Final state: " + conn.state)

print("Machine state: " + machine.get_state(conn.state).name)

# In addition to any transitions added explicitly,
# a to_«state»() method is created automatically
# whenever a state is added to a Machine instance.
# This method transitions to the target state
# no matter which state the machine is currently in

# If you desire, you can disable this behavior
# by setting auto_transitions=False in the Machine initializer.

print("Obtaining correct transitions from closed state: ")
print(machine.get_triggers('closed'))
print("Obtaining correct transitions from SYN_sent state: ")
print(machine.get_triggers('SYN_sent'))

###

# SARSA algorithm
import numpy as np
import random

# we need to build the environment. How?

# Defining the different parameters
epsilon = 0.5 # exploration
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initializing the Q-matrix
#Q = np.zeros((observation_space.n, action_space.n)) # those numbers should be states and transitions length
print("N states: ", len(states))
print("N actions: ", len(actions))
Q = np.zeros((len(states), len(actions)))

# Function to choose the next action
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = random.randint(0,len(actions)-1)
        #print("DEBUG action prob ", actions[action])

    else:
        action = np.argmax(Q[state, :])
        #print("DEBUG action max ", actions[action])
    return action

# Function to learn the Q-value
def update(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    #print("DEBUG updating q matrix")
    Q[state, action] = Q[state, action] + alpha * (target - predict)

# Training the learning agent
# Initializing the reward
reward=0

# Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    state1 = states.index(conn.state)
    action1 = choose_action(state1)
    done = False
    print("Episode", episode)
    while t < max_steps:
        #Getting the next state

        conn.trigger(actions[action1])
        state2 = states.index(conn.state)
        #print("DEBUG state1: ", state1, " state2: ", state2)
        tmp_reward = -1
        if state1 != 0 and state2 == 0:
            done = True

        if state1 == 8 and state2 == 0: # check state values
            print("Reward 1000")
            # if i am at the end passing through time_wait (there should be a better implementation for the reward function)
            tmp_reward = 1000
            done = True
        if state1 != 4 and state2 == 4:
            print("Reward 10")
            tmp_reward = 10

        #Choosing the next action
        action2 = choose_action(state2)

        #print("DEBUG action1: ", action1, " action2: ", action2)

        #Learning the Q-value
        update(state1, state2, tmp_reward, action1, action2)

        state1 = state2
        action1 = action2

        #Updating the respective vaLues
        t += 1
        reward = reward + tmp_reward
        #print("DEBUG reward at iteration ", t, " is of R = ", tmp_reward)

        #If at the end of learning process
        if done:
            print("Done time", t)
            break

#Evaluating the performance
print ("Performance : ", reward/total_episodes)

#Visualizing the Q-matrix
print(Q)
# how to visualise the final path?


conn.to_closed()
print("Restarting... returning to state: " + conn.state)
t = 0
max_steps = 30
tot_reward = 0
while t < max_steps:
    state = states.index(conn.state)
    print("State", state)
    max_action = np.argmax(Q[state, :])
    print("Action", actions[max_action])
    previous_state = conn.state
    conn.trigger(actions[max_action])
    tot_reward += Q[state, max_action]
    print("New state", conn.state)
    if previous_state == 'time_wait' and conn.state == 'closed':
        break
    t += 1

print("End with total reward ", tot_reward)

print("Fine")
