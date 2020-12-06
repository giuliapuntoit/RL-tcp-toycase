from transitions.extensions import GraphMachine as Machine
import numpy as np

from server import Server


class Connection(object):
    pass


def get_states():
    states = ['closed_listen_0', 'start_1', 'closed_listen_rcvd_SYN_2', 'SYN_rcvd_3', 'SYN_sent_4',
                  'SYN_sent_rcvd_SYN_ACK_5',
                  'established_6', 'established_rcvd_FIN_7', 'close_wait_8', 'last_ACK_9', 'FIN_wait_10',
                  'FIN_wait_2_rcvd_ACK_11', 'FIN_wait_rcvd_ACK_FIN_12', 'time_wait_13', 'FIN_wait_rcvd_FIN_14',
                  'closing_15']
    return states


def get_client_actions():
    actions = ["x", "SYN", "ACK", "SYN/ACK", "FIN"]
    return actions


def get_server_actions():
    server_actions = ["server_x", "server_SYN", "server_ACK", "server_SYN/ACK", "server_FIN"]
    return server_actions


def get_transitions(states, actions, server_actions):
    # actions are in the format event/response
    transitions = [
        # x transactions keep the state, do I have to define those triggers?
        # client actions
        {'trigger': actions[1], 'source': states[1], 'dest': states[4]},
        {'trigger': actions[2], 'source': states[5], 'dest': states[6]},
        {'trigger': actions[3], 'source': states[2], 'dest': states[3]},
        {'trigger': actions[4], 'source': states[3], 'dest': states[10]},
        {'trigger': actions[4], 'source': states[6], 'dest': states[10]},
        {'trigger': actions[2], 'source': states[7], 'dest': states[8]},
        {'trigger': actions[4], 'source': states[8], 'dest': states[9]},
        {'trigger': actions[2], 'source': states[12], 'dest': states[13]},
        {'trigger': actions[0], 'source': states[13], 'dest': states[0]},
        {'trigger': actions[2], 'source': states[14], 'dest': states[15]},
        # server actions
        {'trigger': server_actions[1], 'source': states[1], 'dest': states[2]},
        {'trigger': server_actions[2], 'source': states[4], 'dest': states[2]},
        {'trigger': server_actions[3], 'source': states[4], 'dest': states[5]},
        {'trigger': server_actions[2], 'source': states[3], 'dest': states[6]},
        {'trigger': server_actions[4], 'source': states[6], 'dest': states[7]},
        {'trigger': server_actions[2], 'source': states[9], 'dest': states[0]},
        {'trigger': server_actions[2], 'source': states[10], 'dest': states[11]},
        {'trigger': server_actions[4], 'source': states[10], 'dest': states[14]},
        {'trigger': server_actions[4], 'source': states[11], 'dest': states[12]},
        {'trigger': server_actions[2], 'source': states[15], 'dest': states[13]},
        {'trigger': server_actions[0], 'source': states[13], 'dest': states[0]}
    ]
    return transitions


def compute_reward(state1, state2, action1):
    tmp_reward = -1
    done = False
    if state2 == 0 and state1 == 13:  # o state 0?
        # print("Connection closed correctly")
        tmp_reward += 200
    elif state2 == 0 and state1 == 12:
        tmp_reward += 200
    elif state2 == 0 and state1 == 15:
        tmp_reward += 200
    elif state1 != 6 and state2 == 6:  # anche state1 == 5?
        # print("Connection estabilished")
        tmp_reward += 10
    if action1 != 0:
        tmp_reward += -1
    if state2 == 0:
        done = True
    return tmp_reward, done


def follow_final_policy(Q):
    finalPolicy = []
    finalReward = 0

    serv = Server()
    conn = Connection()

    states = get_states()
    actions = get_client_actions()
    server_actions = get_server_actions()
    transitions = get_transitions(states, actions, server_actions)

    machine = Machine(model=conn, states=states, transitions=transitions, initial='start',
                      ignore_invalid_triggers=True, auto_transitions=True, use_pygraphviz=True)

    conn.state = states[1]
    t = 0

    while t < 10:  # 15???
        state = states.index(conn.state)
        print("STATE", state)
        act = serv.server_action(state)
        print("SERVER ACTION", server_actions[act])
        conn.trigger(server_actions[act])
        state = states.index(conn.state)
        print("STATE", state)
        max_action = np.argmax(Q[state, :])
        finalPolicy.append(max_action)
        print("CLIENT ACTION", actions[max_action])
        previous_state = conn.state
        conn.trigger(actions[max_action])
        print("NEXT STATE", conn.state)
        state1 = states.index(previous_state)
        state2 = states.index(conn.state)

        tmp_reward, done = compute_reward(state1, state2, max_action)
        finalReward += tmp_reward

        if done:
            break
        t += 1
    return finalPolicy, finalReward
