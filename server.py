import numpy as np

def start_1(p):
    if p < 0.1:
        return 1  # 10% send SYN
    else:
        return 0  # 90% do nothing x


def syn_received_3(p):
    if p < 0.1:
        return 0  # 10% do nothing x
    else:
        return 2  # 90% send ACK


def syn_sent_4(p):
    if p < 0.1:
        return 0  # 10% do nothing x
    elif p < 0.2:
        return 1  # 10% send SYN
    else:
        return 3  # 80% send SYN/ACK


def connection_established_6(p):
    if p < 0.3:
        return 4  # 30% send FIN
    else:
        return 0  # 70% do nothing x


def last_ack_9(p):
    if p < 0.1:
        return 0  # 10% do nothing x
    else:
        return 2  # 90% send ACK


def fin_wait_10(p):
    if p < 0.1:
        return 4  # 10% send FIN
    elif p < 0.2:
        return 0  # 10% do nothing x
    else:
        return 2  # 80% send ACK


def fin_wait_received_ack_11(p):
    if p < 0.1:
        return 0  # 10% do nothing x
    else:
        return 4  # 90% send FIN


def closing_15(p):
    if p < 0.1:
        return 0  # 10% do nothing x
    else:
        return 2  # 90% send ACK


class Server(object):
    # server_actions = ["x", "SYN", "ACK", "SYN/ACK", "FIN"]

    def server_action(self, current_state):
        # state 0 is closed/listen, the last one
        # state 1 is start state

        prob = np.random.uniform(0, 1)

        switcher = {
            1: start_1(prob),
            2: 0,  # 100% do nothing x
            3: syn_received_3(prob),
            4: syn_sent_4(prob),
            5: 0,  # 100% do nothing x
            6: connection_established_6(prob),
            7: 0,  # 100% do nothing x
            8: 0,  # 100% do nothing x
            9: last_ack_9(prob),
            10: fin_wait_10(prob),
            11: fin_wait_received_ack_11(prob),
            12: 0,  # 100% do nothing x
            13: 0,  # 100% do nothing x
            14: 0,  # 100% do nothing x
            15: closing_15(prob)
        }

        act = switcher.get(current_state)

        if act is None:
            act = -1

        return act
