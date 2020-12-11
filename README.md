# RL for TCP toycase

This is a simplified integration of RL algorithms into the TCP protocol.
Our goal is to train a RL agent to exchange TCP messages with a toy server in order to correctly establish a TCP connection with the server.

We integrate 3 RL algorithms:

* SARSA
* SARSA(λ)
* Q-learning

These algorithms work into an environment which simulates a TCP client and a TCP server. 
The TCP client is the RL agent, whereas the TCP server is probabilistic.

## Motivation

We developed this toycase scenario to understand how RL algorithms can be used inside a protocol-related context.
Also, we implement multiple RL algorithms to understand the meaning of each parameter and highlight the main differences of structures and performance of these algorithms. 

## How to use?

In ``_version`` directories each script can be executed individually. 
Also all ``*_client_class.py`` files can be executed directly. These last files can be executed together through the ``main.py`` script.

**Note**: you will need to install some modules with pip

### Structure

General structure of directories:

* ``prove``: contains trails with State and Connection classes.
* ``simplified_version``: contains sarsa, sarsa(λ) and Q-learning for a simplified version of TCP machine state, only with states and actions specific for opening and closing a TCP connection.
* ``full_version``: contains sarsa, sarsa(λ) and Q-learning for TCP machine state, with no distinction among client and server.
* ``integrated_version`` should be able to run all algorithms and have all results in same plot

All these directories model the scenario in the following figure, with no distinction between a TCP client and a TCP server:

<p align="center"><img src="https://github.com/giuliapuntoit/RL-tcp-toycase/blob/master/images/tcp.png" height="800"></p>


``*_client_class.py`` files contains RL for the client considered as a RL agent, with a probabilistic server as modelled in this image:

<p align="center"><img src="https://github.com/giuliapuntoit/RL-tcp-toycase/blob/master/images/tcp.png" height="800"></p>

These classes rely on:

* ``server.py``: models the behaviour of the probabilistic server
* ``utilities.py``: contains methods for retrieving all states, optimal paths, all actions, all transitions among states, the reward and a method to follow the best policy found. 

The project can be run from the ``main.py`` to use multiple algorithms for the ``*_client_class.py`` series.

#### Output

Throughout the entire learning process, the ``*_client_class.py`` files collect data into external files, inside the ``output`` directory.

All files for one execution of the learning process are identified by the current date in the format ``%Y_%m_%d_%H_%M_%S``.

Inside ``output`` directory:

* ``output_Q_parameters``: this directory contains info collected before and after the learning process. Before the process starts, all values for the configurable parameters are saved into file ``output_parameters_<date>.csv``: information about the path to learn, the optimal policy, the algorithm chosen, the number of episodes, the values for α, γ, λ and ε. If one wants to reproduce an execution of the learning process, all the parameters saved inside this file allow for repeating the learning process using the exact same configuration. Then, at the end of each iteration of the outer loop, the Q matrix is written and updated inside a file ``output_Q_<date>.csv``. The E matrix if present is written into ``output_E_<date>.csv``.
* ``output_csv``: this directory contains ``output_<algorithm>_<date>.csv`` files. This file contains for each episode the reward obtained, the number of time steps and the cumulative reward.
* ``log``: this directory contains log data for each execution. After the learning process has started, for each step t performed by the RL agent a ``log_<date>.log`` file is updated, with information about the current state st, the performed action at, the new state st+1 and the reward rt+1.
* ``log_dates.log`` is a file appending date and algorithm for each execution. It can be used to collect all ids for all executions and put inside the scripts inside the plotter directory.

### Workflow

The structure of the program is analogous for all algorithms.
It first has aninitialization phase, then it performs an outer loop over the number of episodes, and an inner loop over the time steps until the terminal state is achieved.
For efficiency purposes, a maximum number of time steps is set so that if the terminalstate is not reached the episode still finishes. Inside the inner loop the core of each algorithm is implemented. At the end of the outer loop, the learning process is finished and the best policy found can be evaluated.

1. The initialization procedure is analogous to all algorithms: a Q matrix has beeninitialized to 0, with rows equal to the number of states and columns equal to the number of available actions. This means that for the TCP scenario the Q matrix is 16×5. For the SARSA(λ) algorithm, the E matrix, with same dimensions as Q ,has been initialized to 0.
2. Then, for each algorithm we select actions following the-greedy policy. All algorithms use the same function which first generates a random real number n ∈ [0,1]. If n < a random action is chosen, otherwise the action with the highestvalue inside the Q matrix for the current states is selected. 
3. Inside the core of the program, all algorithms deals with the updates of the Q matrix - and of the E matrix for the SARSA(λ) algorithm - in different ways. For this reason, each algorithm has a specific update function, which updates both the Q matrix and, if necessary, the E matrix.
4. Moreover, at each time stepta reward is given to the algorithm together with the information of the current state in which the RL agent is.
5. Finally, after the learning process, the computed Q value function is evaluatedin the same way for all algorithms: starting from the initial state, the best policy suggested by the Q matrix is followed and the final reward is computed

More information can be found in my master's degree thesis.

## Tests

No tests present for now.


## Contribute

Pull Requests are always welcome.

Ensure the PR description clearly describes the problem and solution. It should include:

* Name of the module modified
* Reasons for modification


## Authors

* **Giulia Milan** - *Initial work* - [giuliapuntoit](https://github.com/giuliapuntoit)

See also the list of [contributors](https://github.com/giuliapuntoit/RL-tcp-toycase/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* SARSA implementation example: [SARSA-example](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/?ref=rp)