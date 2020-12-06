import time

from qlearning_client_class import RLFull
from sarsa_client_class import SarsaFull
from sarsa_lambda_client_class import SarsaLambdaFull

episodes = 200
epsilon = 0.6

for i in range(10):
    x_results, y_rew = RLFull(total_episodes=episodes, epsilon=epsilon, disable_graphs=True).run()
    time.sleep(5)

for i in range(10):
    x_results, y_rew = SarsaFull(total_episodes=episodes, epsilon=epsilon, disable_graphs=True).run()
    time.sleep(5)

for i in range(10):
    x_results, y_rew = SarsaLambdaFull(total_episodes=episodes, epsilon=epsilon, disable_graphs=True).run()
    time.sleep(5)
