''' Voglio un main che:
1. chiami ciascuno degli algoritmi
2. passandogli i parametri (episodi, epsilon, lambda, ecc)
3. che verifichi che sia arrivato alla policy ottimale
4. che calcoli il reward di quella policy
5. che salvi il numero di episodi
6. che quantifichi il quanto siamo vicini alla policy ottimale (come faccio?)
7. che plotti con questi valori il numero di episodi per arrivare alla policy ottimale
'''

from sarsa_simplified_class import SarsaSimplified
from sarsa_lambda_simplified_class import SarsaLambdaSimplified
import matplotlib.pyplot as plt

num_episodes = 1000
x = []
y_sarsa_rewards = []
y_sarsa_lambda_rewards = []
dis = True

for n in range(num_episodes):
    x.append(n)
    optimalPolicy, obtainedReward = SarsaSimplified(total_episodes=n, disable_graphs=dis).run()
    if dis == False:
        if optimalPolicy:
            print("[SARSA] Optimal policy was found with reward", obtainedReward)
        else:
            print("[SARSA] No optimal policy reached with reward", obtainedReward)
    y_sarsa_rewards.append(obtainedReward)

    optimalPolicy, obtainedReward = SarsaLambdaSimplified(total_episodes=n, disable_graphs=dis).run()
    if dis == False:
        if optimalPolicy:
            print("[SARSA(lambda)] Optimal policy was found with reward", obtainedReward)
        else:
            print("[SARSA(lambda)] No optimal policy reached with reward", obtainedReward)
    y_sarsa_lambda_rewards.append(obtainedReward)

plt.plot(x, y_sarsa_rewards, label="Sarsa Simplified")
plt.plot(x, y_sarsa_lambda_rewards, label="Sarsa Lambda Simplified")
plt.xlabel('Episodes')
plt.ylabel('Final reward')
plt.title('Final policy over number of episodes chosen.')
plt.legend()
plt.show()
