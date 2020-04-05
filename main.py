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
import matplotlib.pyplot as plt

num_episodes = 10
x = []
y_rewards = []
y2_rewards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for n in range(num_episodes):
    x.append(n)
    optimalPolicy, obtainedReward = SarsaSimplified(total_episodes=n, disable_graphs=True).run()
    if optimalPolicy:
        print("Optimal policy was found with reward", obtainedReward)
    else:
        print("No optimal policy reached with reward", obtainedReward)
    y_rewards.append(obtainedReward)

plt.plot(x, y_rewards, label="Sarsa Simplified")
plt.plot(x, y2_rewards, label="Sarsa Full")
plt.xlabel('Episodes')
plt.ylabel('Final reward')
plt.title('Final policy over number of episodes chosen.')
plt.legend()
plt.show()
