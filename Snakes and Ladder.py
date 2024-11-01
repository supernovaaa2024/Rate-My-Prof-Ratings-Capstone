import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(threshold=np.inf)

transition_matrix = np.zeros((100,100))
print(transition_matrix)
for i in range (99):
    for dice_roll in range (1,7):
        if i+ dice_roll <= 99:
            transition_matrix[i][i+ dice_roll] = 1/6

# ------ Using the items method to store the 
snakes = {30:8, 46:25, 58:37, 74:55, 84:65, 96:61}
ladders = {6:27, 19:57, 33:85, 63:81, 72:94}
# ------

# ------
for start, end in snakes.items():
    transition_matrix[:][start] = 0
    transition_matrix[start][end] = 1/6
for start, end in ladders.items():
    transition_matrix[:][start] = 0
    transition_matrix[start][end] = 1/6
# ------


# ------
for i in range(94,100):
    for dice_roll in range (1,7):
        if i+dice_roll > 100:
            transition_matrix[i][100-i+dice_roll] = 1/6 + 1/6
# ------


# ------
transition_matrix[99][99] = 1 #Absorbing State
initial_position = np.zeros(100)
initial_position[0] = 1
# ------


# ------
df = pd.DataFrame(transition_matrix)
desktop_path = os.path.expanduser("~/Desktop/transition_matrix.xlsx")
df.to_excel(desktop_path, index=False)
# ------

def compute(initial_position, transition_matrix, n_moves):
    turn_n = initial_position
    for i in range(n_moves):
        turn_n = np.dot(turn_n, transition_matrix)

    df2 = pd.DataFrame(turn_n)
    desktop_path = os.path.expanduser("~/Desktop/turn_n.xlsx")
    df2.to_excel(desktop_path, index=False)

    positions = np.linspace(1, 100, 100)
    plt.figure(figsize=(12, 6))
    plt.bar(positions, turn_n)
    plt.title(f"Probabilities at each position after {n_moves} moves")
    plt.xlabel("Position")
    plt.ylabel("Probabilities")

    heatmap = turn_n.reshape((10, 10))
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap, cmap='coolwarm')
    plt.title(f"Heatmap after {n_moves} moves")
    plt.xlabel("Rows")
    plt.ylabel("Columns")
    plt.show()

    return turn_n

compute(initial_position, transition_matrix, 10)








