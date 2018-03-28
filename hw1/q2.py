import numpy as np

word_len = 5

def viterbi_step(last, cost):
    step = last + cost
    min_costs = np.array([np.min(step, axis=0)]).T
    min_args = np.array([np.argmin(step, axis=0)]).T
    return min_args, min_costs


S = 3
k = word_len - 1
char_dict = {0: 'b', 1: 'k', 2: 'o'}
cost = -np.log(np.array([[0.1, 0.325, 0.25], [0.4, 0, 0.4], [0.2, 0.2, 0.2]]))
T_costs = -np.log(np.array([[1], [0], [0]]))
T_args = np.array([[], [], []])

for i in range(k):
    last = np.tile(T_costs[:, i], (S, 1)).T
    min_args, min_costs = viterbi_step(last, cost)
    T_costs = np.concatenate([T_costs, min_costs], axis=1)
    T_args = np.concatenate([T_args, min_args], axis=1)

# final step
final_cost = -np.log(np.array([0.325, 0.2, 0.4]))
final = np.array(T_costs[:, -1])
min_arg, _ = viterbi_step(final, final_cost)

# backprop
result = list(min_arg)
last_arg = result[0]
for i in range(k - 1, -1, -1):
    last_arg = int(T_args[last_arg, i])
    result.append(last_arg)

str_result = ''.join([char_dict[i] for i in result[::-1]])
print(str_result)
