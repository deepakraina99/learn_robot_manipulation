import numpy as np

a = np.array([1,2,3])
b = np.array([1,2,3])
print(-a)
print(np.concatenate((a, b)))

def map_action_values(output_lower_limit, output_upper_limit, action_value):
    input_upper_limit, input_lower_limit = 1, -1
    output = output_lower_limit + ((output_upper_limit - output_lower_limit) / (input_upper_limit - input_lower_limit)) * (action_value - input_lower_limit)
    return output

output_lower_limit = -300
output_upper_limit = 300
action_value = -0.5

output = map_action_values(output_lower_limit, output_upper_limit, action_value)
print('output:', output)


