import numpy as np
import torch

def epsilon_greedy_action_selection(policy, state, epsilon):
    """====================================================================================================
    ## Select Action by Epsilon-Greedy Strategy
    ===================================================================================================="""
    # - Convert State to Tensor
    policy_device = next(policy.parameters()).device
    if torch.is_tensor(state):
        state_tensor = state.to(device=policy_device, dtype=torch.float32)
    else:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=policy_device)
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)

    # - Load Q-Vector for Current State
    with torch.no_grad():
        q_vector = policy(state_tensor).squeeze(0)
    dim_action = int(q_vector.shape[0])

    # - If Random Value is Less than Epsilon, Select a Random Action
    if np.random.rand() < float(epsilon):
        action_idx = int(torch.randint(dim_action, (1,)).item())

    # - Otherwise, Select an Action with the Highest Q-Value
    else:
        max_value = torch.max(q_vector)
        candidate_indexes = torch.nonzero(
            q_vector == max_value, as_tuple=False).flatten()
        random_idx = torch.randint(
            low=0,
            high=candidate_indexes.numel(),
            size=(1,),
            device=candidate_indexes.device,
        )
        action_idx = int(candidate_indexes[random_idx].item())

    # - Convert the Action Index to One-Hot Action Vector
    action = np.zeros(dim_action, dtype=np.float32)
    action[action_idx] = 1.0

    # - Return the Selected Action Vector
    return action


def decay_epsilon(epsilon_start, epsilon_decay, epsilon_end):
    """====================================================================================================
    ## Decaying Epsilon for Q-Learning Algorithm
    ===================================================================================================="""
    # - Decay Epsilon
    next_epsilon = float(epsilon_start) * float(epsilon_decay)

    # - Ensure Epsilon Does Not Fall Below the Minimum Threshold
    if next_epsilon < float(epsilon_end):
        next_epsilon = float(epsilon_end)

    # - Return Decayed Epsilon
    return next_epsilon
