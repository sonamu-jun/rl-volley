# Import Required Internal Libraries
from _00_environment.constants import GROUND_HALF_WIDTH


def normalize_minmax(value, minimum_value, maximum_value):
    """====================================================================================================
    ## Min-Max Normalization Wrapper
    ===================================================================================================="""
    if maximum_value <= minimum_value:
        return 0.0

    normalized_value = (float(value) - float(minimum_value)) / \
        (float(maximum_value) - float(minimum_value))

    if normalized_value < 0.0:
        return 0.0
    if normalized_value > 1.0:
        return 1.0
    return float(normalized_value)


def select_mat_for_reward(materials):
    """====================================================================================================
    ## Load materials for reward design
    ===================================================================================================="""
    # Self Position (x 0~431, y 0~252)
    self_position = materials["self_position"]

    # Opponent Position (x 0~431, y 0~252)
    opponent_position = materials["opponent_position"]

    # Ball Position (x 0~431, y 0~252)
    ball_position = materials["ball_position"]

    # Self Action Name (String)
    self_action_name = str(materials["self_action_name"])

    # Opponent Action Name (String)
    opponent_action_name = str(materials["opponent_action_name"])

    # Rally Frames (Float)
    rally_total_frames_until_point = float(
        materials["rally_total_frames_until_point"])

    # Whether Self Scored a Point (0 or 1)
    point_scored = int(materials["point_result"]["scored"])

    # Whether Self Lost a Point (0 or 1)
    point_lost = int(materials["point_result"]["lost"])

    # Whether Self Used a Spike (0 or 1)
    self_spike_used = int(self_action_name.startswith("spike_"))

    # Whether Self Used a Dive (0 or 1)
    self_dive_used = int(self_action_name.startswith("dive_"))

    # Whether Opponent Used a Dive (0 or 1)
    opponent_dive_used = int(opponent_action_name.startswith("dive_"))

    # Whether Opponent Used a Spike (0 or 1)
    opponent_spike_used = int(opponent_action_name.startswith("spike_"))

    # Whether Self Won the Match (0 or 1)
    match_won = int(materials["match_result"]["won"] > 0.5)

    # x-distance from Self to Net Center (0.0~ )
    self_net_distance = abs(self_position[0] - GROUND_HALF_WIDTH)

    # x-distance from Opponent to Net Center (0.0~ )
    opponent_net_distance = abs(opponent_position[0] - GROUND_HALF_WIDTH)

    # Slect materials for reward design
    SELECTED_MATARIALS = {
        "self_position": self_position,
        "opponent_position": opponent_position,
        "ball_position": ball_position,
        "self_action_name": self_action_name,
        "opponent_action_name": opponent_action_name,
        "self_net_distance": self_net_distance,
        "opponent_net_distance": opponent_net_distance,
        "point_scored": point_scored,
        "point_lost": point_lost,
        "self_spike_used": self_spike_used,
        "self_dive_used": self_dive_used,
        "opponent_dive_used": opponent_dive_used,
        "opponent_spike_used": opponent_spike_used,
        "match_won": match_won,
        "rally_total_frames_until_point": rally_total_frames_until_point,
    }

    # Return selected materials
    return SELECTED_MATARIALS


def calculate_reward(materials):
    """====================================================================================================
    ## Load Materials For Reward Design
    ===================================================================================================="""
    # Load materials for reward design
    mat = select_mat_for_reward(materials)

    """====================================================================================================
    ## Defining Scale Factors and Calculating Reward
    ===================================================================================================="""
    # Define Scale Factor for Point Score Reward
    SCALE_POINT_SCORE_REWARD = 25.0
    SCALE_POINT_LOST_PENALTY = 25.0

    # Define Scale Factor for Self Bonus/Penalty
    SCALE_SELF_SPIKE_BONUS = 0.
    SCALE_SELF_DIVE_BONUS = 0.

    # Define Scale Factor for Opponent Bonus/Penalty
    SCALE_OPPONENT_DIVE_BONUS = 0.
    SCALE_OPPONENT_SPIKE_PENALTY = 0.

    # Define Scale Factor for Rally Frame Reward
    SCALE_RALLY_FRAME = 0.
    SCALE_RALLY_FRAME_MAX = 0.

    # Define Scale Factor for Match Win Bonus
    SCALE_MATCH_WIN_BONUS = 30.0

    """====================================================================================================
    ## Calculating Reward by Accumulating Different Components
    ===================================================================================================="""
    # Initialize Reward at the Certain Transition Step
    reward = 0.0

    # Accumulate Point Score Reward
    reward += SCALE_POINT_SCORE_REWARD * mat["point_scored"]
    reward -= SCALE_POINT_LOST_PENALTY * mat["point_lost"]

    # Accumulate Self Bonus/Penalty Reward
    reward += SCALE_SELF_SPIKE_BONUS * mat["self_spike_used"]
    reward += SCALE_SELF_DIVE_BONUS * mat["self_dive_used"]

    # Accumulate Opponent Bonus/Penalty Reward
    reward += SCALE_OPPONENT_DIVE_BONUS * mat["opponent_dive_used"]
    reward -= SCALE_OPPONENT_SPIKE_PENALTY * mat["opponent_spike_used"]

    # Accumulate Rally Frame Reward
    rally_reward = 0.0
    if mat["point_scored"] > 0.5:
        rally_reward = min(mat["rally_total_frames_until_point"] * SCALE_RALLY_FRAME,
                           SCALE_RALLY_FRAME_MAX)
    reward += rally_reward

    # Accumulate Match Win Reward
    reward += SCALE_MATCH_WIN_BONUS * mat["match_won"]

    # Normalize Reward to 0~1 with a Fixed Reward Window
    # REWARD_NORMALIZE_MIN = -100.0
    # REWARD_NORMALIZE_MAX = 100.0
    # reward = normalize_minmax(
    #     reward, REWARD_NORMALIZE_MIN, REWARD_NORMALIZE_MAX)

    # Return Calculated Reward
    return reward
