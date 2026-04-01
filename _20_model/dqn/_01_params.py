from sympy import python


def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    # Define the training parameters
    TRAIN_PARAMS = {
        # Learning Rate
        "learning_rate": 3e-4,

        # Discount Factor
        "gamma": 0.99,

        # Epsilon-Greedy Exploration Parameters
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.9995,

        # Replay Buffer Parameters
        "replay_buffer_size": 50000,
        "replay_start_size": 2000,
        "batch_size": 128,

        # Neural Network Architecture Parameters
        "hidden_dim": 64,
        "hidden_layer_count": 2,

        "update_every": 4,

        # Target Network Update Interval
        "target_update_interval": 200,

        # Initial Values for Training
        "epsilon_init": None,
        "training_steps_init": 0,

        # Maximum Steps per Episode
        "max_steps_per_episode": 30*30,

        # Progress Display Options
        "show_progress": True,
        "progress_interval": 50,
    }

    # Return the training parameters
    return TRAIN_PARAMS


def get_play_params():
    """====================================================================================================
    ## Hyperparameter Setting for Playing
    ===================================================================================================="""
    # Define the play parameters
    PLAY_PARAMS = {
        # Maximum Steps per Episode
        "max_steps": 30*60*60,
    }

    # Return the play parameters
    return PLAY_PARAMS
