# Import Required Internal Modules
import _00_environment
import _20_model


def run(conf):
    """====================================================================================================
    ## Creation of Environment Instance and Loading model for Each Player
    ===================================================================================================="""
    # - Create Envionment Instance
    env = create_invironment_instance(conf)

    # - Load the model for each player
    model_1p = load_model(conf, player='1p')
    model_2p = load_model(conf, player='2p')

    """====================================================================================================
    ## Playing Episode
    ===================================================================================================="""
    # - Set Environment with Selected Algorithm and Policy for Each Player
    env.set(player1=model_1p, player2=model_2p, random_serve=conf.random_serve)

    # - Wait for 's' key to Start Episode
    command = env.wait_key_for_start(key=ord('s'))
    if command == "quit":
        env.close()
        return

    # - Run Episode
    while True:
        # - Get Play Result for Each Step
        play_result = env.get_play_result()
        done = play_result['done']
        score = play_result['score']

        # - Consume Viewer Command
        command = env.consume_viewer_command()

        # - Check Terminate Condition
        if command == "quit":
            break

        if done is True:
            if score['p1'] > score['p2']:
                winner_text = 'player1'
            elif score['p2'] > score['p1']:
                winner_text = 'player2'
            else:
                winner_text = 'draw'

            # - Print Winner and Final Score
            print("winner: {}".format(winner_text))
            print(f"final score: {score['p1']}:{score['p2']}")

            command = env.wait_for_command(allow_restart=True)
            if command == "restart":
                env.set(player1=model_1p, player2=model_2p, random_serve=conf.random_serve)
                command = env.wait_key_for_start(key=ord('s'))
                if command == "quit":
                    break
                continue
            break

    # - Terminate Episode and Close Environment
    env.close()


def create_invironment_instance(conf):
    """====================================================================================================
    ## Creation of Environment Instance
    ===================================================================================================="""

    # - Load Configuration
    RENDER_MODE = "human"
    TARGET_SCORE = conf.target_score_play
    SEED = conf.seed

    # - Create Envionment Instance
    env = _00_environment.Env(render_mode=RENDER_MODE,
                         target_score=TARGET_SCORE,
                         seed=SEED)

    # - Return Environment Instance
    return env


def load_model(conf, player):
    """====================================================================================================
    ## Loading Policy for Each Player
    ===================================================================================================="""
    # - Check Algorithm and Policy Name for Each Player
    ALGORITHM = conf.algorithm_1p if player == '1p' else conf.algorithm_2p
    POLICY_NAME = conf.policy_1p if player == '1p' else conf.policy_2p

    # - Load Selected Policy for Each Player
    if ALGORITHM == 'human':
        model = 'HUMAN'

    elif ALGORITHM == 'rule':
        model = 'RULE'

    else:
        model = _20_model.create_model(
            conf,
            algorithm_name=ALGORITHM,
            policy_name_for_play=POLICY_NAME,
        )

    # - Return Loaded Model for Each Player
    return model
