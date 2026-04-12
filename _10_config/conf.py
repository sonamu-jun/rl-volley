import _20_model


class Config:
    def __init__(self):
        """====================================================================================================
        ## General Configuration for the Program
        ===================================================================================================="""
        # - Selection of Mode
        self.mode = ['train', 'play'][1]

        # - Set the Target Score
        self.target_score_train = 3
        self.target_score_play = 5

        # - Set the Algorithm and Policy for Player 1
        self.algorithm_1p = 'rule'
        self.policy_1p = None

        # - Set the Algorithm and Policy for Player 2
        self.algorithm_2p = 'human'
        self.policy_2p = None

        # - Set the Game Options
        self.random_serve = True

        # - Set the Random Seed for Reproducibility
        self.seed = None

        # Set the Train Player and Opponent for Training Mode
        self.train_algorithm = 'qlearning'
        self.train_side = ['1p', '2p'][0]
        self.train_rewrite = False
        self.train_opponent = 'rule'
        self.train_policy = None
        self.num_episode = 100

        # Black & White Mode
        BNW_MODE = False
        BNW_MODE_PW = 3011

        """====================================================================================================
        ## Configuration for Path
        ===================================================================================================="""
        for model_name in _20_model.get_available_model_names():
            model_dir = _20_model.get_model_package_dir(model_name)
            setattr(
                self,
                f'path_{model_name}_output',
                str(model_dir / 'outputs'),
            )
            setattr(
                self,
                f'path_{model_name}_policy',
                str(model_dir / 'outputs' / 'policy_trained'),
            )
