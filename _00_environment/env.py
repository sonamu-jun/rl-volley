import importlib
import importlib.util
import inspect
import pickle
import random
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MethodType

import numpy as np

from .actions import ACTION_NAMES
from .actions import build_user_input
from .actions import describe_user_input
from .actions import select_action_name
from .constants import GROUND_HALF_WIDTH
from .constants import PLAYER_TOUCHING_GROUND_Y_COORD
from .engine import Engine
from .input import UserInput
from .state import build_training_state_bundle
from .state import build_state_view


SUPPORTED_RENDER_MODES = ("human", "rgb_array", "log")


class CompatStateView(Mapping):
    __slots__ = ["env"]

    def __init__(self, env):
        self.env = env

    def _full_state(self):
        return self.env._build_state_dict()

    def _perspective_player(self):
        return self.env.compat_state_player

    def __getitem__(self, key):
        full_state = self._full_state()
        if key == "raw":
            return full_state[self._perspective_player()]["raw"]
        if key == "key":
            return full_state[self._perspective_player()]["key"]
        return full_state[key]

    def __iter__(self):
        yield "player1"
        yield "player2"
        yield "raw"
        yield "key"

    def __len__(self):
        return 4

    def __contains__(self, key):
        return key in ("player1", "player2", "raw", "key")

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __repr__(self):
        return repr({key: self[key] for key in self})


class CompatPlayerStateView(Mapping):
    __slots__ = ["env", "player_name"]

    def __init__(self, env, player_name):
        self.env = env
        self.player_name = player_name

    def _state(self):
        return self.env._build_state_dict()[self.player_name]

    def __getitem__(self, key):
        return self._state()[key]

    def __iter__(self):
        return iter(self._state())

    def __len__(self):
        return len(self._state())

    def __contains__(self, key):
        return key in self._state()

    def get(self, key, default=None):
        return self._state().get(key, default)

    def __repr__(self):
        return repr(self._state())


class CompatQTable(dict):
    __slots__ = ["action_count"]

    def __init__(self, action_count):
        super().__init__()
        self.action_count = int(action_count)

    def _normalize_key(self, state_key):
        if hasattr(state_key, "tolist"):
            state_key = state_key.tolist()
        if isinstance(state_key, list):
            return tuple(state_key)
        return state_key

    def __contains__(self, state_key):
        return super().__contains__(self._normalize_key(state_key))

    def __getitem__(self, state_key):
        normalized_key = self._normalize_key(state_key)
        if not super().__contains__(normalized_key):
            super().__setitem__(
                normalized_key,
                np.zeros(self.action_count, dtype=np.float32),
            )
        return super().__getitem__(normalized_key)

    def __setitem__(self, state_key, qvalues):
        normalized_key = self._normalize_key(state_key)
        super().__setitem__(
            normalized_key,
            np.asarray(qvalues, dtype=np.float32),
        )

    def get(self, state_key, default=None):
        return super().get(self._normalize_key(state_key), default)


class Env:
    _model_qlearning_compat_initialized = False

    def __init__(
        self,
        render_mode="rgb_array",
        target_score=15,
        more_random=False,
        seed=None,
        randomize_serve_on_reset=False,
        rally_step_limit=3000,
    ):
        if render_mode not in SUPPORTED_RENDER_MODES:
            raise ValueError(f"unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self.is_log_mode = render_mode == "log"
        self.target_score = target_score
        self.seed = self._normalize_seed(seed)
        self.randomize_serve_on_reset = randomize_serve_on_reset
        self.randomize_serve_enabled = bool(randomize_serve_on_reset)
        self.rally_step_limit = rally_step_limit
        self.point_pause_frames = 4

        self.engine = Engine(False, False, more_random)
        self.engine.create_viewer(render_mode)

        self.direction_memory = {0: 1, 1: 1}
        self.last_action_names = {0: None, 1: None}
        self.started = False
        self.rally_done = False
        self.match_done = False
        self.rally_step_count = 0
        self.is_player2_serve = False
        self.scores = {"player1": 0, "player2": 0}
        self.last_match_events = None
        self.policy_cache = {}
        self.policy_path_cache = {}
        self.module_cache = {}
        self.state_cache = None
        self.training_state_bundle_cache = {}
        self.custom_state_key_function = None
        self.viewer_player_labels = {
            "player1": {"title": "", "detail": ""},
            "player2": {"title": "", "detail": ""},
        }
        self.viewer_player_controllers = {
            "player1": "",
            "player2": "",
        }
        self.compat_play_config = {
            "player1": {"controller": "rule", "policy": None, "model": None},
            "player2": {"controller": "rule", "policy": None, "model": None},
        }
        self.compat_mode_active = False
        self.compat_state_player = "player1"
        self.compat_state_view = CompatStateView(self)
        self.compat_player_state_views = {
            "player1": CompatPlayerStateView(self, "player1"),
            "player2": CompatPlayerStateView(self, "player2"),
        }
        self.compat_pending_actions = {
            "player1": None,
            "player2": None,
        }

        if not Env._model_qlearning_compat_initialized:
            self._enable_model_qlearning_compatibility()
            Env._model_qlearning_compat_initialized = True
        self._enable_train_load_model_compatibility()

        if self.seed is not None:
            self.engine.seed(self.seed)

        self.reset(return_state=False)

    def _normalize_seed(self, seed):
        if seed is None:
            return None
        if isinstance(seed, str):
            normalized_seed = seed.strip()
            if normalized_seed == "" or normalized_seed.lower() == "none":
                return None
            return int(normalized_seed)
        return int(seed)

    def _set_next_serve_side(self, scorer=None):
        if self.randomize_serve_enabled:
            self.is_player2_serve = bool(random.randrange(0, 2))
            return

        if scorer is None:
            self.is_player2_serve = False
            return

        self.is_player2_serve = scorer == "player2"

    def _invalidate_cached_state(self):
        self.state_cache = None
        self.training_state_bundle_cache = {}

    def _enable_model_qlearning_compatibility(self):
        state_modules = []
        for module_name in (
            "_20_model.qlearning._03_state_design",
            "_20_model.qlearning.custom.state_design",
            "_20_model.qlearning.custom._03_state_design",
            "model.qlearning._03_state_design",
            "model.qlearning.custom.state_design",
            "model.qlearning.custom._03_state_design",
        ):
            try:
                state_module = importlib.import_module(module_name)
            except Exception:
                continue
            state_modules.append(state_module)

        for state_module in state_modules:
            original = getattr(state_module, "calculate_state_key", None)
            if original is not None and not getattr(original, "_enpika_hashable_wrapper", False):
                def calculate_state_key_hashable(materials, _original=original):
                    state_key = _original(materials)
                    if hasattr(state_key, "tolist"):
                        state_key = state_key.tolist()
                    if isinstance(state_key, list):
                        return tuple(state_key)
                    return state_key

                calculate_state_key_hashable._enpika_hashable_wrapper = True
                state_module.calculate_state_key = calculate_state_key_hashable

        qtable_modules = []
        for module_name in (
            "_20_model.qlearning._02_qtable",
            "_20_model.qlearning.custom.qtable",
            "_20_model.qlearning.custom._02_qtable",
            "model.qlearning._02_qtable",
            "model.qlearning.custom.qtable",
            "model.qlearning.custom._02_qtable",
        ):
            try:
                qtable_module = importlib.import_module(module_name)
            except Exception:
                continue
            qtable_modules.append(qtable_module)

        if not qtable_modules:
            return

        def load_qtable_compat(path):
            payload = None
            with open(path, "rb") as file:
                try:
                    payload = pickle.load(file)
                except Exception:
                    payload = None

            if payload is None:
                try:
                    import torch
                except ImportError as error:
                    raise ImportError("torch is required to load qlearning qtables") from error

                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as zip_file:
                        has_data_pickle = False
                        for file_name in zip_file.namelist():
                            if file_name.endswith("data.pkl"):
                                has_data_pickle = True
                                break
                        if not has_data_pickle:
                            return CompatQTable(len(ACTION_NAMES))

                payload = torch.load(path, map_location="cpu", weights_only=False)

            loaded_table = payload.get("table", {})
            action_names = payload.get("action_names")
            action_count = len(ACTION_NAMES)
            if action_names is not None:
                action_count = len(action_names)
            elif loaded_table:
                first_key = next(iter(loaded_table))
                first_qvalues = loaded_table[first_key]
                if hasattr(first_qvalues, "tolist"):
                    first_qvalues = first_qvalues.tolist()
                action_count = len(first_qvalues)

            compat_qtable = CompatQTable(action_count)
            for state_key, qvalues in loaded_table.items():
                compat_qtable[state_key] = qvalues
            return compat_qtable

        load_qtable_compat._enpika_compat_wrapper = True
        for qtable_module in qtable_modules:
            original_load_qtable = getattr(qtable_module, "load_qtable", None)
            if original_load_qtable is not None and getattr(original_load_qtable, "_enpika_compat_wrapper", False):
                continue
            qtable_module.load_qtable = load_qtable_compat

    def _enable_train_load_model_compatibility(self):
        train_module = sys.modules.get("_30_src.train")
        if train_module is None:
            return

        original_load_model = getattr(train_module, "load_model", None)
        if original_load_model is None:
            return
        if getattr(original_load_model, "_enpika_train_wrapper", False):
            return

        def load_model_compat(conf, player):
            player_name = str(player).strip()
            train_side = str(getattr(conf, "train_side", "")).strip()
            algorithm_name = None
            policy_name = None
            opponent_name = str(getattr(conf, "train_opponent", "")).strip().lower()

            if not hasattr(conf, "train_rewrite"):
                conf.train_rewrite = False
            if not hasattr(conf, "_enpika_self_play_model"):
                conf._enpika_self_play_model = None
            if not hasattr(conf, "_enpika_self_play_key"):
                conf._enpika_self_play_key = None

            if player_name == train_side:
                algorithm_name = str(getattr(conf, "train_algorithm", "")).strip().lower()
                policy_name = getattr(conf, "train_policy", None)
            else:
                if opponent_name == "self":
                    self_play_key = (
                        str(getattr(conf, "train_algorithm", "")).strip().lower(),
                        getattr(conf, "train_policy", None),
                    )
                    if (
                        conf._enpika_self_play_model is not None
                        and conf._enpika_self_play_key == self_play_key
                    ):
                        return conf._enpika_self_play_model
                    algorithm_name = str(getattr(conf, "train_algorithm", "")).strip().lower()
                    policy_name = getattr(conf, "train_policy", None)
                elif opponent_name:
                    algorithm_name = opponent_name
                else:
                    algorithm_name = None

            if not algorithm_name:
                return original_load_model(conf, player)

            if algorithm_name == "human":
                return "HUMAN"
            if algorithm_name == "rule":
                return "RULE"

            try:
                model_package = importlib.import_module("_20_model")
                model = model_package.create_model(
                    conf,
                    algorithm_name=algorithm_name,
                    policy_name_for_play=policy_name,
                )
            except Exception:
                return original_load_model(conf, player)

            if opponent_name == "self":
                conf._enpika_self_play_model = model
                conf._enpika_self_play_key = (algorithm_name, policy_name)
            return model

            return original_load_model(conf, player)

        load_model_compat._enpika_train_wrapper = True
        train_module.load_model = load_model_compat

    def reset(
        self,
        random_serve=None,
        randomize_serve=None,
        return_state=True,
        player1=None,
        player1_policy=None,
        player2=None,
        player2_policy=None,
    ):
        if player1 is not None or player2 is not None:
            self.compat_mode_active = True
            self._set_compat_play_config(
                player1=player1,
                player1_policy=player1_policy,
                player2=player2,
                player2_policy=player2_policy,
            )

        self.scores = {"player1": 0, "player2": 0}
        self.last_match_events = None
        if randomize_serve is None and random_serve is not None:
            randomize_serve = random_serve
        if randomize_serve is None:
            randomize_serve = self.randomize_serve_on_reset

        self.randomize_serve_enabled = bool(randomize_serve)
        self._set_next_serve_side()

        self.match_done = False
        self.compat_state_player = "player1"
        self.compat_pending_actions = {
            "player1": None,
            "player2": None,
        }
        self._reset_rally(clear_flags=True)
        self.started = True
        if return_state:
            return self.get_state()
        return None

    def _reset_rally(self, clear_flags):
        self._invalidate_cached_state()
        self.engine.reset(self.is_player2_serve)
        self.direction_memory = {0: 1, 1: 1}
        self.last_action_names = {0: None, 1: None}
        self.rally_step_count = 0
        if clear_flags:
            self.rally_done = False
        if self.engine.viewer is not None:
            self.engine.viewer.reset_input_state()
            self._refresh_viewer()

    def _refresh_viewer(self, render_now=True):
        if self.engine.viewer is None:
            return

        self.engine.viewer.set_player_labels(
            self.viewer_player_labels["player1"]["title"],
            self.viewer_player_labels["player1"]["detail"],
            self.viewer_player_labels["player2"]["title"],
            self.viewer_player_labels["player2"]["detail"],
        )
        self.engine.viewer.set_human_controllers(
            self.viewer_player_controllers["player1"] == "human",
            self.viewer_player_controllers["player2"] == "human",
        )
        self.engine.viewer.set_match_status(
            self.scores["player1"],
            self.scores["player2"],
            self._build_winner_text(),
        )
        self.engine.viewer.update()
        if render_now and self.render_mode == "human":
            self.engine.render("human")

    def _pause_after_point(self):
        if self.engine.viewer is None or self.render_mode != "human":
            return

        self._refresh_viewer(render_now=True)
        for _ in range(self.point_pause_frames):
            if self.engine.viewer.closed_requested:
                break
            self.engine.render("human")

    def _build_winner_text(self):
        if not self.match_done:
            return None

        if self.scores["player1"] > self.scores["player2"]:
            return "PLAYER 1 WIN"
        if self.scores["player2"] > self.scores["player1"]:
            return "PLAYER 2 WIN"
        return "DRAW"

    def _set_match_end_player_states(self):
        winner_name = None
        if self.scores["player1"] > self.scores["player2"]:
            winner_name = "player1"
        elif self.scores["player2"] > self.scores["player1"]:
            winner_name = "player2"

        for player_id, player_name in enumerate(("player1", "player2")):
            player = self.engine.players[player_id]
            player.game_ended = True
            player.is_winner = winner_name == player_name
            player.state = 5 if player.is_winner else 6
            player.frame_number = 0
            player.delay_before_next_frame = 0
            player.lying_down_duration_left = -1
            player.y = PLAYER_TOUCHING_GROUND_Y_COORD
            player.y_velocity = 0

    def _clone_events(self, events):
        if events is None:
            return None

        cloned_events = dict(events)
        touch = events.get("touch")
        if isinstance(touch, dict):
            cloned_events["touch"] = dict(touch)
        return cloned_events

    def _build_score_snapshot(self, events=None):
        if events is None:
            events = self.last_match_events

        return {
            "player1": int(self.scores["player1"]),
            "player2": int(self.scores["player2"]),
            "p1": int(self.scores["player1"]),
            "p2": int(self.scores["player2"]),
            "rally_done": bool(self.rally_done),
            "match_done": bool(self.match_done),
            "events": self._clone_events(events),
        }

    def _shorten_label_text(self, text, max_length=24):
        text = str(text).strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def _resolve_viewer_controller_name(self, controller_name, policy_source):
        controller_name = str(controller_name).strip().lower()
        if controller_name != "model":
            return controller_name
        if policy_source is None:
            return "model"

        model_module_name = getattr(policy_source.__class__, "__module__", "")
        module_tokens = str(model_module_name).split(".")
        if "_20_model" in module_tokens:
            module_index = module_tokens.index("_20_model")
            if module_index + 1 < len(module_tokens):
                return str(module_tokens[module_index + 1]).strip().lower()
        if len(module_tokens) >= 2:
            return str(module_tokens[-2]).strip().lower()
        return str(policy_source.__class__.__name__).strip().lower()

    def _build_policy_label_text(self, player_key, controller_name, policy_source):
        controller_name = self._resolve_viewer_controller_name(
            controller_name, policy_source)
        if controller_name == "rule":
            return "policy: -"
        if controller_name == "human":
            if player_key == "player1":
                return "keys: U/H/J/K/Z"
            return "keys: Arrows/Enter"

        if policy_source is None:
            return "policy: -"

        if not isinstance(policy_source, (str, Path)):
            policy_name = getattr(policy_source, "policy_name", None)
            if policy_name is not None:
                return f"{controller_name}: {self._shorten_label_text(policy_name)}"
            return f"{controller_name}: {self._shorten_label_text(policy_source.__class__.__name__)}"

        policy_path = Path(str(policy_source))
        policy_name = policy_path.name or str(policy_source)
        if policy_path.suffix:
            policy_name = policy_path.stem
        return f"policy: {self._shorten_label_text(policy_name)}"

    def _set_viewer_player_labels(self, player1, player1_policy, player2, player2_policy):
        player1_name = self._resolve_viewer_controller_name(
            player1, player1_policy)
        player2_name = self._resolve_viewer_controller_name(
            player2, player2_policy)
        self.viewer_player_controllers = {
            "player1": player1_name,
            "player2": player2_name,
        }
        self.viewer_player_labels = {
            "player1": {
                "title": f"1P {player1_name.upper()}",
                "detail": self._build_policy_label_text("player1", player1_name, player1_policy),
            },
            "player2": {
                "title": f"2P {player2_name.upper()}",
                "detail": self._build_policy_label_text("player2", player2_name, player2_policy),
            },
        }

    def show_play_ready(self, player1, player1_policy, player2, player2_policy):
        if self.engine.viewer is None:
            return

        self._set_viewer_player_labels(player1, player1_policy, player2, player2_policy)
        self.engine.viewer.set_human_controllers(
            self.viewer_player_controllers["player1"] == "human",
            self.viewer_player_controllers["player2"] == "human",
        )
        self.engine.viewer.set_match_status(
            self.scores["player1"],
            self.scores["player2"],
            "PRESS S TO START",
            "Q quit  S start",
        )
        self.engine.viewer.update()
        if self.render_mode == "human":
            self.engine.render("human")

    def _get_frame(self):
        if self.engine.viewer is None:
            return None

        frame = self.engine.render("rgb_array")
        return np.array(frame, copy=True)

    def _player_name_to_id(self, player_name):
        if player_name in (0, "player1"):
            return 0
        if player_name in (1, "player2"):
            return 1
        raise ValueError(f"unknown player name: {player_name}")

    def _player_id_to_name(self, player_id):
        if player_id == 0:
            return "player1"
        if player_id == 1:
            return "player2"
        raise ValueError(f"unknown player id: {player_id}")

    def _normalize_compat_player_name(self, player_name):
        if player_name in (0, "1p", "player1"):
            return "player1"
        if player_name in (1, "2p", "player2"):
            return "player2"
        raise ValueError(f"unknown compatibility player name: {player_name}")

    def _set_compat_play_config(self, player1=None, player1_policy=None, player2=None, player2_policy=None):
        if player1 is not None:
            if isinstance(player1, str):
                self.compat_play_config["player1"]["controller"] = str(player1).strip().lower()
                self.compat_play_config["player1"]["policy"] = player1_policy
                self.compat_play_config["player1"]["model"] = None
            else:
                self._attach_model_runtime(player1)
                self.compat_play_config["player1"]["controller"] = "model"
                self.compat_play_config["player1"]["policy"] = player1
                self.compat_play_config["player1"]["model"] = player1
        if player2 is not None:
            if isinstance(player2, str):
                self.compat_play_config["player2"]["controller"] = str(player2).strip().lower()
                self.compat_play_config["player2"]["policy"] = player2_policy
                self.compat_play_config["player2"]["model"] = None
            else:
                self._attach_model_runtime(player2)
                self.compat_play_config["player2"]["controller"] = "model"
                self.compat_play_config["player2"]["policy"] = player2
                self.compat_play_config["player2"]["model"] = player2

        if self.engine.viewer is not None:
            self._set_viewer_player_labels(
                self.compat_play_config["player1"]["controller"],
                self.compat_play_config["player1"]["policy"],
                self.compat_play_config["player2"]["controller"],
                self.compat_play_config["player2"]["policy"],
            )

    def _attach_model_runtime(self, model):
        model.env = self
        model_module = importlib.import_module(model.__class__.__module__)

        for function_name in (
            "get_transition",
            "update",
            "save",
            "map_to_designed_state",
            "map_to_designed_action",
            "map_to_designed_reward",
        ):
            if hasattr(model, function_name):
                continue
            function = getattr(model_module, function_name, None)
            if function is None:
                continue
            setattr(model, function_name, MethodType(function, model))

    def _select_model_action(self, player_name, model):
        player_name = self._normalize_compat_player_name(player_name)
        state_mat = self._build_state_dict()[player_name]
        for method_name in ("select_action", "get_action", "act"):
            if not hasattr(model, method_name):
                continue

            method = getattr(model, method_name)
            signature = inspect.signature(method)
            call_kwargs = {}
            can_call = True

            for parameter_name, parameter in signature.parameters.items():
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                if parameter_name in ("state", "state_mat", "observation", "obs"):
                    call_kwargs[parameter_name] = state_mat
                elif parameter_name in ("player", "player_name", "side"):
                    call_kwargs[parameter_name] = player_name
                elif parameter_name == "env":
                    call_kwargs[parameter_name] = self
                elif parameter_name == "epsilon":
                    call_kwargs[parameter_name] = 0.0
                elif parameter.default is inspect._empty:
                    can_call = False
                    break

            if can_call:
                action = method(**call_kwargs)
                if isinstance(action, tuple):
                    return action[0]
                return action

        state = state_mat
        if hasattr(model, "map_to_designed_state"):
            state = model.map_to_designed_state(state_mat)
        else:
            model_module_name = model.__class__.__module__
            model_module = importlib.import_module(model_module_name)
            if hasattr(model_module, "map_to_designed_state"):
                state = model_module.map_to_designed_state(model, state_mat)

        model_module_name = model.__class__.__module__
        package_name = model_module_name.rsplit(".", 1)[0]
        algorithm_module = importlib.import_module(package_name + "._06_algorithm")
        action_mat = algorithm_module.epsilon_greedy_action_selection(
            policy=model.policy,
            state=state,
            epsilon=0.0,
        )
        if hasattr(model, "map_to_designed_action"):
            return model.map_to_designed_action(action_mat)

        model_module = importlib.import_module(model_module_name)
        if hasattr(model_module, "map_to_designed_action"):
            return model_module.map_to_designed_action(model, action_mat)
        return action_mat

    def _set_compat_state_player(self, player_name):
        self.compat_state_player = self._normalize_compat_player_name(player_name)

    def _other_player_name(self, player_name):
        player_name = self._normalize_compat_player_name(player_name)
        if player_name == "player1":
            return "player2"
        return "player1"

    def _repo_root(self):
        return Path(__file__).resolve().parents[1]

    def _load_module_from_repo_file(self, cache_key, relative_path):
        if cache_key in self.module_cache:
            return self.module_cache[cache_key]

        module_path = self._repo_root() / relative_path
        if not module_path.exists():
            raise FileNotFoundError(f"module file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(cache_key, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load module spec: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module_cache[cache_key] = module
        return module

    def _get_custom_state_key_function(self):
        if self.custom_state_key_function is not None:
            return self.custom_state_key_function

        module = None
        for cache_key, relative_path in (
            (
                "compat_model_qlearning__03_state_design",
                "_20_model/qlearning/_03_state_design.py",
            ),
            (
                "compat_model_qlearning_custom_state_design",
                "_20_model/qlearning/custom/state_design.py",
            ),
            (
                "compat_model_qlearning_custom__03_state_design",
                "_20_model/qlearning/custom/_03_state_design.py",
            ),
            (
                "compat_legacy_model_qlearning__03_state_design",
                "model/qlearning/_03_state_design.py",
            ),
            (
                "compat_legacy_model_qlearning_custom_state_design",
                "model/qlearning/custom/state_design.py",
            ),
            (
                "compat_legacy_model_qlearning_custom__03_state_design",
                "model/qlearning/custom/_03_state_design.py",
            ),
        ):
            module_path = self._repo_root() / relative_path
            if not module_path.exists():
                continue
            module = self._load_module_from_repo_file(cache_key, relative_path)
            break
        if module is None:
            raise FileNotFoundError("qlearning state design module not found")
        self.custom_state_key_function = module.calculate_state_key
        return self.custom_state_key_function

    def _resolve_policy_path(self, policy_source):
        policy_text = str(policy_source).strip()
        if not policy_text:
            raise ValueError("policy source is required")
        if policy_text in self.policy_path_cache:
            return self.policy_path_cache[policy_text]

        repo_root = self._repo_root()
        candidates = []

        direct_path = Path(policy_text)
        if direct_path.suffix == ".pt":
            candidates.extend([direct_path, repo_root / direct_path])
        else:
            candidates.extend(
                [
                    direct_path,
                    repo_root / direct_path,
                    Path(policy_text + ".pt"),
                    repo_root / (policy_text + ".pt"),
                ]
            )

        normalized_name = direct_path.name
        if not normalized_name.endswith(".pt"):
            normalized_name = normalized_name + ".pt"

        for base_dir in (
            "_20_model/qlearning/policy_trained",
            "_20_model/qlearning/outputs/policy_trained",
            "old/qlearning/outputs/policy_trained",
            ".models/qlearning/policy_trained",
            "model/qlearning/policy_trained",
            "model/qlearning/outputs/policy_trained",
            "models/qlearning/policy_trained",
            "models/qlearning/outputs/policy_trained",
            "models_old/qlearning/outputs/policy_trained",
        ):
            candidates.append(repo_root / base_dir / normalized_name)

        seen = set()
        for candidate in candidates:
            resolved = Path(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                self.policy_path_cache[policy_text] = resolved
                return resolved

        raise FileNotFoundError(f"policy file not found: {policy_source}")

    def _load_qlearning_policy_payload(self, policy_source):
        policy_path = self._resolve_policy_path(policy_source)
        cache_key = f"qlearning:{policy_path}"
        if cache_key in self.policy_cache:
            return self.policy_cache[cache_key]

        payload = None
        with open(policy_path, "rb") as file:
            try:
                payload = pickle.load(file)
            except Exception:
                payload = None

        if payload is None:
            try:
                import torch
            except ImportError as error:
                raise ImportError("torch is required to load qlearning policies") from error

            if zipfile.is_zipfile(policy_path):
                with zipfile.ZipFile(policy_path) as zip_file:
                    has_data_pickle = False
                    for file_name in zip_file.namelist():
                        if file_name.endswith("data.pkl"):
                            has_data_pickle = True
                            break
                    if not has_data_pickle:
                        loaded = {
                            "path": policy_path,
                            "action_names": list(ACTION_NAMES),
                            "table": {},
                        }
                        self.policy_cache[cache_key] = loaded
                        return loaded

            payload = torch.load(policy_path, map_location="cpu", weights_only=False)

        action_names = payload.get("action_names")
        if action_names is None:
            action_names = list(ACTION_NAMES)
        else:
            action_names = [str(action_name).strip().lower() for action_name in action_names]

        table = {}
        for state_key, qvalues in payload.get("table", {}).items():
            table[tuple(state_key)] = np.asarray(qvalues, dtype=np.float32)

        loaded = {
            "path": policy_path,
            "action_names": action_names,
            "table": table,
        }
        self.policy_cache[cache_key] = loaded
        return loaded

    def _select_action_from_qvalues(self, qvalues, action_names):
        qvalues = np.asarray(qvalues, dtype=np.float32)
        if qvalues.size == 0:
            return "idle"

        best_qvalue = float(np.max(qvalues))
        best_indexes = np.flatnonzero(qvalues == best_qvalue)
        if best_indexes.size == 0:
            return "idle"

        chosen_index = int(best_indexes[random.randrange(best_indexes.size)])
        if chosen_index >= len(action_names):
            return "idle"
        return str(action_names[chosen_index]).strip().lower()

    def _select_policy_action(self, player_name, policy_source):
        player_name = self._normalize_compat_player_name(player_name)
        state_key_func = self._get_custom_state_key_function()
        player_state = self._build_state_dict()[player_name]
        state_key = tuple(state_key_func(player_state))

        if hasattr(policy_source, "select_action"):
            _, action_name = policy_source.select_action(state_key, epsilon=0.0)
            return str(action_name).strip().lower()

        if isinstance(policy_source, dict) and "table" in policy_source:
            loaded_policy = {
                "action_names": list(policy_source.get("action_names", ACTION_NAMES)),
                "table": {
                    tuple(key): list(value)
                    for key, value in policy_source.get("table", {}).items()
                },
            }
        elif isinstance(policy_source, dict):
            loaded_policy = {
                "action_names": list(ACTION_NAMES),
                "table": {
                    tuple(key): list(value)
                    for key, value in policy_source.items()
                },
            }
        else:
            loaded_policy = self._load_qlearning_policy_payload(policy_source)

        action_names = list(loaded_policy["action_names"])
        qvalues = loaded_policy["table"].get(state_key)
        if qvalues is None:
            qvalues = [0.0] * len(action_names)
        return self._select_action_from_qvalues(qvalues, action_names)

    def _compat_controller_action(self, player_name, run_type=None, action=None):
        player_name = self._normalize_compat_player_name(player_name)
        player_id = self._player_name_to_id(player_name)
        configured = self.compat_play_config[player_name]
        controller_name = configured["controller"]

        if run_type == "human":
            controller_name = "human"
        elif run_type == "rule":
            controller_name = "rule"
        elif run_type in ("ai", "action"):
            if action is None and configured["controller"] == "qlearning":
                controller_name = "qlearning"
            else:
                controller_name = "action"

        if controller_name == "action":
            if action is None:
                raise ValueError(f"{player_name} action controller needs action")
            return action

        if controller_name == "qlearning":
            policy_source = configured["policy"]
            if policy_source is None:
                raise ValueError(f"{player_name} qlearning controller needs policy")
            return self._select_policy_action(player_name, policy_source)

        if controller_name == "model":
            model = configured.get("model")
            if model is None:
                raise ValueError(f"{player_name} model controller needs model")
            return self._select_model_action(player_name, model)

        if controller_name == "rule":
            user_input = self.engine.let_computer_decide_user_input(player_id)
            return user_input

        if controller_name == "human":
            if self.engine.viewer is None:
                raise ValueError("human controller needs human render_mode")
            user_input, _ = self.engine.viewer.get_human_input(
                player_id,
                player=self.engine.players[player_id],
            )
            return user_input

        raise ValueError(
            f"{player_name} compatibility controller is unsupported without explicit action: "
            f"{configured['controller']}"
        )

    def _build_compat_reward_materials(self, player_name, score, rewards, actions):
        player_name = self._normalize_compat_player_name(player_name)
        opponent_name = self._other_player_name(player_name)
        next_state_bundle = self.get_training_state_bundle(player_name)
        return self.get_reward_materials(
            train_side=player_name,
            next_state_bundle=next_state_bundle,
            point_reward=rewards[player_name],
            self_action=actions.get(player_name),
            opponent_action=actions.get(opponent_name),
            score=score,
        )

    def _normalize_action_source(self, action_source):
        if hasattr(action_source, "tolist"):
            return action_source.tolist()
        return action_source

    def _compat_run(self, player, run_type=None, action=None, state=None, return_frame=None):
        player_name = self._normalize_compat_player_name(player)
        player_action = self._compat_controller_action(
            player_name,
            run_type=run_type,
            action=action,
        )

        self.compat_pending_actions[player_name] = player_action
        player1_action = self.compat_pending_actions["player1"]
        player2_action = self.compat_pending_actions["player2"]
        next_player_name = self._other_player_name(player_name)
        self._set_compat_state_player(next_player_name)

        if player1_action is None or player2_action is None:
            missing_player_name = None
            if player1_action is None:
                missing_player_name = "player1"
            elif player2_action is None:
                missing_player_name = "player2"

            if missing_player_name is not None:
                missing_controller = self.compat_play_config[
                    missing_player_name]["controller"]
                if missing_controller != "action":
                    self.compat_pending_actions[missing_player_name] = \
                        self._compat_controller_action(missing_player_name)
                    player1_action = self.compat_pending_actions["player1"]
                    player2_action = self.compat_pending_actions["player2"]

        if player1_action is None or player2_action is None:
            score = {
                "player1": self.scores["player1"],
                "player2": self.scores["player2"],
                "rally_done": self.rally_done,
                "match_done": self.match_done,
                "events": None,
            }
            rewards = {"player1": 0.0, "player2": 0.0}
            actions = {
                "player1": player1_action,
                "player2": player2_action,
            }
            reward_materials = self._build_compat_reward_materials(
                next_player_name,
                score,
                rewards,
                actions,
            )
            return score, self.get_state(), reward_materials, self.match_done

        self.compat_pending_actions = {
            "player1": None,
            "player2": None,
        }
        if self.is_log_mode and not return_frame:
            _, score, actions, next_state, rewards = self._run_action_step_log_fast(
                player1_action=player1_action,
                player2_action=player2_action,
                state=state,
            )
        else:
            _, score, actions, next_state, rewards = self._run_action_step(
                player1_action=player1_action,
                player2_action=player2_action,
                state=state,
                return_frame=return_frame,
            )
        self._set_compat_state_player(player_name)
        reward_materials = self._build_compat_reward_materials(
            player_name,
            score,
            rewards,
            actions,
        )
        return score, self.get_state(player=player_name), reward_materials, score["match_done"]

    def _build_action_input(self, player_id, action_source):
        player = self.engine.players[player_id]
        if isinstance(action_source, UserInput):
            user_input = action_source
            action_name = describe_user_input(
                player_id,
                user_input,
                player=player,
            )
            return user_input, action_name

        action_source = self._normalize_action_source(action_source)
        action_name, _ = select_action_name(action_source)
        opponent = self.engine.players[1 - player_id]
        ball = self.engine.ball
        return build_user_input(
            action_name,
            player_id,
            player,
            opponent,
            ball,
            self.direction_memory[player_id],
        )

    def _capture_step_context(self):
        previous_collision_flags = {
            "player1": self.engine.players[0].is_collision_with_ball_happening,
            "player2": self.engine.players[1].is_collision_with_ball_happening,
        }
        previous_ball_x = self.engine.ball.x
        return previous_collision_flags, previous_ball_x

    def _finalize_step(self, previous_collision_flags, previous_ball_x, is_ball_touching_ground):
        self.rally_step_count += 1
        return self._update_match_progress(
            previous_collision_flags,
            previous_ball_x,
            is_ball_touching_ground,
        )

    def _apply_actions(self, player1_action=None, player2_action=None, refresh_expected_landing=True):
        self._invalidate_cached_state()
        previous_collision_flags, previous_ball_x = self._capture_step_context()

        player1_input, player1_action_name = self._build_action_input(0, player1_action)
        player2_input, player2_action_name = self._build_action_input(1, player2_action)

        self._update_direction_memory(0, player1_input)
        self._update_direction_memory(1, player2_input)
        self.last_action_names[0] = player1_action_name
        self.last_action_names[1] = player2_action_name

        is_ball_touching_ground = self.engine.step(
            (player1_input, player2_input),
            refresh_expected_landing=refresh_expected_landing,
        )
        score, rewards = self._finalize_step(
            previous_collision_flags,
            previous_ball_x,
            is_ball_touching_ground,
        )
        actions = {"player1": player1_action_name, "player2": player2_action_name}
        return score, actions, rewards

    def _update_direction_memory(self, player_id, user_input):
        if user_input.x_direction == 0:
            return

        if player_id == 0:
            self.direction_memory[player_id] = user_input.x_direction
        else:
            self.direction_memory[player_id] = -user_input.x_direction

    def _build_state_dict(self):
        if self.state_cache is not None:
            return self.state_cache

        self.engine.update_expected_landing_point()
        self.state_cache = {
            "player1": build_state_view(
                self.engine,
                0,
                self.direction_memory,
                self.last_action_names,
                self.scores,
                self.rally_done,
                self.match_done,
                self.rally_step_count,
            ),
            "player2": build_state_view(
                self.engine,
                1,
                self.direction_memory,
                self.last_action_names,
                self.scores,
                self.rally_done,
                self.match_done,
                self.rally_step_count,
            ),
        }
        return self.state_cache

    def get_score(self):
        return {
            "player1": int(self.scores["player1"]),
            "player2": int(self.scores["player2"]),
        }

    def is_match_done(self):
        return bool(self.match_done)

    def get_match_status(self):
        return self._build_score_snapshot()

    def get_state(self, player=None):
        if player is not None:
            player_name = self._normalize_compat_player_name(player)
            if self.compat_mode_active:
                return self.compat_player_state_views[player_name]
            state = self._build_state_dict()
            return state[player_name]

        if self.compat_mode_active:
            return self.compat_state_view
        state = self._build_state_dict()
        return state

    def get_state_keys(self):
        state = self._build_state_dict()
        return {"player1": state["player1"]["key"], "player2": state["player2"]["key"]}

    def get_training_state_bundle(self, player_name):
        player_name = self._normalize_compat_player_name(player_name)
        if player_name in self.training_state_bundle_cache:
            return self.training_state_bundle_cache[player_name]

        player_id = self._player_name_to_id(player_name)
        self.engine.update_expected_landing_point()
        state_bundle = build_training_state_bundle(
            self.engine,
            player_id,
            self.direction_memory,
            self.last_action_names,
            self.scores,
            self.rally_step_count,
        )
        self.training_state_bundle_cache[player_name] = state_bundle
        return state_bundle

    def get_training_state_materials(self, player_name="player1", state_bundle=None):
        if state_bundle is None:
            state_bundle = self.get_training_state_bundle(player_name)

        raw_state = state_bundle["raw"]
        self_raw = raw_state["self"]
        opponent_raw = raw_state["opponent"]
        ball_raw = raw_state["ball"]

        materials = {}
        materials["self_position"] = (self_raw["x"], self_raw["y"])
        materials["self_action_name"] = self_raw["action_name"]
        materials["self_direction"] = self_raw["direction"]
        materials["self_spike_used"] = self_raw["spike_used"]
        materials["opponent_position"] = (opponent_raw["x"], opponent_raw["y"])
        materials["opponent_action_name"] = opponent_raw["action_name"]
        materials["opponent_direction"] = opponent_raw["direction"]
        materials["opponent_spike_used"] = opponent_raw["spike_used"]
        materials["ball_position"] = (ball_raw["x"], ball_raw["y"])
        materials["ball_velocity"] = (ball_raw["x_velocity"], ball_raw["y_velocity"])
        materials["ball_side"] = ball_raw["side"]
        materials["expected_landing_x"] = ball_raw["expected_landing_x"]
        materials["rally_frame"] = raw_state["rally_step"]
        return materials

    def get_reward_materials(
        self,
        train_side="player1",
        next_state_bundle=None,
        point_reward=0.0,
        self_action=None,
        opponent_action=None,
        score=None,
    ):
        if next_state_bundle is None:
            next_state_bundle = self.get_training_state_bundle(train_side)

        raw_state = next_state_bundle["raw"]
        self_raw = raw_state["self"]
        opponent_raw = raw_state["opponent"]
        ball_raw = raw_state["ball"]

        self_action = self._normalize_action_source(self_action)
        opponent_action = self._normalize_action_source(opponent_action)
        self_action_name, _ = select_action_name(self_action)
        opponent_action_name, _ = select_action_name(opponent_action)

        point_scored = float(point_reward > 0.0)
        point_lost = float(point_reward < 0.0)
        rally_frame = float(raw_state["rally_step"])
        rally_total_frames_until_point = 0.0
        if point_scored or point_lost:
            rally_total_frames_until_point = rally_frame

        other_side = "player1"
        if train_side == "player1":
            other_side = "player2"

        match_won = 0.0
        if score is not None and score["match_done"] and score[train_side] > score[other_side]:
            match_won = 1.0

        materials = {}
        materials["opponent_position"] = (opponent_raw["x"], opponent_raw["y"])
        materials["self_position"] = (self_raw["x"], self_raw["y"])
        materials["ball_position"] = (ball_raw["x"], ball_raw["y"])
        materials["self_action_name"] = self_action_name
        materials["opponent_action_name"] = opponent_action_name
        materials["point_result"] = {"scored": point_scored, "lost": point_lost}
        materials["match_result"] = {"won": match_won}
        materials["rally_total_frames_until_point"] = rally_total_frames_until_point
        return materials

    def _update_match_progress(self, previous_collision_flags, previous_ball_x, is_ball_touching_ground):
        rewards = {"player1": 0.0, "player2": 0.0}
        events = {
            "touch": {"player1": False, "player2": False},
            "crossed_net_to": None,
            "scorer": None,
            "rally_step": self.rally_step_count,
            "timeout": False,
        }

        if (
            self.engine.players[0].is_collision_with_ball_happening
            and not previous_collision_flags["player1"]
        ):
            events["touch"]["player1"] = True

        if (
            self.engine.players[1].is_collision_with_ball_happening
            and not previous_collision_flags["player2"]
        ):
            events["touch"]["player2"] = True

        if previous_ball_x <= GROUND_HALF_WIDTH < self.engine.ball.x:
            events["crossed_net_to"] = "player2"
        elif previous_ball_x >= GROUND_HALF_WIDTH > self.engine.ball.x:
            events["crossed_net_to"] = "player1"

        if is_ball_touching_ground:
            self.rally_done = True

            if self.engine.ball.punch_effect_x > GROUND_HALF_WIDTH:
                scorer = "player1"
                loser = "player2"
            else:
                scorer = "player2"
                loser = "player1"

            self._set_next_serve_side(scorer=scorer)

            self.scores[scorer] += 1
            rewards[scorer] = 10.0
            rewards[loser] = -10.0
            events["scorer"] = scorer

            if self.scores[scorer] >= self.target_score:
                self.match_done = True
                self._set_match_end_player_states()
            else:
                self._pause_after_point()
                self._reset_rally(clear_flags=False)

        if (
            not self.match_done
            and not self.rally_done
            and self.rally_step_limit is not None
            and self.rally_step_count >= int(self.rally_step_limit)
        ):
            self.rally_done = True
            self.match_done = True
            events["timeout"] = True
            self._set_match_end_player_states()

        if self.engine.viewer is not None and (not self.rally_done or self.match_done):
            self._refresh_viewer()

        self.last_match_events = self._clone_events(events)
        score = self._build_score_snapshot(events=events)
        return score, rewards

    def run_training_step(
        self,
        train_side="player1",
        action=None,
        opponent_action=None,
        opponent=None,
        opponent_policy=None,
    ):
        train_player_id = self._player_name_to_id(train_side)
        opponent_player_id = 1 - train_player_id

        if opponent_action is None and (opponent is not None or opponent_policy is not None):
            opponent_name = self._player_id_to_name(opponent_player_id)
            controller_name = "rule"
            if opponent is not None:
                controller_name = str(opponent).strip().lower()

            if controller_name in ("rule", "human"):
                opponent_action = self._compat_controller_action(
                    opponent_name,
                    run_type=controller_name,
                )
            elif controller_name in ("qlearning", "ai", "self"):
                if opponent_policy is None:
                    raise ValueError(f"{opponent_name} qlearning opponent needs policy")
                opponent_action = self._select_policy_action(opponent_name, opponent_policy)
            else:
                raise ValueError(f"unsupported opponent controller: {controller_name}")

        if not self.started or self.match_done:
            self.reset(return_state=False)

        self.rally_done = False

        if train_player_id == 0:
            score, actions, rewards = self._apply_actions(
                player1_action=action,
                player2_action=opponent_action,
                refresh_expected_landing=False,
            )
        else:
            score, actions, rewards = self._apply_actions(
                player1_action=opponent_action,
                player2_action=action,
                refresh_expected_landing=False,
            )

        next_state_bundle = self.get_training_state_bundle(train_side)
        return score, actions, next_state_bundle, rewards

    def run_training_material_step(
        self,
        train_side="player1",
        action=None,
        opponent_action=None,
        opponent=None,
        opponent_policy=None,
    ):
        score, actions, next_state_bundle, rewards = self.run_training_step(
            train_side=train_side,
            action=action,
            opponent_action=opponent_action,
            opponent=opponent,
            opponent_policy=opponent_policy,
        )

        other_side = "player1"
        if train_side == "player1":
            other_side = "player2"

        state_materials = self.get_training_state_materials(train_side, next_state_bundle)
        reward_materials = self.get_reward_materials(
            train_side=train_side,
            next_state_bundle=next_state_bundle,
            point_reward=rewards[train_side],
            self_action=actions[train_side],
            opponent_action=actions[other_side],
            score=score,
        )
        result = {
            "score": score,
            "actions": actions,
            "rewards": rewards,
            "done": score["match_done"],
            "state_bundle": next_state_bundle,
        }
        return result, state_materials, reward_materials

    def _run_action_step(
        self,
        player1_action=None,
        player2_action=None,
        state=None,
        return_frame=None,
    ):
        if state == "init" or not self.started or self.match_done:
            self.reset(return_state=False)

        self.rally_done = False
        if return_frame is None:
            return_frame = self.render_mode != "log"

        score, actions, rewards = self._apply_actions(
            player1_action=player1_action,
            player2_action=player2_action,
        )

        frame = None
        if return_frame:
            frame = self._get_frame()

        next_state = self._build_state_dict()
        return frame, score, actions, next_state, rewards

    def _run_action_step_log_fast(
        self,
        player1_action=None,
        player2_action=None,
        state=None,
    ):
        if state == "init" or not self.started or self.match_done:
            self.reset(return_state=False)

        self.rally_done = False
        score, actions, rewards = self._apply_actions(
            player1_action=player1_action,
            player2_action=player2_action,
        )
        next_state = self._build_state_dict()
        return None, score, actions, next_state, rewards

    def run(
        self,
        player1_action=None,
        player2_action=None,
        state=None,
        return_frame=None,
        player=None,
        run_type=None,
        action=None,
        **kwargs,
    ):
        player1 = kwargs.pop("player1", None)
        player2 = kwargs.pop("player2", None)
        player1_policy = kwargs.pop("player1_policy", None)
        player2_policy = kwargs.pop("player2_policy", None)

        if player1 is not None or player2 is not None:
            self.compat_mode_active = True
            self._set_compat_play_config(
                player1=player1,
                player1_policy=player1_policy,
                player2=player2,
                player2_policy=player2_policy,
            )
            frame, score, actions, next_state, rewards = self._run_action_step(
                player1_action=self._compat_controller_action("player1"),
                player2_action=self._compat_controller_action("player2"),
                state=state,
                return_frame=return_frame,
            )
            self._set_compat_state_player("player1")
            return frame, score, actions, next_state, rewards

        if player is not None or run_type is not None:
            return self._compat_run(
                player=player,
                run_type=run_type,
                action=action,
                state=state,
                return_frame=return_frame,
            )

        if self.is_log_mode and not return_frame:
            return self._run_action_step_log_fast(
                player1_action=player1_action,
                player2_action=player2_action,
                state=state,
            )

        return self._run_action_step(
            player1_action=player1_action,
            player2_action=player2_action,
            state=state,
            return_frame=return_frame,
        )

    def step(
        self,
        player1_action=None,
        player2_action=None,
        state=None,
        return_frame=None,
    ):
        return self.run(
            player1_action=player1_action,
            player2_action=player2_action,
            state=state,
            return_frame=return_frame,
        )

    def run_play_step(
        self,
        player1=None,
        player2=None,
        player1_policy=None,
        player2_policy=None,
        state=None,
        return_frame=None,
    ):
        if (
            self.is_log_mode
            and player1 is None
            and player2 is None
            and player1_policy is None
            and player2_policy is None
            and not return_frame
        ):
            frame, score, actions, next_state, rewards = self._run_action_step_log_fast(
                player1_action=self._compat_controller_action("player1"),
                player2_action=self._compat_controller_action("player2"),
                state=state,
            )
            self._set_compat_state_player("player1")
            return {
                "frame": frame,
                "score": score,
                "actions": actions,
                "state": next_state,
                "rewards": rewards,
                "done": bool(score["match_done"]),
            }

        if (
            player1 is not None
            or player2 is not None
            or player1_policy is not None
            or player2_policy is not None
        ):
            if player1 is None:
                if self.compat_play_config["player1"]["controller"] == "model":
                    player1 = self.compat_play_config["player1"]["model"]
                else:
                    player1 = self.compat_play_config["player1"]["controller"]
            if player2 is None:
                if self.compat_play_config["player2"]["controller"] == "model":
                    player2 = self.compat_play_config["player2"]["model"]
                else:
                    player2 = self.compat_play_config["player2"]["controller"]
            if player1_policy is None:
                player1_policy = self.compat_play_config["player1"]["policy"]
            if player2_policy is None:
                player2_policy = self.compat_play_config["player2"]["policy"]

            self.compat_mode_active = True
            self._set_compat_play_config(
                player1=player1,
                player1_policy=player1_policy,
                player2=player2,
                player2_policy=player2_policy,
            )

        frame, score, actions, next_state, rewards = self._run_action_step(
            player1_action=self._compat_controller_action("player1"),
            player2_action=self._compat_controller_action("player2"),
            state=state,
            return_frame=return_frame,
        )
        self._set_compat_state_player("player1")
        return {
            "frame": frame,
            "score": score,
            "actions": actions,
            "state": next_state,
            "rewards": rewards,
            "done": bool(score["match_done"]),
        }

    def get_play_result(self):
        if (
            self.render_mode == "human"
            and self.compat_mode_active
            and self.compat_pending_actions["player1"] is None
            and self.compat_pending_actions["player2"] is None
        ):
            return self.run_play_step()

        score = self.get_match_status()
        done = bool(score["match_done"])
        if not done and isinstance(score.get("events"), dict):
            done = bool(score["events"].get("timeout", False))
        return {
            "score": score,
            "done": done,
        }

    def close(self):
        self.engine.close()

    def wait_until_exit(self):
        if self.engine.viewer is None:
            return None
        return self.engine.viewer.wait_for_command()

    def set(
        self,
        player1=None,
        player2=None,
        random_serve=None,
        randomize_serve=None,
        return_state=True,
    ):
        return self.reset(
            player1=player1,
            player2=player2,
            random_serve=random_serve,
            randomize_serve=randomize_serve,
            return_state=return_state,
        )

    def wait_key_for_start(self, key=None):
        if self.engine.viewer is None:
            return None

        self.reset_viewer_commands()
        self.show_play_ready(
            self.compat_play_config["player1"]["controller"],
            self.compat_play_config["player1"]["policy"],
            self.compat_play_config["player2"]["controller"],
            self.compat_play_config["player2"]["policy"],
        )
        return self.wait_for_command(allow_start=True)

    def wait_key_for_terminate(self, key=None):
        if self.engine.viewer is None:
            return None

        if self.render_mode == "human":
            self.engine.render("human")
        return self.consume_viewer_command()

    def wait_for_command(self, allow_restart=False, allow_start=False):
        if self.engine.viewer is None:
            return None
        return self.engine.viewer.wait_for_command(
            allow_restart=allow_restart,
            allow_start=allow_start,
        )

    def consume_viewer_command(self):
        if self.engine.viewer is None:
            return None
        return self.engine.viewer.consume_command()

    def reset_viewer_commands(self):
        if self.engine.viewer is None:
            return
        self.engine.viewer.reset_runtime_flags()
