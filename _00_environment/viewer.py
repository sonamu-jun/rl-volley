import base64
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .sdl import configure_sdl_video_driver


configure_sdl_video_driver()

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import pygame

from .audio import ViewerAudio
from .constants import GROUND_HALF_WIDTH
from .constants import GROUND_HEIGHT
from .constants import GROUND_WIDTH
from .constants import HUMAN_DISPLAY_HEIGHT
from .constants import HUMAN_DISPLAY_WIDTH
from .input import UserInput


IMAGES_PATH = Path(__file__).with_name("images.json")
CONF_PATH = Path(__file__).resolve().parents[1] / "_10_config" / "conf.py"

with IMAGES_PATH.open(encoding="utf-8") as image_file:
    ENCODED_IMAGES = json.load(image_file)


def get_image_index(state, frame_number):
    if state < 4:
        return 5 * state + frame_number
    if state == 4:
        return 17 + frame_number
    return 18 + 5 * (state - 5) + frame_number


def load_image(image_key):
    image_bytes = base64.b64decode(ENCODED_IMAGES[image_key])
    return pygame.image.load(io.BytesIO(image_bytes))


def is_bnw_mode_enabled():
    bnw_mode, _ = load_bnw_config()
    return bnw_mode


def load_bnw_config():
    try:
        conf_text = CONF_PATH.read_text(encoding="utf-8")
    except OSError:
        return True, "301"

    bnw_mode_false_match = re.search(
        r"^\s*BNW_MODE\s*=\s*False\b",
        conf_text,
        flags=re.MULTILINE,
    )
    password_match = re.search(
        r"^\s*BNW_MODE_PW\s*=\s*([^\n#]+)",
        conf_text,
        flags=re.MULTILINE,
    )

    password_value = "301"
    if password_match is not None:
        password_value = password_match.group(1).strip().strip("'\"")

    bnw_mode_enabled = not (
        bnw_mode_false_match is not None
        and password_value == "301"
    )

    return bnw_mode_enabled, password_value


class Viewer:
    def __init__(self, engine):
        pygame.init()

        self.engine = engine
        self.bnw_mode, self.bnw_mode_password = load_bnw_config()
        self.audio_enabled_by_config = (
            not self.bnw_mode and str(self.bnw_mode_password) == "301"
        )
        self.bnw_code_buffer = ""
        self.headless = True
        self.closed_requested = False
        self.pending_command = None
        self.pressed_keys = set()
        self.human_controllers = {
            "player1": False,
            "player2": False,
        }

        self.screen = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))
        self.display_surface = None
        self.background = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))
        self.clock = pygame.time.Clock()
        self.small_font = pygame.font.SysFont(None, 26)
        self.info_font = pygame.font.SysFont(None, 20)
        self.large_font = pygame.font.SysFont(None, 36)
        self.medium_font = pygame.font.SysFont(None, 30)
        self.secret_font = pygame.font.SysFont(None, 10)
        self.score_text = "0 : 0"
        self.status_text = ""
        self.status_command_text = ""
        self.player_labels = {
            "player1": {"title": "", "detail": ""},
            "player2": {"title": "", "detail": ""},
        }
        self.overlay_dirty = True
        self.overlay_surface = pygame.Surface(
            (GROUND_WIDTH, GROUND_HEIGHT), pygame.SRCALPHA
        )
        self.ball_punch_cache = {}
        self.audio = None
        self.current_score = None
        self.game_end_animation_frame = 0
        self.last_player_states = {
            "player1": int(engine.players[0].state),
            "player2": int(engine.players[1].state),
        }
        self.last_power_hit_active = False
        self.last_ground_effect_signature = None

        self.ball_images = []
        self.player_images = []
        self.ball_hyper_image = None
        self.ball_trail_image = None
        self.ball_punch_image = None
        self.shadow_image = None
        self.player_images_flipped = []
        self.standard_assets_loaded = False

        self.ball = engine.ball
        self.player1 = engine.players[0]
        self.player2 = engine.players[1]

        if self.bnw_mode:
            self._build_bnw_background()
        else:
            self._load_standard_assets()

    def init_screen(self):
        pygame.display.init()
        self.display_surface = pygame.display.set_mode(
            (HUMAN_DISPLAY_WIDTH, HUMAN_DISPLAY_HEIGHT)
        )
        self.headless = False
        self._ensure_audio()
        self.screen = self.screen.convert()
        self.background = self.background.convert()
        self.overlay_surface = self.overlay_surface.convert_alpha()
        if not self.bnw_mode:
            self._convert_standard_assets_for_display()

    def _present_display(self):
        if self.headless or self.display_surface is None:
            return

        # 내부 게임 좌표계는 그대로 두고, human 창에만 확대해서 보여준다.
        pygame.transform.scale(
            self.screen,
            (HUMAN_DISPLAY_WIDTH, HUMAN_DISPLAY_HEIGHT),
            self.display_surface,
        )

    def reset_runtime_flags(self):
        self.closed_requested = False
        self.pending_command = None
        self.pressed_keys.clear()

    def reset_input_state(self):
        self.pressed_keys.clear()
        self.bnw_code_buffer = ""

    def _ensure_audio(self):
        if self.audio is not None:
            return

        self.audio = ViewerAudio(enabled=self.audio_enabled_by_config)
        if self.audio_enabled_by_config and not self.audio.available:
            if self.audio.assets_found:
                print(f"[audio] disabled: mixer init failed ({self.audio.init_error})")
            return

        self.audio.play_bgm()
        if self.audio_enabled_by_config:
            bgm_name = self.audio.bgm_path.name if self.audio.bgm_path is not None else "-"
            loaded_effects = ",".join(self.audio.loaded_effect_names) if self.audio.loaded_effect_names else "-"
            driver_name = self.audio.audio_driver or "default"
            print(
                f"[audio] enabled: driver={driver_name} bgm={bgm_name} effects={loaded_effects}"
            )
        if self.audio_enabled_by_config and self.audio.init_error:
            print(f"[audio] warning: {self.audio.init_error}")

    def set_human_controllers(self, player1_human, player2_human):
        self.human_controllers["player1"] = bool(player1_human)
        self.human_controllers["player2"] = bool(player2_human)
        self.overlay_dirty = True

    def _quit_key_text(self):
        return "Q quit"

    def _control_hint_text(self):
        hints = []
        if self.human_controllers["player1"]:
            hints.append("1P U/H/J/K/Z")
        if self.human_controllers["player2"]:
            hints.append("2P Arrows/Enter")
        return "  ".join(hints)

    def set_match_status(self, player1_score, player2_score, winner_text, command_text=None):
        previous_winner_text = self.status_text
        self.current_score = (int(player1_score), int(player2_score))
        self.score_text = f"{player1_score} : {player2_score}"
        if winner_text is None:
            self.status_text = ""
            self.status_command_text = ""
            self.game_end_animation_frame = 0
        else:
            self.status_text = winner_text
            if command_text is None:
                self.status_command_text = "Q quit  C restart"
            else:
                self.status_command_text = str(command_text)
            if winner_text != previous_winner_text and self._is_game_end_status():
                self.game_end_animation_frame = 0
        self.overlay_dirty = True

    def set_player_labels(self, player1_title, player1_detail, player2_title, player2_detail):
        self.player_labels["player1"] = {
            "title": str(player1_title),
            "detail": str(player1_detail),
        }
        self.player_labels["player2"] = {
            "title": str(player2_title),
            "detail": str(player2_detail),
        }
        self.overlay_dirty = True

    def _draw_text(self, target_surface, text, font, y_value, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(GROUND_WIDTH // 2, y_value))
        target_surface.blit(text_surface, text_rect)

    def _draw_text_at_x(self, target_surface, text, font, x_value, y_value, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(x_value, y_value))
        target_surface.blit(text_surface, text_rect)

    def _draw_scaled_text(self, target_surface, text, font, center_x, center_y, color, scale):
        text_surface = font.render(text, True, color)
        if abs(scale - 1.0) > 0.01:
            scaled_width = max(1, int(text_surface.get_width() * scale))
            scaled_height = max(1, int(text_surface.get_height() * scale))
            text_surface = pygame.transform.smoothscale(
                text_surface,
                (scaled_width, scaled_height),
            )
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        target_surface.blit(text_surface, text_rect)

    def _is_game_end_status(self):
        return self.status_text in {"PLAYER 1 WIN", "PLAYER 2 WIN", "DRAW"}

    def _get_display_frame_number(self, player):
        if getattr(player, "game_ended", False) and int(player.state) in (5, 6):
            return min(self.game_end_animation_frame // 6, 4)
        return int(player.frame_number)

    def _draw_player_label(self, target_surface, player_key, center_x, title_color):
        label = self.player_labels[player_key]
        if label["title"]:
            self._draw_text_at_x(
                target_surface,
                label["title"],
                self.info_font,
                center_x,
                18,
                title_color,
            )
        if label["detail"]:
            self._draw_text_at_x(
                target_surface,
                label["detail"],
                self.info_font,
                center_x,
                38,
                (0, 0, 0) if self.bnw_mode else (255, 255, 255),
            )

    def _build_bnw_background(self):
        self.background.fill((255, 255, 255))
        ground_y = 264
        sand_top_y = ground_y + 2
        net_top_y = 176
        net_bottom_y = 264
        net_left_x = GROUND_HALF_WIDTH - 5
        net_right_x = GROUND_HALF_WIDTH + 5

        pygame.draw.line(self.background, (0, 0, 0), (0, ground_y), (GROUND_WIDTH, ground_y), 2)

        for x_value in range(4, GROUND_WIDTH, 9):
            sand_y = sand_top_y + (x_value * 7 % 45)
            sand_color = (170, 170, 170) if x_value % 3 else (80, 80, 80)
            self.background.set_at((x_value, sand_y), sand_color)
            if sand_y + 9 < GROUND_HEIGHT and x_value + 3 < GROUND_WIDTH:
                self.background.set_at((x_value + 3, sand_y + 9), sand_color)

        for x_value in range(10, GROUND_WIDTH, 32):
            pygame.draw.line(
                self.background,
                (120, 120, 120),
                (x_value, sand_top_y + 16),
                (x_value + 8, sand_top_y + 18),
                1,
            )

        pygame.draw.line(self.background, (0, 0, 0), (net_left_x, net_top_y), (net_left_x, net_bottom_y), 2)
        pygame.draw.line(self.background, (0, 0, 0), (net_right_x, net_top_y), (net_right_x, net_bottom_y), 2)
        pygame.draw.line(self.background, (0, 0, 0), (net_left_x - 2, net_top_y), (net_right_x + 2, net_top_y), 2)

        for y_value in range(net_top_y + 6, net_bottom_y, 8):
            pygame.draw.line(
                self.background,
                (70, 70, 70),
                (net_left_x + 1, y_value),
                (net_right_x - 1, y_value),
                1,
            )

        for x_value in range(net_left_x + 2, net_right_x, 3):
            pygame.draw.line(
                self.background,
                (70, 70, 70),
                (x_value, net_top_y + 2),
                (x_value, net_bottom_y),
                1,
            )

        if self.bnw_mode_password:
            hidden_text = self.secret_font.render(
                str(self.bnw_mode_password),
                True,
                (228, 228, 228),
            )
            self.background.blit(
                hidden_text,
                (GROUND_WIDTH - hidden_text.get_width() - 5, GROUND_HEIGHT - hidden_text.get_height() - 5),
            )

    def _load_standard_assets(self):
        if self.standard_assets_loaded:
            return

        self.ball_images = []
        self.player_images = []

        for index in range(5):
            self.ball_images.append(load_image(f"ball_{index}"))

        for state_index in range(7):
            for frame_index in range(5):
                if state_index == 3 and frame_index == 2:
                    break
                if state_index == 4 and frame_index == 1:
                    break
                self.player_images.append(load_image(f"pikachu_{state_index}_{frame_index}"))

        self.ball_hyper_image = load_image("ball_hyper")
        self.ball_trail_image = load_image("ball_trail")
        self.ball_punch_image = load_image("ball_punch")
        self.shadow_image = load_image("shadow")
        self.player_images_flipped = [
            pygame.transform.flip(image, True, False)
            for image in self.player_images
        ]

        ground_red_image = load_image("ground_red")
        ground_yellow_image = load_image("ground_yellow")
        ground_line_image = load_image("ground_line")
        ground_line_left_image = load_image("ground_line_leftmost")
        ground_line_right_image = load_image("ground_line_rightmost")
        mountain_image = load_image("mountain")
        sky_blue_image = load_image("sky_blue")
        net_pillar_image = load_image("net_pillar")
        net_pillar_top_image = load_image("net_pillar_top")

        self.background = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))

        for column_index in range(27):
            self.background.blit(ground_red_image, (16 * column_index, 248))

        for column_index in range(27):
            for row_index in range(2):
                self.background.blit(
                    ground_yellow_image,
                    (16 * column_index, 280 + 16 * row_index),
                )

        for column_index in range(1, 26):
            self.background.blit(ground_line_image, (16 * column_index, 264))

        self.background.blit(ground_line_left_image, (0, 264))
        self.background.blit(ground_line_right_image, (416, 264))
        self.background.blit(mountain_image, (0, 188))

        for column_index in range(27):
            for row_index in range(12):
                self.background.blit(sky_blue_image, (16 * column_index, 16 * row_index))

        self.background.blit(net_pillar_top_image, (213, 176))

        for row_index in range(12):
            self.background.blit(net_pillar_image, (213, 184 + 8 * row_index))

        self.standard_assets_loaded = True

    def _convert_standard_assets_for_display(self):
        self.ball_images = [image.convert_alpha() for image in self.ball_images]
        self.player_images = [image.convert_alpha() for image in self.player_images]
        self.player_images_flipped = [
            image.convert_alpha() for image in self.player_images_flipped
        ]
        self.ball_hyper_image = self.ball_hyper_image.convert_alpha()
        self.ball_trail_image = self.ball_trail_image.convert_alpha()
        self.ball_punch_image = self.ball_punch_image.convert_alpha()
        self.shadow_image = self.shadow_image.convert_alpha()

    def _maybe_disable_bnw_mode(self, key_value):
        if not self.bnw_mode or not self.bnw_mode_password:
            return

        digit_map = {
            pygame.K_0: "0",
            pygame.K_1: "1",
            pygame.K_2: "2",
            pygame.K_3: "3",
            pygame.K_4: "4",
            pygame.K_5: "5",
            pygame.K_6: "6",
            pygame.K_7: "7",
            pygame.K_8: "8",
            pygame.K_9: "9",
            pygame.K_KP0: "0",
            pygame.K_KP1: "1",
            pygame.K_KP2: "2",
            pygame.K_KP3: "3",
            pygame.K_KP4: "4",
            pygame.K_KP5: "5",
            pygame.K_KP6: "6",
            pygame.K_KP7: "7",
            pygame.K_KP8: "8",
            pygame.K_KP9: "9",
        }
        digit_value = digit_map.get(key_value)
        if digit_value is None:
            return

        password_value = str(self.bnw_mode_password)
        self.bnw_code_buffer = (self.bnw_code_buffer + digit_value)[-len(password_value):]
        if self.bnw_code_buffer == password_value:
            self._disable_bnw_mode()
            self.bnw_code_buffer = ""

    def _disable_bnw_mode(self):
        if not self.bnw_mode:
            return

        self._load_standard_assets()
        self.bnw_mode = False
        if not self.headless:
            self.background = self.background.convert()
            self._convert_standard_assets_for_display()
        self.overlay_dirty = True

    def _handle_keydown(self, key_value, allow_restart=False, allow_start=False):
        self.pressed_keys.add(key_value)
        self._maybe_disable_bnw_mode(key_value)
        if key_value in (pygame.K_ESCAPE, pygame.K_q):
            self.closed_requested = True
            self.pending_command = "quit"
        elif allow_restart and key_value == pygame.K_c:
            self.pending_command = "restart"
        elif allow_start and key_value == pygame.K_s:
            self.pending_command = "start"

    def _draw_bnw_face(self, head_center, expression, facing_direction):
        head_x, head_y = head_center
        color = (0, 0, 0)
        eye_y = head_y - 2
        left_eye_x = head_x - 3
        right_eye_x = head_x + 3

        if expression == "blink":
            pygame.draw.line(self.screen, color, (left_eye_x - 1, eye_y), (left_eye_x + 1, eye_y), 1)
            pygame.draw.line(self.screen, color, (right_eye_x - 1, eye_y), (right_eye_x + 1, eye_y), 1)
        elif expression == "determined":
            pygame.draw.line(self.screen, color, (left_eye_x - 2, eye_y - 1), (left_eye_x + 1, eye_y), 1)
            pygame.draw.line(self.screen, color, (right_eye_x - 1, eye_y), (right_eye_x + 2, eye_y - 1), 1)
            pygame.draw.circle(self.screen, color, (left_eye_x, eye_y + 1), 1)
            pygame.draw.circle(self.screen, color, (right_eye_x, eye_y + 1), 1)
        else:
            pygame.draw.circle(self.screen, color, (left_eye_x, eye_y), 1)
            pygame.draw.circle(self.screen, color, (right_eye_x, eye_y), 1)

        if expression == "smile":
            pygame.draw.arc(self.screen, color, (head_x - 4, head_y + 1, 8, 5), 0.2, math.pi - 0.2, 1)
        elif expression == "frown":
            pygame.draw.arc(self.screen, color, (head_x - 4, head_y + 4, 8, 5), math.pi + 0.2, 2 * math.pi - 0.2, 1)
        elif expression == "surprised":
            pygame.draw.circle(self.screen, color, (head_x, head_y + 4), 2, 1)
        elif expression == "grim":
            pygame.draw.line(self.screen, color, (head_x - 3, head_y + 4), (head_x + 3, head_y + 4), 1)
        else:
            offset_x = 1 if facing_direction >= 0 else -1
            pygame.draw.arc(
                self.screen,
                color,
                (head_x - 4 + offset_x, head_y + 2, 8, 4),
                0.3,
                math.pi - 0.3,
                1,
            )

    def _draw_bnw_torso(self, torso_rect):
        pygame.draw.ellipse(self.screen, (245, 245, 245), torso_rect)
        pygame.draw.ellipse(self.screen, (0, 0, 0), torso_rect, 2)

    def _draw_bnw_leg(self, start_point, end_point):
        pygame.draw.line(self.screen, (0, 0, 0), start_point, end_point, 3)
        pygame.draw.ellipse(
            self.screen,
            (0, 0, 0),
            (end_point[0] - 3, end_point[1] - 1, 6, 4),
            1,
        )

    def _draw_bnw_arm(self, shoulder_point, hand_point):
        pygame.draw.line(self.screen, (0, 0, 0), shoulder_point, hand_point, 2)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            hand_point,
            (hand_point[0] + 2, hand_point[1] + 1),
            1,
        )

    def _draw_bnw_back_spikes(self, torso_rect, facing_direction):
        back_x = torso_rect.centerx - facing_direction * 6
        spike_points = [
            [
                (back_x - facing_direction * 3, torso_rect.y + 3),
                (back_x, torso_rect.y - 4),
                (back_x + facing_direction * 3, torso_rect.y + 4),
            ],
            [
                (back_x - facing_direction * 8, torso_rect.y + 6),
                (back_x - facing_direction * 5, torso_rect.y - 1),
                (back_x - facing_direction * 2, torso_rect.y + 7),
            ],
            [
                (back_x - facing_direction * 13, torso_rect.y + 9),
                (back_x - facing_direction * 10, torso_rect.y + 2),
                (back_x - facing_direction * 7, torso_rect.y + 10),
            ],
        ]
        for spike in spike_points:
            pygame.draw.polygon(self.screen, (235, 235, 235), spike)
            pygame.draw.polygon(self.screen, (0, 0, 0), spike, 1)

    def _draw_bnw_tail(self, base_point, facing_direction, lift_amount):
        mid_point = (
            base_point[0] - facing_direction * 8,
            base_point[1] - lift_amount,
        )
        tip_point = (
            base_point[0] - facing_direction * 13,
            base_point[1] - lift_amount - 4,
        )
        pygame.draw.line(self.screen, (0, 0, 0), base_point, mid_point, 4)
        pygame.draw.line(self.screen, (0, 0, 0), mid_point, tip_point, 3)
        pygame.draw.circle(self.screen, (0, 0, 0), tip_point, 1)

    def _draw_bnw_player_head(self, head_center, expression, facing_direction):
        head_x, head_y = head_center
        snout_rect = pygame.Rect(
            head_x + facing_direction * 5 - 5,
            head_y + 1,
            12,
            8,
        )
        crest_points = [
            [
                (head_x - facing_direction * 8, head_y - 2),
                (head_x - facing_direction * 12, head_y - 11),
                (head_x - facing_direction * 4, head_y - 6),
            ],
            [
                (head_x - facing_direction * 3, head_y - 5),
                (head_x - facing_direction * 7, head_y - 14),
                (head_x + facing_direction * 1, head_y - 9),
            ],
        ]

        for crest in crest_points:
            pygame.draw.polygon(self.screen, (235, 235, 235), crest)
            pygame.draw.polygon(self.screen, (0, 0, 0), crest, 1)

        pygame.draw.circle(self.screen, (255, 255, 255), head_center, 9)
        pygame.draw.circle(self.screen, (0, 0, 0), head_center, 9, 2)
        pygame.draw.ellipse(self.screen, (255, 255, 255), snout_rect)
        pygame.draw.ellipse(self.screen, (0, 0, 0), snout_rect, 1)
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            (head_x + facing_direction * 10, head_y + 4),
            1,
        )
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            (head_x + facing_direction * 8, head_y + 4),
            1,
        )
        self._draw_bnw_face(head_center, expression, facing_direction)

    def _draw_bnw_player(self, player):
        x_value = int(player.x)
        y_value = int(player.y)
        color = (0, 0, 0)
        display_frame = self._get_display_frame_number(player)

        if player.state == 5:
            sway = (display_frame % 2) * 2
            head_center = (x_value, y_value - 22)
            hip_y = y_value + 4
            pygame.draw.circle(self.screen, color, head_center, 7, 2)
            pygame.draw.arc(
                self.screen,
                color,
                (head_center[0] - 4, head_center[1] + 1, 8, 5),
                0.2,
                math.pi - 0.2,
                1,
            )
            pygame.draw.line(self.screen, color, (x_value, y_value - 13), (x_value, hip_y), 2)
            pygame.draw.line(self.screen, color, (x_value, hip_y), (x_value - 9, y_value + 20), 2)
            pygame.draw.line(self.screen, color, (x_value, hip_y), (x_value + 9, y_value + 20), 2)
            pygame.draw.line(self.screen, color, (x_value, y_value - 8), (x_value - 10 - sway, y_value - 20), 2)
            pygame.draw.line(self.screen, color, (x_value, y_value - 8), (x_value + 10 + sway, y_value - 20), 2)
            return

        if player.state == 6:
            slump = display_frame
            head_center = (x_value - 3, y_value - 10 + slump)
            pygame.draw.circle(self.screen, color, head_center, 6, 2)
            pygame.draw.line(self.screen, color, (x_value, y_value - 4), (x_value + 6, y_value + 10), 2)
            pygame.draw.line(self.screen, color, (x_value + 6, y_value + 10), (x_value - 4, y_value + 18), 2)
            pygame.draw.line(self.screen, color, (x_value + 6, y_value + 10), (x_value + 14, y_value + 18), 2)
            pygame.draw.line(self.screen, color, (x_value + 2, y_value + 1), (x_value - 8, y_value + 6), 2)
            pygame.draw.line(self.screen, color, (x_value + 5, y_value + 4), (x_value + 14, y_value + 8), 2)
            return

        if player.state in (3, 4):
            direction = player.diving_direction
            if direction == 0:
                direction = 1
            head_x = x_value + direction * 12
            head_y = y_value - 6
            pygame.draw.circle(self.screen, color, (head_x, head_y), 6, 2)
            pygame.draw.line(
                self.screen,
                color,
                (x_value - direction * 14, y_value + 2),
                (x_value + direction * 8, y_value - 2),
                2,
            )
            pygame.draw.line(
                self.screen,
                color,
                (x_value - direction * 6, y_value - 8),
                (x_value + direction * 2, y_value + 6),
                2,
            )
            pygame.draw.line(
                self.screen,
                color,
                (x_value - direction * 10, y_value + 8),
                (x_value + direction * 2, y_value + 2),
                2,
            )
            return

        head_center = (x_value, y_value - 20)
        hip_y = y_value + 6
        arm_y = y_value - 4
        foot_y = y_value + 20

        pygame.draw.circle(self.screen, color, head_center, 7, 2)
        pygame.draw.line(self.screen, color, (x_value, y_value - 13), (x_value, hip_y), 2)
        pygame.draw.line(self.screen, color, (x_value - 10, arm_y), (x_value + 10, arm_y), 2)
        pygame.draw.line(self.screen, color, (x_value, hip_y), (x_value - 9, foot_y), 2)
        pygame.draw.line(self.screen, color, (x_value, hip_y), (x_value + 9, foot_y), 2)

    def _draw_bnw_spike_flames(self):
        ball_x = float(self.ball.x)
        ball_y = float(self.ball.y)
        x_velocity = float(getattr(self.ball, "x_velocity", 0.0))
        y_velocity = float(getattr(self.ball, "y_velocity", 0.0))
        velocity_norm = math.hypot(x_velocity, y_velocity)
        if velocity_norm < 1.0:
            trail_dx = -1.0
            trail_dy = 0.0
        else:
            trail_dx = -x_velocity / velocity_norm
            trail_dy = -y_velocity / velocity_norm

        perp_dx = -trail_dy
        perp_dy = trail_dx

        for flame_index, flame_length in enumerate((20, 15, 11)):
            base_distance = 10 + flame_index * 8
            base_x = ball_x + trail_dx * base_distance
            base_y = ball_y + trail_dy * base_distance
            spread = 6 - flame_index
            flame_points = [
                (base_x + trail_dx * flame_length, base_y + trail_dy * flame_length),
                (base_x + perp_dx * spread, base_y + perp_dy * spread),
                (base_x - perp_dx * spread, base_y - perp_dy * spread),
            ]
            pygame.draw.polygon(self.screen, (210, 210, 210), flame_points)
            pygame.draw.polygon(self.screen, (0, 0, 0), flame_points, 1)

    def _draw_bnw_ball(self):
        ball_center = (int(self.ball.x), int(self.ball.y))
        ball_radius = 13
        ball_rect = pygame.Rect(
            ball_center[0] - ball_radius,
            ball_center[1] - ball_radius,
            ball_radius * 2,
            ball_radius * 2,
        )

        if self.ball.is_power_hit:
            self._draw_bnw_spike_flames()

        pygame.draw.circle(self.screen, (245, 245, 245), ball_center, ball_radius)
        pygame.draw.circle(self.screen, (0, 0, 0), ball_center, ball_radius, 2)
        pygame.draw.arc(self.screen, (0, 0, 0), ball_rect, 0.4, 2.3, 1)
        pygame.draw.arc(self.screen, (0, 0, 0), ball_rect.inflate(-8, -6), 2.7, 5.2, 1)
        pygame.draw.arc(self.screen, (0, 0, 0), ball_rect.inflate(-5, -10), 5.3, 7.3, 1)

        if self.ball.is_power_hit:
            pygame.draw.circle(self.screen, (0, 0, 0), ball_center, 18, 1)

        if self.ball.punch_effect_radius > 0:
            self.ball.punch_effect_radius -= 2
            effect_radius = int(self.ball.punch_effect_radius)
            if effect_radius > 0:
                pygame.draw.circle(
                    self.screen,
                    (0, 0, 0),
                    (int(self.ball.punch_effect_x), int(self.ball.punch_effect_y)),
                    effect_radius,
                    1,
                )

    def _draw_overlay(self):
        if self.overlay_dirty:
            self.overlay_surface.fill((0, 0, 0, 0))
            primary_color = (0, 0, 0) if self.bnw_mode else (255, 255, 255)
            player1_color = (0, 0, 0) if self.bnw_mode else (255, 220, 120)
            player2_color = (0, 0, 0) if self.bnw_mode else (160, 235, 255)
            status_color = (0, 0, 0) if self.bnw_mode else (255, 240, 60)
            self._draw_text(
                self.overlay_surface,
                self.score_text,
                self.large_font,
                24,
                primary_color,
            )
            self._draw_player_label(
                self.overlay_surface,
                "player1",
                GROUND_HALF_WIDTH // 2,
                player1_color,
            )
            self._draw_player_label(
                self.overlay_surface,
                "player2",
                GROUND_HALF_WIDTH + GROUND_HALF_WIDTH // 2,
                player2_color,
            )
            quit_text = self._quit_key_text()
            control_hint = self._control_hint_text()
            if self._is_game_end_status():
                banner_progress = min(self.game_end_animation_frame, 50)
                banner_scale = 1.0 + (4.0 * (50 - banner_progress) / 50.0)
                self._draw_scaled_text(
                    self.overlay_surface,
                    "GAME END",
                    self.large_font,
                    GROUND_WIDTH // 2,
                    76,
                    status_color,
                    banner_scale,
                )
                self._draw_text(
                    self.overlay_surface,
                    self.status_text,
                    self.medium_font,
                    120,
                    primary_color,
                )
                self._draw_text(
                    self.overlay_surface,
                    self.status_command_text,
                    self.small_font,
                    144,
                    primary_color,
                )
            elif self.status_text:
                status_font = self.large_font
                if self.status_text == "PRESS S TO START":
                    status_font = self.medium_font
                self._draw_text(
                    self.overlay_surface,
                    self.status_text,
                    status_font,
                    52,
                    status_color,
                )
                self._draw_text(
                    self.overlay_surface,
                    self.status_command_text,
                    self.small_font,
                    78,
                    primary_color,
                )
            else:
                self._draw_text(
                    self.overlay_surface,
                    quit_text,
                    self.small_font,
                    24 + 28,
                    primary_color,
                )
            if control_hint:
                self._draw_text(
                    self.overlay_surface,
                    control_hint,
                    self.info_font,
                    GROUND_HEIGHT - 14,
                    primary_color,
                )
            self.overlay_dirty = False

        self.screen.blit(self.overlay_surface, (0, 0))

    def _update_audio_state(self):
        if self.audio is None:
            return

        self.audio.play_bgm()

        for player_key, player in (
            ("player1", self.player1),
            ("player2", self.player2),
        ):
            previous_state = self.last_player_states[player_key]
            current_state = int(player.state)
            if previous_state == 0 and current_state in (1, 2, 3):
                self.audio.play_effect("jump")
            self.last_player_states[player_key] = current_state

        power_hit_active = bool(
            self.ball.is_power_hit
            and self.ball.punch_effect_radius > 0
            and int(self.ball.punch_effect_y) < 260
        )
        if power_hit_active and not self.last_power_hit_active:
            self.audio.play_effect("power_hit")
        self.last_power_hit_active = power_hit_active

        if (
            self.ball.punch_effect_radius > 0
            and int(self.ball.punch_effect_y) >= 260
        ):
            ground_signature = (
                int(self.ball.punch_effect_x),
                int(self.ball.punch_effect_y),
                self.current_score,
            )
            if ground_signature != self.last_ground_effect_signature:
                self.audio.play_effect("ball_ground")
                self.last_ground_effect_signature = ground_signature
        else:
            self.last_ground_effect_signature = None

    def _process_events(self, allow_restart=False, allow_start=False):
        if self.headless:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed_requested = True
                self.pending_command = "quit"
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(
                    event.key,
                    allow_restart=allow_restart,
                    allow_start=allow_start,
                )
            elif event.type == pygame.KEYUP:
                self.pressed_keys.discard(event.key)

    def get_human_input(self, player_id, player=None):
        self._process_events()

        user_input = UserInput()
        if player_id == 0:
            if pygame.K_h in self.pressed_keys and pygame.K_k not in self.pressed_keys:
                user_input.x_direction = -1
            elif pygame.K_k in self.pressed_keys and pygame.K_h not in self.pressed_keys:
                user_input.x_direction = 1

            if pygame.K_u in self.pressed_keys and pygame.K_j not in self.pressed_keys:
                user_input.y_direction = -1
            elif pygame.K_j in self.pressed_keys and pygame.K_u not in self.pressed_keys:
                user_input.y_direction = 1

            if pygame.K_z in self.pressed_keys:
                user_input.power_hit = 1
        else:
            if pygame.K_LEFT in self.pressed_keys and pygame.K_RIGHT not in self.pressed_keys:
                user_input.x_direction = -1
            elif pygame.K_RIGHT in self.pressed_keys and pygame.K_LEFT not in self.pressed_keys:
                user_input.x_direction = 1

            if pygame.K_UP in self.pressed_keys and pygame.K_DOWN not in self.pressed_keys:
                user_input.y_direction = -1
            elif pygame.K_DOWN in self.pressed_keys and pygame.K_UP not in self.pressed_keys:
                user_input.y_direction = 1

            if pygame.K_RETURN in self.pressed_keys or pygame.K_KP_ENTER in self.pressed_keys:
                user_input.power_hit = 1

        action_name = "idle"
        if player is not None:
            from .actions import describe_user_input

            action_name = describe_user_input(player_id, user_input, player=player)
        return user_input, action_name

    def update(self):
        if self._is_game_end_status():
            self.game_end_animation_frame += 1
            self.overlay_dirty = True

        self._update_audio_state()

        if self.bnw_mode:
            self.screen.blit(self.background, (0, 0))
            self._draw_bnw_player(self.player1)
            self._draw_bnw_player(self.player2)
            self._draw_bnw_ball()
            self._draw_overlay()
            return

        ball_image = self.ball_images[self.ball.rotation]
        player1_frame = self._get_display_frame_number(self.player1)
        player2_frame = self._get_display_frame_number(self.player2)
        player1_image = self.player_images[get_image_index(self.player1.state, player1_frame)]
        player2_image = self.player_images[get_image_index(self.player2.state, player2_frame)]

        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.shadow_image, (self.player1.x - 16, 269))
        self.screen.blit(self.shadow_image, (self.player2.x - 16, 269))
        self.screen.blit(self.shadow_image, (self.ball.x - 16, 269))

        if (self.player1.state == 3 or self.player1.state == 4) and self.player1.diving_direction == -1:
            player1_image = self.player_images_flipped[
                get_image_index(self.player1.state, player1_frame)
            ]

        if not ((self.player2.state == 3 or self.player2.state == 4) and self.player2.diving_direction == 1):
            player2_image = self.player_images_flipped[
                get_image_index(self.player2.state, player2_frame)
            ]

        self.screen.blit(player1_image, (self.player1.x - 32, self.player1.y - 32))
        self.screen.blit(player2_image, (self.player2.x - 32, self.player2.y - 32))

        if self.ball.is_power_hit:
            self.screen.blit(
                self.ball_trail_image,
                (self.ball.previous_previous_x - 20, self.ball.previous_previous_y - 20),
            )
            self.screen.blit(
                self.ball_hyper_image,
                (self.ball.previous_x - 20, self.ball.previous_y - 20),
            )

        self.screen.blit(ball_image, (self.ball.x - 20, self.ball.y - 20))

        if self.ball.punch_effect_radius > 0:
            self.ball.punch_effect_radius -= 2
            effect_size = self.ball.punch_effect_radius
            if effect_size > 0:
                if effect_size not in self.ball_punch_cache:
                    self.ball_punch_cache[effect_size] = pygame.transform.scale(
                        self.ball_punch_image, (effect_size * 2, effect_size * 2)
                    )
                ball_punch_image = self.ball_punch_cache[effect_size]
                self.screen.blit(
                    ball_punch_image,
                    (self.ball.punch_effect_x - effect_size, self.ball.punch_effect_y - effect_size),
                )

        self._draw_overlay()

    def render(self):
        if not self.headless:
            self._process_events()
            self._present_display()
            self.clock.tick(30)
            pygame.display.update()

    def consume_command(self):
        command = self.pending_command
        self.pending_command = None
        return command

    def wait_for_command(self, allow_restart=False, allow_start=False):
        if self.headless:
            return None

        while True:
            self._process_events(allow_restart=allow_restart, allow_start=allow_start)
            self.update()
            self._present_display()
            self.clock.tick(30)
            pygame.display.update()
            command = self.consume_command()
            if command is not None:
                return command

    def get_screen_rgb_array(self):
        return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def close(self):
        if self.audio is not None:
            self.audio.close()
        pygame.quit()
        pygame.display.quit()
