import os
from pathlib import Path

import pygame


ASSETS_PATH = Path(__file__).with_name("assets")

SOUND_CANDIDATES = {
    "bgm": ("bgm.mp3", "bgm.ogg", "bgm.wav"),
    "jump": ("jump.wav", "WAVE142_1.wav"),
    "power_hit": ("power_hit.wav", "WAVE141_1.wav"),
    "ball_ground": ("ball_ground.wav", "WAVE146_1.wav"),
}


class ViewerAudio:
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.available = False
        self.effects = {}
        self.bgm_path = None
        self.bgm_started = False
        self.bgm_sound = None
        self.bgm_channel = None
        self.init_error = None
        self.loaded_effect_names = []
        self.audio_driver = None
        self.assets_found = False

        if not self.enabled:
            return

        if not self._init_mixer():
            return

        self.available = True
        self.bgm_path = self._resolve_path(SOUND_CANDIDATES["bgm"])
        if self.bgm_path is not None:
            self.assets_found = True

        for effect_name in ("jump", "power_hit", "ball_ground"):
            sound_path = self._resolve_path(SOUND_CANDIDATES[effect_name])
            if sound_path is None:
                continue
            try:
                loaded_sound = pygame.mixer.Sound(str(sound_path))
                self.effects[effect_name] = loaded_sound
                self.loaded_effect_names.append(effect_name)
                self.assets_found = True
            except Exception as error:
                self.init_error = str(error)
                continue

        if not self.assets_found:
            self.available = False
            try:
                if pygame.mixer.get_init() is not None:
                    pygame.mixer.quit()
            except Exception:
                pass
            return

        self._apply_default_volumes()

    def _init_mixer(self):
        if pygame.mixer.get_init() is not None:
            self.audio_driver = os.environ.get("SDL_AUDIODRIVER")
            return True

        env_audio_driver = os.environ.get("SDL_AUDIODRIVER")
        if env_audio_driver:
            driver_attempts = [env_audio_driver]
        else:
            # On modern Linux desktops, pulse/alsa are usually the real outputs.
            driver_attempts = ["pulseaudio", "alsa", "pipewire", "dsp", None]

        init_attempts = (
            {},
            {"frequency": 44100, "size": -16, "channels": 2, "buffer": 512},
            {"frequency": 22050, "size": -16, "channels": 2, "buffer": 1024},
        )

        for driver_name in driver_attempts:
            if pygame.mixer.get_init() is not None:
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass

            if driver_name is None:
                os.environ.pop("SDL_AUDIODRIVER", None)
            else:
                os.environ["SDL_AUDIODRIVER"] = driver_name

            for init_kwargs in init_attempts:
                try:
                    pygame.mixer.init(**init_kwargs)
                    self.audio_driver = os.environ.get("SDL_AUDIODRIVER")
                    self.init_error = None
                    return True
                except Exception as error:
                    attempted_driver = driver_name if driver_name is not None else "default"
                    self.init_error = f"{attempted_driver}: {error}"

        return False

    def _resolve_path(self, candidates):
        for candidate in candidates:
            candidate_path = ASSETS_PATH / candidate
            if candidate_path.exists():
                return candidate_path
        return None

    def _apply_default_volumes(self):
        if not self.available:
            return

        for effect_name, sound in self.effects.items():
            if effect_name == "ball_ground":
                sound.set_volume(0.45)
            elif effect_name == "jump":
                sound.set_volume(0.35)
            else:
                sound.set_volume(0.5)

        try:
            pygame.mixer.music.set_volume(0.25)
        except Exception:
            pass

    def play_bgm(self):
        if not self.available or self.bgm_path is None:
            return

        try:
            if self.bgm_started and pygame.mixer.music.get_busy():
                return
            pygame.mixer.music.load(str(self.bgm_path))
            pygame.mixer.music.play(-1)
            self.bgm_started = True
            self.bgm_sound = None
            self.bgm_channel = None
            return
        except Exception as error:
            self.init_error = str(error)

        try:
            if self.bgm_sound is None:
                self.bgm_sound = pygame.mixer.Sound(str(self.bgm_path))
                self.bgm_sound.set_volume(0.25)
            if self.bgm_channel is None or not self.bgm_channel.get_busy():
                self.bgm_channel = self.bgm_sound.play(loops=-1)
            self.bgm_started = self.bgm_channel is not None
        except Exception as error:
            self.init_error = str(error)
            self.bgm_started = False

    def stop_bgm(self):
        if not self.available:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            if self.bgm_channel is not None:
                self.bgm_channel.stop()
        except Exception:
            pass
        self.bgm_started = False

    def play_effect(self, effect_name):
        if not self.available:
            return

        sound = self.effects.get(str(effect_name))
        if sound is None:
            return

        try:
            sound.play()
        except Exception as error:
            self.init_error = str(error)
            pass

    def close(self):
        self.stop_bgm()
