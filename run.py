import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dxcam
import numpy as np
import pydirectinput as pdi
import torch
import win32gui
import win32ui
from pynput import keyboard

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (  # noqa: E402
    ActionConditionedVideoPolicy,
    ModelConfig,
    TemporalState,
    get_key_names as MODEL_GET_KEY_NAMES,
    get_mouse_button_names as MODEL_GET_MOUSE_BUTTON_NAMES,
    policy_checkpoint_family_mismatch_reason,
)


pdi.FAILSAFE = True
autopilot = False
cam = dxcam.create(output_color="BGR")
button_states: Dict[str, bool] = {}
size = pdi.size()


@dataclass
class RuntimeConfig(ModelConfig):
    ckpt_dir: str = "C:/Users/Abhil/Desktop/vs_code_stuff/python/ai/checkpoints_rt"
    ckpt_path: Optional[str] = None

    decision_interval: float = 1.0 / 20.0
    print_every: int = 20
    print_prob_decimals: int = 3

    max_mouse_move: float = 1200.0
    mouse_move_duration: float = 0.0
    mouse_move_enabled: bool = True

    capture_width: int = 3840
    capture_height: int = 2160

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decision_interval = float(self.decision_interval)
        self.max_mouse_move = float(self.max_mouse_move)
        self.mouse_move_duration = float(self.mouse_move_duration)
        self.mouse_move_enabled = bool(self.mouse_move_enabled)
        self.capture_width = int(self.capture_width)
        self.capture_height = int(self.capture_height)


RUNTIME_CFG = RuntimeConfig()
MOUSE_NAME_MAP = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
}


def release_all() -> None:
    for key_name in RUNTIME_CFG.key_names:
        mapped = key_name.split(".", 1)[1] if key_name.startswith("Key.") else key_name
        pdi.keyUp(mapped, _pause=False)
    for button_name in RUNTIME_CFG.mouse_button_names:
        mapped = MOUSE_NAME_MAP.get(button_name)
        if mapped is not None:
            pdi.mouseUp(button=mapped, _pause=False)


def on_press(key) -> None:
    global autopilot

    key_str = str(key).replace("'", "")
    if key_str in ("=", "+"):
        autopilot = not autopilot
        print(f"Autopilot: {autopilot}")
    elif key_str in ("-", "_"):
        autopilot = False
        release_all()
        for name in list(button_states.keys()):
            button_states[name] = False
        print("Autopilot disabled. Released all keys and mouse buttons.")


keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()


def get_cursor_info():
    try:
        flags, hcursor, (x, y) = win32gui.GetCursorInfo()
        return flags, hcursor, x, y
    except Exception:
        return None, None, 0, 0


def _cursor_bgra_from_hicon(hcursor):
    f_icon, x_hot, y_hot, hbm_mask, hbm_color = win32gui.GetIconInfo(hcursor)
    del f_icon
    try:
        if not hbm_color:
            return None, x_hot, y_hot
        bmp = win32ui.CreateBitmapFromHandle(hbm_color)
        info = bmp.GetInfo()
        w, h = info["bmWidth"], info["bmHeight"]
        raw = bmp.GetBitmapBits(True)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return arr, x_hot, y_hot
    finally:
        if hbm_mask:
            win32gui.DeleteObject(hbm_mask)
        if hbm_color:
            win32gui.DeleteObject(hbm_color)


def draw_cursor_on_image(img_bgr, cursor_x, cursor_y, hcursor, cfg: RuntimeConfig):
    try:
        cur_bgra, hot_x, hot_y = _cursor_bgra_from_hicon(hcursor)
        if cur_bgra is None:
            mx = int(cursor_x * img_bgr.shape[1] / size[0])
            my = int(cursor_y * img_bgr.shape[0] / size[1])
            return cv2.circle(img_bgr, (mx, my), 6, (0, 255, 0), 2)

        scale_x = cfg.model_size / float(cfg.capture_width)
        scale_y = cfg.model_size / float(cfg.capture_height)
        cur_bgra = cv2.resize(
            cur_bgra,
            (max(1, int(cur_bgra.shape[1] * scale_x)), max(1, int(cur_bgra.shape[0] * scale_y))),
            interpolation=cv2.INTER_NEAREST,
        )
        hot_x = int(hot_x * scale_x)
        hot_y = int(hot_y * scale_y)

        ch, cw = cur_bgra.shape[:2]
        dst_x = int(cursor_x * img_bgr.shape[1] / size[0]) - hot_x
        dst_y = int(cursor_y * img_bgr.shape[0] / size[1]) - hot_y

        x0 = max(dst_x, 0)
        y0 = max(dst_y, 0)
        x1 = min(dst_x + cw, img_bgr.shape[1])
        y1 = min(dst_y + ch, img_bgr.shape[0])
        if x0 >= x1 or y0 >= y1:
            return img_bgr

        cx0 = x0 - dst_x
        cy0 = y0 - dst_y
        cx1 = cx0 + (x1 - x0)
        cy1 = cy0 + (y1 - y0)

        roi = img_bgr[y0:y1, x0:x1]
        cur_roi = cur_bgra[cy0:cy1, cx0:cx1]
        cur_bgr = cur_roi[:, :, :3].astype(np.float32)
        alpha = cur_roi[:, :, 3:4].astype(np.float32) / 255.0
        blended = (cur_bgr * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        roi[:] = blended
        return img_bgr
    except Exception:
        mx = int(cursor_x * img_bgr.shape[1] / size[0])
        my = int(cursor_y * img_bgr.shape[0] / size[1])
        return cv2.circle(img_bgr, (mx, my), 3, (0, 255, 0), -1)


def capture_frame(cfg: RuntimeConfig) -> Optional[torch.Tensor]:
    img = cam.grab()
    if img is None:
        return None

    mx, my = pdi.position()
    _, hcursor, _, _ = get_cursor_info()

    img = cv2.resize(img, (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_AREA)
    img = draw_cursor_on_image(img, mx, my, hcursor, cfg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def _coerce_config_types(cfg: RuntimeConfig) -> RuntimeConfig:
    if cfg.key_names is None:
        cfg.key_names = MODEL_GET_KEY_NAMES(cfg.selected_game)
    if cfg.mouse_button_names is None:
        cfg.mouse_button_names = MODEL_GET_MOUSE_BUTTON_NAMES(cfg.selected_game)
    if cfg.mouse_velocity_scales is None:
        cfg.mouse_velocity_scales = (1.0, 1.0)
    cfg.seq_len = int(cfg.seq_len)
    cfg.train_seq_stride = int(cfg.train_seq_stride)
    cfg.val_seq_stride = int(cfg.val_seq_stride)
    cfg.model_size = int(cfg.model_size)
    cfg.max_context = int(cfg.max_context)
    cfg.local_context_frames = int(cfg.local_context_frames)
    cfg.prediction_dt = float(cfg.prediction_dt)
    cfg.require_pretrained_backbone = bool(cfg.require_pretrained_backbone)
    cfg.num_bin = len(cfg.key_names) + len(cfg.mouse_button_names)
    cfg.prev_action_dim = cfg.num_bin + 2
    return cfg


def _apply_config_overrides(cfg: RuntimeConfig, overrides: Dict) -> RuntimeConfig:
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return _coerce_config_types(cfg)


def _find_checkpoint_path(cfg: RuntimeConfig) -> str:
    if cfg.ckpt_path is not None:
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
        return cfg.ckpt_path

    candidates = [
        cfg.ckpt_dir,
        os.path.join(os.getcwd(), cfg.ckpt_dir),
        os.path.join(os.path.dirname(__file__), cfg.ckpt_dir),
    ]
    for directory in candidates:
        best = os.path.join(directory, "model_best.pt")
        if os.path.exists(best):
            return best
        epoch_ckpts = glob.glob(os.path.join(directory, "model_epoch_*.pt"))
        if epoch_ckpts:
            epoch_ckpts.sort(key=os.path.getmtime)
            return epoch_ckpts[-1]
    raise FileNotFoundError(f"No checkpoints found under {cfg.ckpt_dir!r}")


def load_checkpoint(
    cfg: RuntimeConfig,
    device: torch.device,
) -> Tuple[RuntimeConfig, Dict]:
    ckpt_path = _find_checkpoint_path(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and isinstance(state.get("config"), dict):
        family_reason = policy_checkpoint_family_mismatch_reason(state["config"])
        if family_reason is not None:
            raise RuntimeError(f"Cannot load policy checkpoint {ckpt_path}: {family_reason}")
        cfg = _apply_config_overrides(cfg, state["config"])
    else:
        family_reason = policy_checkpoint_family_mismatch_reason(None)
        if family_reason is not None:
            raise RuntimeError(f"Cannot load policy checkpoint {ckpt_path}: {family_reason}")
    if isinstance(state, dict) and "velocity_scales" in state:
        cfg.mouse_velocity_scales = tuple(float(x) for x in state["velocity_scales"])

    cfg.ckpt_path = ckpt_path
    model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    return cfg, model_state


def initialize_model_lazy_layers(
    model: ActionConditionedVideoPolicy,
    cfg: RuntimeConfig,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_frames = torch.zeros(1, 1, 3, cfg.model_size, cfg.model_size, device=device)
        dummy_prev = torch.zeros(1, 1, cfg.prev_action_dim, device=device)
        dummy_dt = torch.full((1, 1), float(cfg.prediction_dt), device=device)
        _ = model(dummy_frames, prev_actions=dummy_prev, dt=dummy_dt)
    model.train(was_training)


class ActionController:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.step = 0
        self.last_print_t: Optional[float] = None
        self.last_print_step = 0
        self.key_name_map = {name: (name.split(".", 1)[1] if name.startswith("Key.") else name) for name in cfg.key_names}
        self.mouse_name_map = dict(MOUSE_NAME_MAP)
        unsupported_buttons = [name for name in cfg.mouse_button_names if name not in self.mouse_name_map]
        if unsupported_buttons:
            raise ValueError(f"Unsupported runtime mouse buttons: {unsupported_buttons}")
        button_states.clear()
        button_states.update({name: False for name in cfg.key_names + cfg.mouse_button_names})

    def _display_name(self, action_name: str) -> str:
        return action_name.split(".", 1)[1] if action_name.startswith("Key.") else action_name

    def _format_prob_line(self, label: str, names: List[str], probs: np.ndarray) -> str:
        decimals = int(self.cfg.print_prob_decimals)
        flat_probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        parts = [f"{self._display_name(name)}={float(prob):.{decimals}f}" for name, prob in zip(names, flat_probs)]
        return f"{label}: " + " | ".join(parts)

    def _set_key_state(self, key_name: str, should_press: bool) -> None:
        mapped = self.key_name_map[key_name]
        if should_press and not button_states[key_name]:
            pdi.keyDown(mapped, _pause=False)
            button_states[key_name] = True
        elif button_states[key_name] and not should_press:
            pdi.keyUp(mapped, _pause=False)
            button_states[key_name] = False

    def _set_mouse_button_state(self, button_name: str, should_press: bool) -> None:
        mapped = self.mouse_name_map[button_name]
        if should_press and not button_states[button_name]:
            pdi.mouseDown(button=mapped, _pause=False)
            button_states[button_name] = True
        elif button_states[button_name] and not should_press:
            pdi.mouseUp(button=mapped, _pause=False)
            button_states[button_name] = False

    def apply(
        self,
        button_state: torch.Tensor,
        mouse_delta: torch.Tensor,
        button_probs: Optional[torch.Tensor] = None,
        press_probs: Optional[torch.Tensor] = None,
        release_probs: Optional[torch.Tensor] = None,
        mouse_active_prob: Optional[torch.Tensor] = None,
    ) -> None:
        self.step += 1
        buttons = button_state.detach().cpu().float().numpy()
        mouse = mouse_delta.detach().cpu().float().numpy()

        for idx, key_name in enumerate(self.cfg.key_names):
            self._set_key_state(key_name, bool(buttons[idx] >= 0.5))

        for idx, button_name in enumerate(self.cfg.mouse_button_names):
            state_idx = len(self.cfg.key_names) + idx
            self._set_mouse_button_state(button_name, bool(buttons[state_idx] >= 0.5))

        dx = int(np.clip(mouse[0], -self.cfg.max_mouse_move, self.cfg.max_mouse_move))
        dy = int(np.clip(mouse[1], -self.cfg.max_mouse_move, self.cfg.max_mouse_move))
        if self.cfg.mouse_move_enabled and (dx != 0 or dy != 0):
            pdi.moveRel(
                dx,
                dy,
                duration=float(self.cfg.mouse_move_duration),
                relative=True,
                _pause=False,
                disable_mouse_acceleration=True,
            )

        if self.cfg.print_every and self.step % int(self.cfg.print_every) == 0:
            now = time.perf_counter()
            fps = None
            if self.last_print_t is not None:
                elapsed = now - self.last_print_t
                steps = self.step - self.last_print_step
                if elapsed > 0.0 and steps > 0:
                    fps = steps / elapsed
            self.last_print_t = now
            self.last_print_step = self.step
            fps_str = "?" if fps is None else f"{fps:.1f}"
            extra = ""
            if mouse_active_prob is not None:
                extra = f" | mouse_active={float(mouse_active_prob.detach().cpu().item()):.{int(self.cfg.print_prob_decimals)}f}"
            print(f"FPS={fps_str} | mouse=({dx:+4d},{dy:+4d}){extra}")

            if button_probs is not None:
                all_names = self.cfg.key_names + self.cfg.mouse_button_names
                print(self._format_prob_line("state", all_names, button_probs.detach().cpu().float().numpy()))
            if press_probs is not None:
                all_names = self.cfg.key_names + self.cfg.mouse_button_names
                print(self._format_prob_line("press", all_names, press_probs.detach().cpu().float().numpy()))
            if release_probs is not None:
                all_names = self.cfg.key_names + self.cfg.mouse_button_names
                print(self._format_prob_line("release", all_names, release_probs.detach().cpu().float().numpy()))


def main() -> None:
    global RUNTIME_CFG

    cfg = RUNTIME_CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    cfg, model_state = load_checkpoint(cfg, device)
    RUNTIME_CFG = cfg

    model = ActionConditionedVideoPolicy(cfg).to(device)
    initialize_model_lazy_layers(model, cfg, device)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print("Checkpoint/model mismatch:")
        if missing:
            print("  Missing:", missing)
        if unexpected:
            print("  Unexpected:", unexpected)

    inference_dtype = torch.float32
    if device.type == "cuda" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        inference_dtype = torch.bfloat16
    if inference_dtype != torch.float32:
        try:
            model = model.to(dtype=inference_dtype)
        except Exception:
            inference_dtype = torch.float32
            model = model.to(dtype=inference_dtype)

    model.eval()
    controller = ActionController(cfg)

    state: TemporalState = model.init_state(batch_size=1, device=device, dtype=inference_dtype)
    was_autopilot = False
    last_step_time: Optional[float] = None

    print("=" * 60)
    print("Running. Press '=' to toggle autopilot, '-' to disable, Ctrl+C to quit.")
    print("=" * 60)

    try:
        while True:
            loop_start = time.perf_counter()

            if autopilot and not was_autopilot:
                state = model.init_state(batch_size=1, device=device, dtype=inference_dtype)
                last_step_time = None
                print("Autopilot ENABLED - temporal state reset")

            if (not autopilot) and was_autopilot:
                state = model.init_state(batch_size=1, device=device, dtype=inference_dtype)
                last_step_time = None
                release_all()
                print("Autopilot DISABLED - temporal state reset")

            was_autopilot = autopilot

            if autopilot:
                frame_cpu = capture_frame(cfg)
                if frame_cpu is not None:
                    now = time.perf_counter()
                    if last_step_time is None:
                        dt_seconds = float(cfg.decision_interval)
                    else:
                        dt_seconds = now - last_step_time
                    last_step_time = now
                    dt_seconds = float(np.clip(dt_seconds, 1.0 / 60.0, 0.25))

                    frame = frame_cpu.to(device, non_blocking=True)
                    if frame.dtype != inference_dtype:
                        frame = frame.to(dtype=inference_dtype)
                    dt = torch.tensor([dt_seconds], device=device, dtype=frame.dtype)

                    with torch.inference_mode():
                        output, state = model.forward_step(frame.unsqueeze(0), dt=dt, state=state)

                    controller.apply(
                        state.prev_button_state[0],
                        output.mouse_delta[0],
                        button_probs=torch.sigmoid(output.button_logits[0]),
                        press_probs=torch.sigmoid(output.press_logits[0]),
                        release_probs=torch.sigmoid(output.release_logits[0]),
                        mouse_active_prob=torch.sigmoid(output.mouse_active_logits[0, 0]),
                    )

                elapsed = time.perf_counter() - loop_start
                remaining = float(cfg.decision_interval) - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except pdi.FailSafeException:
        print("Fail-safe triggered.")
    finally:
        release_all()
        print("Released all inputs. Exiting.")


if __name__ == "__main__":
    main()
