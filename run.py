import csv
import os
import sys
import time
import ctypes  # Added for high-res clock period adjustments
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
    DrivingVideoPolicy,
    ModelConfig,
    TemporalState,
)

pdi.FAILSAFE = True
autopilot = False
cam = dxcam.create(output_color="BGR")
button_states: Dict[str, bool] = {}
size = pdi.size()

# --- FIX 1: Native High-Resolution Windows Timer Support ---
def enable_high_resolution_timer():
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        print("Sharp 1ms Windows timer system resolution enabled.")
    except Exception as exc:
        print(f"Warning: failed to enable high-resolution scheduler: {exc}")

def disable_high_resolution_timer():
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass


@dataclass
class RuntimeConfig(ModelConfig):
    ckpt_dir: str = "./checkpoints_rt"
    ckpt_path: Optional[str] = None
    pos_weight_power: float = 0.5
    pos_weight_clamp: float = 8.0
    button_threshold_from_pos_weight: bool = False
    button_threshold_min: float = 0.5  # Lowered to help sensitivity sliders
    button_threshold_max: float = 0.9
    use_checkpoint_button_thresholds: bool = True

    decision_interval: float = 1.0 / 20.0
    # Actions execute the single configured future-frame prediction.
    command_horizon: int = 1
    print_every: int = 2
    print_prob_decimals: int = 3

    mouse_buttons_enabled: bool = False
    gru_memory_frames: int = 80
    # data.py records at 512x512 INTER_LINEAR before DALI linear-resizes to
    # model_size; runtime capture must mirror that two-stage path.
    record_frame_size: int = 512

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decision_interval = float(self.decision_interval)
        self.command_horizon = max(1, int(self.command_horizon))
        self.pos_weight_power = max(0.0, float(self.pos_weight_power))
        self.pos_weight_clamp = max(1.0, float(self.pos_weight_clamp))
        self.button_threshold_from_pos_weight = bool(self.button_threshold_from_pos_weight)
        self.use_checkpoint_button_thresholds = bool(self.use_checkpoint_button_thresholds)
        self.button_threshold_min = float(np.clip(float(self.button_threshold_min), 0.0, 1.0))
        self.button_threshold_max = float(np.clip(float(self.button_threshold_max), self.button_threshold_min, 1.0))
        self.mouse_buttons_enabled = bool(self.mouse_buttons_enabled)
        self.gru_memory_frames = max(1, int(self.gru_memory_frames))
        self.record_frame_size = max(1, int(self.record_frame_size))


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
    if key_str in ("1", "+"):
        autopilot = not autopilot
        print(f"Autopilot: {autopilot}")
    elif key_str in ("2", "_"):
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


def draw_cursor_on_image(img_bgr, cursor_x, cursor_y, hcursor):
    try:
        cur_bgra, hot_x, hot_y = _cursor_bgra_from_hicon(hcursor)
        if cur_bgra is None:
            mx = int(cursor_x * img_bgr.shape[1] / size[0])
            my = int(cursor_y * img_bgr.shape[0] / size[1])
            return cv2.circle(img_bgr, (mx, my), 6, (0, 255, 0), 2)

        scale_x = img_bgr.shape[1] / float(size[0])
        scale_y = img_bgr.shape[0] / float(size[1])
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
    flags, hcursor, _, _ = get_cursor_info()

    record_size = int(cfg.record_frame_size)
    img = cv2.resize(img, (record_size, record_size), interpolation=cv2.INTER_LINEAR)
    if flags == 1 and hcursor:
        img = draw_cursor_on_image(img, mx, my, hcursor)
    if record_size != int(cfg.model_size):
        img = cv2.resize(img, (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def _coerce_config_types(cfg: RuntimeConfig) -> RuntimeConfig:
    cfg.seq_len = int(cfg.seq_len)
    cfg.train_seq_stride = int(cfg.train_seq_stride)
    cfg.val_seq_stride = int(cfg.val_seq_stride)
    cfg.model_size = int(cfg.model_size)
    cfg.prediction_horizon = 1
    offsets = getattr(cfg, "prediction_horizon_offsets", None)
    if offsets is None:
        cfg.prediction_horizon_offsets = (1,)
    else:
        cfg.prediction_horizon_offsets = tuple(int(offset) for offset in offsets)
        if not cfg.prediction_horizon_offsets:
            cfg.prediction_horizon_offsets = (1,)
        if any(offset <= 0 for offset in cfg.prediction_horizon_offsets):
            raise ValueError(f"prediction_horizon_offsets must be positive, got {cfg.prediction_horizon_offsets}.")
        if len(cfg.prediction_horizon_offsets) != 1:
            raise ValueError(f"Single-horizon checkpoints must provide exactly one prediction offset, got {cfg.prediction_horizon_offsets}.")
    cfg.command_horizon = 1
    cfg.d_model = int(cfg.d_model)
    cfg.mouse_buttons_enabled = bool(cfg.mouse_buttons_enabled)
    cfg.gru_memory_frames = max(1, int(getattr(cfg, "gru_memory_frames", 80)))
    cfg.num_bin = len(cfg.key_names) + len(cfg.mouse_button_names)
    cfg.button_state_threshold = float(np.clip(float(cfg.button_state_threshold), 0.0, 1.0))
    cfg.use_checkpoint_button_thresholds = bool(getattr(cfg, "use_checkpoint_button_thresholds", False))
    
    if cfg.button_state_thresholds is None:
        cfg.button_state_thresholds = tuple(float(cfg.button_state_threshold) for _ in range(cfg.num_bin))
    else:
        thresholds = tuple(float(np.clip(float(x), 0.0, 1.0)) for x in cfg.button_state_thresholds)
        if len(thresholds) != cfg.num_bin:
            thresholds = tuple(float(cfg.button_state_threshold) for _ in range(cfg.num_bin))
        # FIX 2: Removed hardcoded (0.5, 0.5, 0.5, 0.3) line to support custom decision configurations
        cfg.button_state_thresholds = thresholds
    cfg.record_frame_size = max(1, int(getattr(cfg, "record_frame_size", 512)))
    return cfg


def _thresholds_from_counts(pos: np.ndarray, total: int, cfg: RuntimeConfig) -> Tuple[float, ...]:
    if total <= 0 or not bool(cfg.button_threshold_from_pos_weight):
        return tuple(float(cfg.button_state_threshold) for _ in range(cfg.num_bin))
    neg = np.maximum(float(total) - pos.astype(np.float64), 0.0)
    weights = np.power(neg / np.maximum(pos.astype(np.float64), 1.0), float(cfg.pos_weight_power))
    weights = np.clip(weights, 1.0, float(cfg.pos_weight_clamp))
    thresholds = weights / (weights + 1.0)
    thresholds = np.clip(thresholds, float(cfg.button_threshold_min), float(cfg.button_threshold_max))
    return tuple(float(x) for x in thresholds)


def _derive_button_thresholds_from_csv(cfg: RuntimeConfig) -> Optional[Tuple[float, ...]]:
    roots = []
    if cfg.data_root:
        roots.append(Path(str(cfg.data_root)))
    roots.append(Path(__file__).resolve().parent / "data" / str(cfg.selected_game))

    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    pos = np.zeros(len(names), dtype=np.float64)
    total = 0
    seen_paths = set()
    for root in roots:
        if not root.exists():
            continue
        for csv_path in root.glob(f"*{cfg.csv_ext}"):
            resolved = str(csv_path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or any(name not in reader.fieldnames for name in names):
                    continue
                for row in reader:
                    total += 1
                    for idx, name in enumerate(names):
                        try:
                            pos[idx] += 1.0 if float(row.get(name, 0.0)) > 0.5 else 0.0
                        except (TypeError, ValueError):
                            pass
    if total <= 0:
        return None
    return _thresholds_from_counts(pos, total, cfg)


def _apply_checkpoint_config(cfg: RuntimeConfig, overrides: Dict) -> RuntimeConfig:
    use_checkpoint_thresholds = bool(getattr(cfg, "use_checkpoint_button_thresholds", False))
    checkpoint_has_thresholds = use_checkpoint_thresholds and overrides.get("button_state_thresholds") is not None
    threshold_keys = {
        "button_state_threshold",
        "button_state_thresholds",
        "button_threshold_from_pos_weight",
        "button_threshold_min",
        "button_threshold_max",
    }
    for key, value in overrides.items():
        if key in threshold_keys and not use_checkpoint_thresholds:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg = _coerce_config_types(cfg)
    if not checkpoint_has_thresholds:
        derived = _derive_button_thresholds_from_csv(cfg)
        if derived is not None:
            cfg.button_state_thresholds = derived
    return cfg


def _checkpoint_path(cfg: RuntimeConfig) -> str:
    if cfg.ckpt_path is not None:
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
        return cfg.ckpt_path

    best_path = os.path.join(cfg.ckpt_dir, "model_best.pt")
    if os.path.exists(best_path):
        return best_path
    latest_path = os.path.join(cfg.ckpt_dir, "model_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    raise FileNotFoundError(f"Expected checkpoint at {best_path!r} or {latest_path!r}.")


def load_checkpoint(
    cfg: RuntimeConfig,
    device: torch.device,
) -> Tuple[RuntimeConfig, Dict]:
    ckpt_path = _checkpoint_path(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict) or not isinstance(state.get("model_state"), dict):
        raise RuntimeError("Current checkpoints must contain dict keys: config and model_state.")

    checkpoint_config = dict(state["config"])
    model_state = state["model_state"]

    cfg = _apply_checkpoint_config(cfg, checkpoint_config)
    cfg = _coerce_config_types(cfg)
    cfg.ckpt_path = ckpt_path
    print(f"Checkpoint: epoch={state.get('epoch')} step={state.get('global_step')} best={state.get('best_score')}")
    return cfg, model_state


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
        button_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.step += 1
        buttons = button_state.detach().cpu().float().numpy()
        applied_buttons = button_state.detach().clone()

        for idx, key_name in enumerate(self.cfg.key_names):
            self._set_key_state(key_name, bool(buttons[idx] >= 0.5))

        for idx, button_name in enumerate(self.cfg.mouse_button_names):
            state_idx = len(self.cfg.key_names) + idx
            should_press = bool(buttons[state_idx] >= 0.5) and bool(self.cfg.mouse_buttons_enabled)
            self._set_mouse_button_state(button_name, should_press)
            applied_buttons[state_idx] = 1.0 if should_press else 0.0

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
            print(f"FPS={fps_str}")

            if button_probs is not None:
                all_names = self.cfg.key_names + self.cfg.mouse_button_names
                print(self._format_prob_line("state", all_names, button_probs.detach().cpu().float().numpy()))

        return applied_buttons


def main() -> None:
    global RUNTIME_CFG

    enable_high_resolution_timer()  # Turn on 1ms Windows precision boundaries
    cfg = RUNTIME_CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    cfg, model_state = load_checkpoint(cfg, device)

    # Custom sensitivity optimization overrides (Tweak these variables to adjust turning rules!)
    # w, a, s, d
    # cfg.button_state_thresholds = (0.50, 0.5, 0.4, 0.5)
    RUNTIME_CFG = cfg

    model = DrivingVideoPolicy(cfg).to(device)
    model.load_state_dict(model_state)
    print(
        "Model:",
        f"size={cfg.model_size}",
        f"prediction_offset=+{int(cfg.prediction_horizon_offsets[0])}",
        f"d_model={cfg.d_model}",
        "temporal=convgru",
        "input=masked_rgb" + ("+last_action" if cfg.last_action_conditioning else ""),
    )
    print(
        "Button thresholds:",
        f"use_checkpoint={cfg.use_checkpoint_button_thresholds}",
        " ".join(
            f"{name}={threshold:.3f}"
            for name, threshold in zip(cfg.key_names + cfg.mouse_button_names, cfg.button_state_thresholds)
        ),
    )
    print(
        "Mouse buttons:",
        f"buttons_enabled={cfg.mouse_buttons_enabled}",
    )

    inference_dtype = torch.float32

    model.eval()
    controller = ActionController(cfg)

    thresholds = torch.tensor(
        list(cfg.button_state_thresholds),
        device=device,
        dtype=inference_dtype,
    )

    temporal_state = TemporalState()
    prev_action = torch.zeros((1, cfg.num_bin), device=device, dtype=inference_dtype)
    was_autopilot = False

    print("=" * 60)
    print("Running. Press '1' to toggle autopilot, '2' to disable, Ctrl+C to quit.")
    print("=" * 60)

    try:
        while True:
            loop_start = time.perf_counter()

            if autopilot and not was_autopilot:
                temporal_state = TemporalState()
                prev_action = torch.zeros((1, cfg.num_bin), device=device, dtype=inference_dtype)
                print("Autopilot ENABLED - temporal state reset")

            if (not autopilot) and was_autopilot:
                temporal_state = TemporalState()
                prev_action = torch.zeros((1, cfg.num_bin), device=device, dtype=inference_dtype)
                release_all()
                print("Autopilot DISABLED - temporal state reset")

            was_autopilot = autopilot

            if autopilot:
                frame_cpu = capture_frame(cfg)
                if frame_cpu is not None:
                    frame = frame_cpu.to(device, non_blocking=True)
                    if frame.dtype != inference_dtype:
                        frame = frame.to(dtype=inference_dtype)
                    with torch.inference_mode():
                        frame_batch = frame.unsqueeze(0)
                        output, temporal_state = model.forward_step(
                            frame_batch,
                            temporal_state,
                            prev_action=prev_action,
                        )
                        button_logits = output.button_logits
                    button_probs = torch.sigmoid(button_logits[0])
                    predicted_buttons = (button_probs >= thresholds).to(dtype=button_logits.dtype)

                    applied_buttons = controller.apply(
                        predicted_buttons,
                        button_probs=button_probs,
                    )
                    prev_action = applied_buttons.detach().reshape(1, cfg.num_bin).to(device=device, dtype=inference_dtype)

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
        disable_high_resolution_timer()
        release_all()
        print("Released all inputs. Exiting.")


if __name__ == "__main__":
    main()
