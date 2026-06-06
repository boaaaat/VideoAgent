import sys
import time
import ctypes  # Added for high-res clock period adjustments
from dataclasses import dataclass, fields
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
    ARCHITECTURE_VERSION,
    MODEL_FAMILY,
    DrivingVideoPolicy,
    ModelConfig,
    TemporalState,
    validate_policy_checkpoint,
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
    ckpt_dir: str = "./checkpoints_multihorizon"
    ckpt_path: Optional[str] = None

    decision_interval: float = 1.0 / 20.0
    command_horizon: int = 1
    print_every: int = 2
    print_prob_decimals: int = 3

    mouse_buttons_enabled: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decision_interval = float(self.decision_interval)
        self.command_horizon = max(1, int(self.command_horizon))
        self.mouse_buttons_enabled = bool(self.mouse_buttons_enabled)


RUNTIME_CFG = RuntimeConfig()
MOUSE_NAME_MAP = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
}
SCROLL_ACTIONS = {
    "scroll_up": 1,
    "scroll_down": -1,
}


def release_all() -> None:
    for key_name in RUNTIME_CFG.key_names:
        mapped = key_name.split(".", 1)[1] if key_name.startswith("Key.") else key_name
        pdi.keyUp(mapped, _pause=False)
        button_states[key_name] = False
    for button_name in RUNTIME_CFG.mouse_button_names:
        mapped = MOUSE_NAME_MAP.get(button_name)
        if mapped is not None:
            pdi.mouseUp(button=mapped, _pause=False)
        button_states[button_name] = False


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
    _, hcursor, _, _ = get_cursor_info()

    img = cv2.resize(img, (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_AREA)
    img = draw_cursor_on_image(img, mx, my, hcursor)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def _coerce_config_types(cfg: RuntimeConfig) -> RuntimeConfig:
    cfg.command_horizon = max(1, min(int(cfg.command_horizon), int(cfg.prediction_horizon)))
    cfg.decision_interval = max(0.0, float(cfg.decision_interval))
    cfg.mouse_buttons_enabled = bool(cfg.mouse_buttons_enabled)
    return cfg


def _apply_checkpoint_config(cfg: RuntimeConfig, overrides: Dict) -> RuntimeConfig:
    model_fields = {field.name for field in fields(ModelConfig)}
    runtime_fields = {field.name for field in fields(RuntimeConfig)} - model_fields
    runtime_values = {key: getattr(cfg, key) for key in runtime_fields}
    cfg_kwargs = {key: value for key, value in overrides.items() if key in model_fields}
    cfg_kwargs.update(runtime_values)
    return _coerce_config_types(RuntimeConfig(**cfg_kwargs))


def _existing_path(path: str) -> Optional[Path]:
    raw_path = Path(path).expanduser()
    base_dir = Path(__file__).resolve().parent
    candidates = [raw_path] if raw_path.is_absolute() else [Path.cwd() / raw_path, base_dir / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _checkpoint_path(cfg: RuntimeConfig) -> str:
    if cfg.ckpt_path is not None:
        resolved = _existing_path(cfg.ckpt_path)
        if resolved is None:
            raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
        return str(resolved)

    ckpt_dir = _existing_path(cfg.ckpt_dir)
    if ckpt_dir is None or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {cfg.ckpt_dir}")

    candidates = [
        ckpt_dir / "model_best.pt",
        ckpt_dir / "model_latest.pt",
    ]
    epoch_candidates = sorted(
        ckpt_dir.glob("model_epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    candidates.extend(reversed(epoch_candidates))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Expected checkpoint under {str(ckpt_dir)!r}.")


def load_checkpoint(
    cfg: RuntimeConfig,
) -> Tuple[RuntimeConfig, Dict]:
    ckpt_path = _checkpoint_path(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict) or not isinstance(state.get("model_state"), dict):
        raise RuntimeError("Current checkpoints must contain dict keys: config and model_state.")
    if state.get("model_family") != MODEL_FAMILY or int(state.get("architecture_version", -1)) != ARCHITECTURE_VERSION:
        raise RuntimeError(f"Checkpoint must be {MODEL_FAMILY} v{ARCHITECTURE_VERSION}; legacy checkpoints are unsupported.")

    checkpoint_config = dict(state["config"])
    cfg = _apply_checkpoint_config(cfg, checkpoint_config)
    validate_policy_checkpoint(state, cfg)
    cfg.ckpt_path = ckpt_path
    print(f"Checkpoint: epoch={state.get('epoch')} step={state.get('global_step')} best={state.get('best_score')}")
    return cfg, state["ema_model_state"]


class ActionController:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.step = 0
        self.last_print_t: Optional[float] = None
        self.last_print_step = 0
        self.key_name_map = {name: (name.split(".", 1)[1] if name.startswith("Key.") else name) for name in cfg.key_names}
        self.mouse_name_map = dict(MOUSE_NAME_MAP)
        unsupported_buttons = [
            name
            for name in cfg.mouse_button_names
            if name not in self.mouse_name_map and name not in SCROLL_ACTIONS
        ]
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
        if button_name in SCROLL_ACTIONS:
            if should_press and not button_states[button_name]:
                pdi.scroll(SCROLL_ACTIONS[button_name], _pause=False)
            button_states[button_name] = should_press
            return
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
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    cfg, model_state = load_checkpoint(cfg)
    RUNTIME_CFG = cfg

    model = DrivingVideoPolicy(cfg)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with {MODEL_FAMILY} v{ARCHITECTURE_VERSION}."
        ) from exc
    del model_state
    model = model.to(device).to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        "Model:",
        f"size={cfg.model_size}",
        f"horizon={cfg.prediction_horizon}",
        f"command_horizon={cfg.command_horizon}",
        f"command_frame_offset=+{int(cfg.prediction_horizon_offsets[cfg.command_horizon - 1])}",
        f"action_label_offset={int(cfg.action_label_offset):+d}",
        f"effective_target_offset=+{int(cfg.prediction_horizon_offsets[cfg.command_horizon - 1]) + int(cfg.action_label_offset)}",
        f"d_model={cfg.d_model}",
        f"architecture={MODEL_FAMILY}_v{ARCHITECTURE_VERSION}",
        f"spatial_gru={cfg.spatial_channels}@{cfg.model_size // 4}x{cfg.model_size // 4}",
        f"compressor={cfg.compressor_channels}@{cfg.spatial_pool_size}x{cfg.spatial_pool_size}",
        "input=masked_rgb+frame_difference+learned_action_context",
    )
    command_idx = max(0, min(int(cfg.command_horizon) - 1, int(cfg.prediction_horizon) - 1))
    command_thresholds = tuple(float(value) for value in cfg.horizon_button_thresholds[command_idx])
    print(
        f"Button thresholds @+{int(cfg.prediction_horizon_offsets[command_idx]) + int(cfg.action_label_offset)}:",
        " ".join(
            f"{name}={threshold:.3f}"
            for name, threshold in zip(cfg.key_names + cfg.mouse_button_names, command_thresholds)
        ),
    )
    print(
        "Mouse buttons:",
        f"buttons_enabled={cfg.mouse_buttons_enabled}",
    )

    use_autocast = False
    inference_dtype = torch.float32
    if device.type == "cuda":
        if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float16
        use_autocast = True
    print(
        "Inference precision:",
        f"dtype={inference_dtype}",
        f"autocast={use_autocast}",
    )

    model.eval()
    controller = ActionController(cfg)

    temporal_state = TemporalState()
    prev_action = torch.zeros((1, cfg.num_bin), device=device, dtype=inference_dtype)
    last_frame_time: Optional[float] = None
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
                last_frame_time = None
                print("Autopilot ENABLED - temporal state reset")

            if (not autopilot) and was_autopilot:
                temporal_state = TemporalState()
                prev_action = torch.zeros((1, cfg.num_bin), device=device, dtype=inference_dtype)
                last_frame_time = None
                release_all()
                print("Autopilot DISABLED - temporal state reset")

            was_autopilot = autopilot

            if autopilot:
                frame_cpu = capture_frame(cfg)
                if frame_cpu is not None:
                    frame_time = time.perf_counter()
                    frame_dt = float(cfg.prediction_dt) if last_frame_time is None else frame_time - last_frame_time
                    last_frame_time = frame_time
                    frame_dt = min(max(frame_dt, 1.0 / 240.0), 0.5)
                    frame = frame_cpu.to(device, non_blocking=True)
                    if frame.dtype != inference_dtype:
                        frame = frame.to(dtype=inference_dtype)
                    with torch.inference_mode():
                        frame_batch = frame.unsqueeze(0)
                        dt = torch.tensor([frame_dt], device=device, dtype=frame_batch.dtype)
                        with torch.amp.autocast(
                            device_type=device.type,
                            dtype=inference_dtype,
                            enabled=use_autocast,
                        ):
                            output, temporal_state = model.forward_step(
                                frame_batch,
                                dt,
                                temporal_state,
                                prev_action=prev_action,
                            )
                        button_logits = output.horizon_button_logits
                    button_probs = torch.sigmoid(button_logits[0, command_idx])
                    thresholds = torch.tensor(
                        list(command_thresholds),
                        device=button_probs.device,
                        dtype=button_probs.dtype,
                    )
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
