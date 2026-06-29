import pydirectinput as pdi
import cv2
import time
import threading
import os
import csv
import datetime
import dxcam
import numpy as np
import win32gui
import win32ui
import win32con
import subprocess
import shutil
import ctypes
from collections import deque
from ctypes import wintypes

from action_space import game_data_root, get_key_names, get_mouse_button_names, selected_game

TIMER_RESOLUTION_MS = 1
WHEEL_DELTA = 120
DRAW_CURSOR_OVERLAY = False
LOG_FFMPEG_STDERR = False
timer_resolution_enabled = False


def enable_high_resolution_timer():
    global timer_resolution_enabled
    if timer_resolution_enabled:
        return

    try:
        if ctypes.windll.winmm.timeBeginPeriod(TIMER_RESOLUTION_MS) == 0:
            timer_resolution_enabled = True
            print(f"Enabled {TIMER_RESOLUTION_MS}ms timer resolution.")
    except Exception as exc:
        print(f"Warning: failed to enable {TIMER_RESOLUTION_MS}ms timer resolution: {exc}")


def disable_high_resolution_timer():
    global timer_resolution_enabled
    if not timer_resolution_enabled:
        return

    try:
        ctypes.windll.winmm.timeEndPeriod(TIMER_RESOLUTION_MS)
    except Exception as exc:
        print(f"Warning: failed to restore timer resolution: {exc}")
    finally:
        timer_resolution_enabled = False


def sleep_until(deadline):
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


class RawInputReader:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.scroll_up = 0
        self.scroll_down = 0
        self.lock = threading.Lock()
        
        # --- Windows Structs Definitions ---
        class RAWINPUTDEVICE(ctypes.Structure):
            _fields_ = [("usUsagePage", wintypes.USHORT),
                        ("usUsage", wintypes.USHORT),
                        ("dwFlags", wintypes.DWORD),
                        ("hwndTarget", wintypes.HWND)]

        class RAWMOUSE(ctypes.Structure):
            _fields_ = [("usFlags", wintypes.USHORT),
                        ("ulButtons", wintypes.ULONG),
                        ("ulRawButtons", wintypes.ULONG),
                        ("lLastX", ctypes.c_long),
                        ("lLastY", ctypes.c_long),
                        ("ulExtraInformation", wintypes.ULONG)]

        class RAWINPUTHEADER(ctypes.Structure):
            _fields_ = [("dwType", wintypes.DWORD),
                        ("dwSize", wintypes.DWORD),
                        ("hDevice", wintypes.HANDLE),
                        ("wParam", wintypes.WPARAM)]

        class RAWINPUT(ctypes.Structure):
            _fields_ = [("header", RAWINPUTHEADER),
                        ("mouse", RAWMOUSE)]

        # --- Attach to self so _wnd_proc can see them ---
        self.PRAWINPUT = ctypes.POINTER(RAWINPUT)
        self.RAWINPUT = RAWINPUT
        self.RAWINPUTDEVICE = RAWINPUTDEVICE
        self.RAWINPUTHEADER = RAWINPUTHEADER  # <--- THIS WAS MISSING
        
        # Start the listener thread
        self.thread = threading.Thread(target=self._run_msg_loop, daemon=True)
        self.thread.start()

    def _run_msg_loop(self):
        # Create a hidden window to receive WM_INPUT messages
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self._wnd_proc
        wc.lpszClassName = "RawInputWindow"
        wc.hInstance = win32gui.GetModuleHandle(None)
        
        try:
            class_atom = win32gui.RegisterClass(wc)
        except Exception:
            # Class might already be registered if script restarted
            class_atom = win32gui.GetClassInfo(wc.hInstance, wc.lpszClassName)

        self.hwnd = win32gui.CreateWindow(class_atom, "Raw Input", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)

        # Register Generic Mouse (Usage Page 1, Usage 2)
        rid = self.RAWINPUTDEVICE(1, 2, 0x00000100, self.hwnd)
        
        if not ctypes.windll.user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
            print("Failed to register Raw Input device")
            return

        # Pump messages
        msg = wintypes.MSG()
        while ctypes.windll.user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == 0x00FF: # WM_INPUT
            data_size = wintypes.DWORD(0)
            
            # Get size of data
            ctypes.windll.user32.GetRawInputData(
                lparam, 
                0x10000003, # RID_INPUT
                None, 
                ctypes.byref(data_size), 
                ctypes.sizeof(self.RAWINPUTHEADER) # Now this works
            )
            
            if data_size.value > 0:
                raw_data = self.RAWINPUT()
                
                # Get actual data
                copied = ctypes.windll.user32.GetRawInputData(
                    lparam, 
                    0x10000003, 
                    ctypes.byref(raw_data), 
                    ctypes.byref(data_size), 
                    ctypes.sizeof(self.RAWINPUTHEADER)
                )

                if copied == data_size.value:
                    if raw_data.header.dwType == 0: # RIM_TYPEMOUSE
                        dx = raw_data.mouse.lLastX
                        dy = raw_data.mouse.lLastY
                        button_flags = raw_data.mouse.ulButtons & 0xFFFF
                        button_data = ctypes.c_short((raw_data.mouse.ulButtons >> 16) & 0xFFFF).value
                        # Accumulate
                        with self.lock:
                            self.x += dx
                            self.y += dy
                            if button_flags & 0x0400: # RI_MOUSE_WHEEL
                                if button_data > 0:
                                    self.scroll_up += button_data / WHEEL_DELTA
                                elif button_data < 0:
                                    self.scroll_down += abs(button_data) / WHEEL_DELTA
        
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def get_and_reset(self):
        """Returns accumulated mouse deltas and wheel events since last call, then resets counters."""
        with self.lock:
            dx, dy = self.x, self.y
            scroll_up, scroll_down = self.scroll_up, self.scroll_down
            self.x = 0
            self.y = 0
            self.scroll_up = 0
            self.scroll_down = 0
        return dx, dy, scroll_up, scroll_down

# ==========================================
# MAIN SCRIPT
# ==========================================

GAME_NAME = selected_game

key_states = {key_name: 0 for key_name in get_key_names(GAME_NAME)}
KEY_TO_VK = {
    "Key.shift": win32con.VK_SHIFT,
    "Key.space": win32con.VK_SPACE,
    "Key.ctrl_l": win32con.VK_LCONTROL,
    "Key.ctrl_r": win32con.VK_RCONTROL,
    "Key.alt_l": win32con.VK_LMENU,
    "Key.alt_r": win32con.VK_RMENU,
    "Key.tab": win32con.VK_TAB,
    "Key.esc": win32con.VK_ESCAPE,
}
ALL_MOUSE_BUTTON_TO_VK = {
    "left_click": win32con.VK_LBUTTON,
    "right_click": win32con.VK_RBUTTON,
    "middle_click": win32con.VK_MBUTTON,
}
SCROLL_ACTION_NAMES = {"scroll_up", "scroll_down"}
configured_mouse_buttons = get_mouse_button_names(GAME_NAME)
unsupported_mouse_buttons = [
    button_name
    for button_name in configured_mouse_buttons
    if button_name not in ALL_MOUSE_BUTTON_TO_VK and button_name not in SCROLL_ACTION_NAMES
]
if unsupported_mouse_buttons:
    raise ValueError(f"Unsupported mouse buttons for {GAME_NAME}: {unsupported_mouse_buttons}")
MOUSE_BUTTON_TO_VK = {
    button_name: ALL_MOUSE_BUTTON_TO_VK[button_name]
    for button_name in configured_mouse_buttons
    if button_name in ALL_MOUSE_BUTTON_TO_VK
}
VK_OEM_4 = getattr(win32con, "VK_OEM_4", 0xDB)
DATA_COLLECTION_CONTROLS = {
    "start": "1",
    "stop": "2",
}
START_COLLECTION_VKS = (ord(DATA_COLLECTION_CONTROLS["start"]), getattr(win32con, "VK_NUMPAD1", 0x61))
STOP_COLLECTION_VKS = (ord(DATA_COLLECTION_CONTROLS["stop"]), getattr(win32con, "VK_NUMPAD2", 0x62))
HOTKEY_VKS = {
    "start": START_COLLECTION_VKS,
    "stop": STOP_COLLECTION_VKS,
    "print_mouse": (VK_OEM_4,),
}
hotkey_prev_down = {name: False for name in HOTKEY_VKS}

track_rare_keys = False
# Percent of historical frames an action can be active and still be tracked as rare.
tracked_key_thresh = 5.0
tracked_rare_keys = []
rare_key_clip_seconds = 4.0

data_directory = game_data_root(GAME_NAME)
cam = None

collecting_data = False
recording_lock = threading.Lock()

raw_mouse = None

# Variables for video writing
video_writer = None
csv_writer = None
csv_file = None
start_time = None
last_frame_time = None
next_frame_deadline = None
active_recording_base = None
active_recording_reason = None
clip_start_reference_time = None
clip_last_written_source_time = None
auto_clip_end_time = None
capture_history = deque()
previous_input_state = None
tracked_window_hwnd = None
tracked_window_title = None
rare_tracking_was_focused = None
COLLECT_FPS = 20

fps = COLLECT_FPS
target_frame_time = 1 / fps
frame_size = (512, 512)

size = (1, 1)
scale = (1.0, 1.0)

# FFmpegWriter Class
class FFmpegWriter:
    def __init__(self, out_path, fps, width, height, ffmpeg_path=None, use_nvenc_codec="h264_nvenc",
                 container_ext="mp4", cq=23, preset="p5", pix_fmt_in="bgr24", stderr_path=None,
                 gop_size=16):
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt_in = pix_fmt_in
        self.container_ext = container_ext
        self.gop_size = int(gop_size)
        self.stderr_file = None

        self.final_file = out_path if out_path.endswith(f".{container_ext}") else f"{out_path}.{container_ext}"
        base_path, ext = os.path.splitext(self.final_file)
        self.out_file = f"{base_path}_temp{ext}"

        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Add ffmpeg to PATH or set FFMPEG_PATH.")

        self.cmd = [
            self.ffmpeg_path, "-y", "-f", "rawvideo", "-pix_fmt", self.pix_fmt_in,
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "pipe:0", "-an", "-c:v", use_nvenc_codec,
            "-preset", preset, "-rc", "vbr", "-cq", str(cq), "-pix_fmt", "yuv420p",
            "-g", str(self.gop_size), "-keyint_min", str(self.gop_size),
            "-bf", "0", "-forced-idr", "1", "-movflags", "+faststart", self.out_file
        ]

        stderr_target = subprocess.DEVNULL
        if stderr_path and LOG_FFMPEG_STDERR:
            self.stderr_file = open(stderr_path, "w", encoding="utf-8", errors="replace")
            stderr_target = self.stderr_file

        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_target)

    def write(self, frame_bgr):
        if not self.proc or not self.proc.stdin or self.proc.stdin.closed:
            return False

        try:
            self.proc.stdin.write(frame_bgr.tobytes())
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def release(self):
        if self.proc:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            return_code = self.proc.returncode
            self.proc = None
            if self.stderr_file is not None:
                self.stderr_file.close()
                self.stderr_file = None
            if return_code == 0:
                os.replace(self.out_file, self.final_file)
            else:
                print(f"FFmpeg failed with exit code {return_code}; leaving temp video: {self.out_file}")

ffmpeg_writer = None
FFMPEG_PATH = None # Set this if ffmpeg isn't in system PATH


def csv_value_is_active(value):
    try:
        return float(value) > 0.5
    except (TypeError, ValueError):
        return False


def available_capture_key_names():
    return list(key_states.keys()) + list(configured_mouse_buttons)


def iter_existing_data_csv_paths():
    if not os.path.isdir(data_directory):
        return []
    return [
        os.path.join(data_directory, filename)
        for filename in sorted(os.listdir(data_directory))
        if filename.startswith("run_") and filename.lower().endswith(".csv")
    ]


def compute_tracked_rare_keys_from_history():
    available_keys = available_capture_key_names()
    sample_counts = {key_name: 0 for key_name in available_keys}
    active_counts = {key_name: 0 for key_name in available_keys}
    csv_count = 0
    row_count = 0

    for csv_path in iter_existing_data_csv_paths():
        try:
            with open(csv_path, "r", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                sampled_keys = [key_name for key_name in available_keys if key_name in fieldnames]
                if not sampled_keys:
                    continue

                csv_count += 1
                for row in reader:
                    row_count += 1
                    for key_name in sampled_keys:
                        sample_counts[key_name] += 1
                        if csv_value_is_active(row.get(key_name)):
                            active_counts[key_name] += 1
        except OSError as exc:
            print(f"Warning: failed to read {csv_path}: {exc}")
        except csv.Error as exc:
            print(f"Warning: failed to parse {csv_path}: {exc}")

    try:
        threshold_percent = float(tracked_key_thresh)
    except (TypeError, ValueError):
        threshold_percent = 0.0
    threshold_percent = max(0.0, threshold_percent)

    rare_keys = []
    rates = {}
    for key_name in available_keys:
        samples = sample_counts[key_name]
        if samples <= 0:
            continue
        active_percent = (active_counts[key_name] / samples) * 100.0
        rates[key_name] = active_percent
        if active_percent < threshold_percent:
            rare_keys.append(key_name)

    return rare_keys, rates, csv_count, row_count, threshold_percent


def enumerate_visible_windows():
    windows = []

    def collect_window(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return True
        windows.append((hwnd, title))
        return True

    win32gui.EnumWindows(collect_window, None)
    return windows


def select_tracking_window():
    global tracked_window_hwnd, tracked_window_title

    windows = enumerate_visible_windows()
    if not windows:
        print("No visible windows found for rare-key focus tracking.")
        return False

    print("Select the window to track rare keys in:")
    for index, (hwnd, title) in enumerate(windows, start=1):
        print(f"{index}. {title} (hwnd={hwnd})")

    while True:
        selection = input("Window number: ").strip()
        try:
            selected_index = int(selection)
        except ValueError:
            print("Enter a window number from the list.")
            continue

        if 1 <= selected_index <= len(windows):
            tracked_window_hwnd, tracked_window_title = windows[selected_index - 1]
            print(f"Tracking rare keys only while focused: {tracked_window_title}")
            return True

        print(f"Enter a number between 1 and {len(windows)}.")


def is_tracked_window_focused():
    if tracked_window_hwnd is None:
        return False
    try:
        return win32gui.GetForegroundWindow() == tracked_window_hwnd
    except Exception as exc:
        print(f"Warning: failed to check focused window: {exc}")
        return False


def initialize_capture_runtime():
    global cam, raw_mouse, size, scale, track_rare_keys

    if track_rare_keys and not select_tracking_window():
        track_rare_keys = False
        print("Rare-key tracking disabled because no tracking window was selected.")

    os.makedirs(data_directory, exist_ok=True)
    cam = dxcam.create(output_color='BGR')
    raw_mouse = RawInputReader()
    size = pdi.size()
    scale = (frame_size[0] / size[0], frame_size[1] / size[1])
    print(size, scale)

    if track_rare_keys:
        rare_keys, rare_key_rates, rare_csv_count, rare_row_count, threshold_percent = (
            compute_tracked_rare_keys_from_history()
        )
        tracked_rare_keys[:] = rare_keys
        print(
            "Auto-selected tracked_rare_keys:",
            f"threshold=<{threshold_percent:.3g}%",
            f"csvs={rare_csv_count}",
            f"rows={rare_row_count}",
            f"keys={tracked_rare_keys}",
        )
        if tracked_rare_keys:
            rate_summary = ", ".join(
                f"{key_name}={rare_key_rates[key_name]:.3g}%"
                for key_name in tracked_rare_keys
            )
            print(f"Rare-key active rates: {rate_summary}")

        available_capture_keys = set(available_capture_key_names())
        invalid_tracked_rare_keys = [key_name for key_name in tracked_rare_keys if key_name not in available_capture_keys]
        if invalid_tracked_rare_keys:
            print(f"Warning: ignoring unknown tracked_rare_keys: {invalid_tracked_rare_keys}")
        tracked_rare_keys[:] = [key_name for key_name in tracked_rare_keys if key_name in available_capture_keys]
        if tracked_rare_keys:
            print(
                "Rare-key tracking enabled.",
                f"Capturing +/- {rare_key_clip_seconds:.1f}s around transitions for: {tracked_rare_keys}",
            )
        else:
            track_rare_keys = False
            print("Rare-key tracking disabled because tracked_rare_keys is empty.")
    else:
        print(
            f"Press '{DATA_COLLECTION_CONTROLS['start']}' to start data collection "
            f"and '{DATA_COLLECTION_CONTROLS['stop']}' to stop."
        )


def stop_recording():
    global collecting_data, csv_file, csv_writer, start_time, ffmpeg_writer
    global last_frame_time, next_frame_deadline, active_recording_base, active_recording_reason
    global clip_start_reference_time, clip_last_written_source_time, auto_clip_end_time

    with recording_lock:
        was_collecting = collecting_data or ffmpeg_writer is not None or csv_file is not None
        collecting_data = False
        writer = ffmpeg_writer
        csv_handle = csv_file
        finished_base = active_recording_base
        finished_reason = active_recording_reason
        ffmpeg_writer = None
        csv_file = None
        csv_writer = None
        start_time = None
        last_frame_time = None
        next_frame_deadline = None
        active_recording_base = None
        active_recording_reason = None
        clip_start_reference_time = None
        clip_last_written_source_time = None
        auto_clip_end_time = None

    if csv_handle:
        csv_handle.close()
    if writer:
        writer.release()
    disable_high_resolution_timer()

    return {
        "was_collecting": was_collecting,
        "base_path": finished_base,
        "reason": finished_reason,
    }


def key_name_to_vk(key_name):
    if key_name in KEY_TO_VK:
        return KEY_TO_VK[key_name]
    if len(key_name) == 1 and key_name.isascii():
        return ord(key_name.upper())
    return None


def is_key_down(vk_code):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def is_any_vk_down(vk_codes):
    return any(is_key_down(vk_code) for vk_code in vk_codes)


def consume_hotkey_press(name):
    down = is_any_vk_down(HOTKEY_VKS[name])
    pressed = down and not hotkey_prev_down[name]
    hotkey_prev_down[name] = down
    return pressed


def snapshot_key_states():
    snapshot = {}
    for key_name in key_states.keys():
        vk_code = key_name_to_vk(key_name)
        snapshot[key_name] = 1 if vk_code is not None and is_key_down(vk_code) else 0
    return snapshot


def snapshot_mouse_click_states():
    return {
        button_name: bool(is_key_down(vk_code))
        for button_name, vk_code in MOUSE_BUTTON_TO_VK.items()
    }


def snapshot_mouse_scroll_states(scroll_up_count, scroll_down_count):
    return {
        "scroll_up": float(scroll_up_count),
        "scroll_down": float(scroll_down_count),
    }


def tracked_rare_key_set():
    return set(tracked_rare_keys)


def trim_capture_history(now):
    cutoff = now - max(float(rare_key_clip_seconds), 0.0) - target_frame_time
    while capture_history and capture_history[0]["time"] < cutoff:
        capture_history.popleft()


def detect_rare_key_transitions(prev_state, current_state):
    if prev_state is None:
        return []

    transitions = []
    tracked_keys = tracked_rare_key_set()
    for key_name in tracked_keys:
        prev_value = 1 if bool(prev_state.get(key_name, 0)) else 0
        curr_value = 1 if bool(current_state.get(key_name, 0)) else 0
        if prev_value != curr_value:
            transitions.append((key_name, "pressed" if curr_value else "released"))
    return transitions


def frame_entry_to_csv_row(frame_entry, frame_dt):
    elapsed = max(0.0, float(frame_entry["time"]) - float(clip_start_reference_time))
    row = [elapsed, max(0.0, float(frame_dt))]

    input_state = frame_entry["input_state"]
    for key_name in key_states.keys():
        row.append(int(bool(input_state[key_name])))

    for button_name in configured_mouse_buttons:
        if button_name in SCROLL_ACTION_NAMES:
            row.append(float(input_state.get(button_name, 0.0)))
        else:
            row.append(1 if bool(input_state.get(button_name, 0)) else 0)
    row.extend([int(frame_entry["raw_dx"]), int(frame_entry["raw_dy"])])
    return row


def append_frame_to_active_clip(frame_entry):
    global clip_start_reference_time, clip_last_written_source_time

    with recording_lock:
        if not collecting_data or ffmpeg_writer is None or csv_writer is None:
            return True

        frame_time = float(frame_entry["time"])
        if clip_last_written_source_time is not None and frame_time <= float(clip_last_written_source_time):
            return True

        if clip_start_reference_time is None:
            clip_start_reference_time = frame_time

        if not ffmpeg_writer.write(frame_entry["frame"]):
            return False

        if clip_last_written_source_time is None:
            frame_dt = target_frame_time
        else:
            frame_dt = frame_time - float(clip_last_written_source_time)

        csv_writer.writerow(frame_entry_to_csv_row(frame_entry, frame_dt))
        clip_last_written_source_time = frame_time
        return True

def get_cursor_info():
    """Get cursor position, visibility, and icon handle"""
    try:
        flags, hcursor, (x, y) = win32gui.GetCursorInfo()
        return flags, hcursor, x, y
    except:
        return None, None, 0, 0

def _cursor_bgra_from_hicon(hcursor):
    # Use the color bitmap from the cursor handle (has alpha)
    try:
        fIcon, xHot, yHot, hbmMask, hbmColor = win32gui.GetIconInfo(hcursor)
        if not hbmColor:
            return None, xHot, yHot
        bmp = win32ui.CreateBitmapFromHandle(hbmColor)
        info = bmp.GetInfo()
        w, h = info['bmWidth'], info['bmHeight']
        raw = bmp.GetBitmapBits(True)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return arr, xHot, yHot
    except:
        return None, 0, 0
    finally:
        # Cleanup GDI objects is important to prevent leaks
        if 'hbmMask' in locals() and hbmMask: win32gui.DeleteObject(hbmMask)
        if 'hbmColor' in locals() and hbmColor: win32gui.DeleteObject(hbmColor)

def draw_cursor_on_image(img_bgr, cursor_x, cursor_y, hcursor):
    try:
        cur_bgra, hot_x, hot_y = _cursor_bgra_from_hicon(hcursor)
        if cur_bgra is None:
            mx = int(cursor_x * img_bgr.shape[1] / size[0])
            my = int(cursor_y * img_bgr.shape[0] / size[1])
            cv2.circle(img_bgr, (mx, my), 6, (0, 255, 0), 2)
            return img_bgr
        
        # Resize cursor
        cur_h, cur_w = cur_bgra.shape[:2]
        new_w, new_h = int(cur_w * scale[0]), int(cur_h * scale[1])
        if new_w <= 0 or new_h <= 0: return img_bgr
        
        cur_bgra = cv2.resize(cur_bgra, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        hot_x = int(hot_x * scale[0])
        hot_y = int(hot_y * scale[1])

        # Destination coordinates
        dst_x = int(cursor_x * scale[0]) - hot_x
        dst_y = int(cursor_y * scale[1]) - hot_y
        
        # Blending logic (same as before but safer bounds)
        ch, cw = cur_bgra.shape[:2]
        x0 = max(dst_x, 0)
        y0 = max(dst_y, 0)
        x1 = min(dst_x + cw, img_bgr.shape[1])
        y1 = min(dst_y + ch, img_bgr.shape[0])
        
        if x0 >= x1 or y0 >= y1: return img_bgr

        cx0 = x0 - dst_x
        cy0 = y0 - dst_y
        cx1 = cx0 + (x1 - x0)
        cy1 = cy0 + (y1 - y0)

        roi = img_bgr[y0:y1, x0:x1]
        cur_roi = cur_bgra[cy0:cy1, cx0:cx1]
        
        cur_bgr = cur_roi[:, :, :3].astype(np.float32)
        alpha = cur_roi[:, :, 3:4].astype(np.float32) / 255.0
        
        out = (cur_bgr * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        roi[:] = out
        return img_bgr
    except Exception:
        return img_bgr

def start_recording(prebuffer_entries=None, clip_reason=None, reset_mouse_accumulator=True):
    global collecting_data, fps, target_frame_time, video_writer, csv_file, csv_writer
    global start_time, ffmpeg_writer, last_frame_time, next_frame_deadline
    global active_recording_base, active_recording_reason, clip_start_reference_time
    global clip_last_written_source_time

    buffered_entries = list(prebuffer_entries or [])
    with recording_lock:
        if collecting_data:
            return False

        if reset_mouse_accumulator and raw_mouse is not None:
            raw_mouse.get_and_reset()
        fps = COLLECT_FPS
        target_frame_time = 1 / fps

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(data_directory, f"run_{timestamp_str}")
        video_filename_no_ext = base
        csv_filename = f"{base}.csv"

        try:
            ffmpeg_writer = FFmpegWriter(
                out_path=video_filename_no_ext,
                fps=fps,
                width=frame_size[0],
                height=frame_size[1],
                ffmpeg_path=FFMPEG_PATH,
                use_nvenc_codec="h264_nvenc",
                container_ext="mp4",
                cq=23,
                preset="p5",
                pix_fmt_in="bgr24",
                stderr_path=None,
                gop_size=16,
            )
        except Exception as exc:
            collecting_data = False
            print(f"Failed to start FFmpeg: {exc}")
            return False

        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['timestamp', 'dt']
            + list(key_states.keys())
            + list(configured_mouse_buttons)
            + ['delta_x', 'delta_y']
        )

        enable_high_resolution_timer()
        start_time = time.perf_counter()
        last_frame_time = None
        next_frame_deadline = start_time
        active_recording_base = base
        active_recording_reason = clip_reason
        clip_start_reference_time = buffered_entries[0]["time"] if buffered_entries else None
        clip_last_written_source_time = None
        collecting_data = True

    print(f"Started: {video_filename_no_ext}.mp4, fps: {fps}")
    if clip_reason:
        print(f"Reason: {clip_reason}")

    for frame_entry in buffered_entries:
        if not append_frame_to_active_clip(frame_entry):
            stop_recording()
            print(f"Stopped: failed while writing buffered clip frames for {clip_reason}.")
            return False

    return True


def handle_rare_key_trigger(trigger_time, transitions):
    global auto_clip_end_time

    transition_desc = ", ".join(f"{key_name}:{event_name}" for key_name, event_name in transitions)
    new_clip_end_time = float(trigger_time) + float(rare_key_clip_seconds)
    was_collecting = False
    old_end_time = None
    with recording_lock:
        was_collecting = collecting_data
        old_end_time = auto_clip_end_time

    if not was_collecting:
        clip_start_time = float(trigger_time) - float(rare_key_clip_seconds)
        buffered_entries = [entry for entry in capture_history if float(entry["time"]) >= clip_start_time]
        started = start_recording(
            prebuffer_entries=buffered_entries,
            clip_reason=f"rare-key transition ({transition_desc})",
            reset_mouse_accumulator=False,
        )
        if started:
            with recording_lock:
                auto_clip_end_time = new_clip_end_time
            print(
                "Auto clip started:",
                f"trigger={transition_desc}",
                f"buffered_frames={len(buffered_entries)}",
                f"until=+{rare_key_clip_seconds:.1f}s",
            )
        return

    with recording_lock:
        if auto_clip_end_time is None:
            auto_clip_end_time = new_clip_end_time
        else:
            auto_clip_end_time = max(float(auto_clip_end_time), new_clip_end_time)
        updated_end_time = auto_clip_end_time

    if old_end_time is None or float(updated_end_time) > float(old_end_time):
        extension_seconds = float(updated_end_time) - float(trigger_time)
        print(
            "Auto clip extended:",
            f"trigger={transition_desc}",
            f"remaining={extension_seconds:.2f}s",
        )


def handle_control_hotkeys():
    try:
        if not track_rare_keys:
            if consume_hotkey_press("start"):
                start_recording()

            if consume_hotkey_press("stop"):
                stop_info = stop_recording()
                if stop_info["was_collecting"]:
                    print("Stopped.")

        if consume_hotkey_press("print_mouse"):
            print(f'Mouse pos: {pdi.position()}')
    except Exception as exc:
        print(f"Error handling hotkeys: {exc}")

def main():
    global previous_input_state, rare_tracking_was_focused

    initialize_capture_runtime()
    try:
        capture_deadline = None
        while True:
            handle_control_hotkeys()
            rare_tracking_focused = bool(track_rare_keys and is_tracked_window_focused())
            if track_rare_keys and rare_tracking_focused != rare_tracking_was_focused:
                rare_tracking_was_focused = rare_tracking_focused
                if rare_tracking_focused:
                    print(f"Rare-key tracking active: {tracked_window_title}")
                else:
                    capture_history.clear()
                    previous_input_state = None
                    if raw_mouse is not None:
                        raw_mouse.get_and_reset()
                    print(f"Rare-key tracking paused until focused: {tracked_window_title}")

            with recording_lock:
                is_collecting = collecting_data
                active_auto_clip_end = auto_clip_end_time

            should_capture = bool(rare_tracking_focused or is_collecting)

            if should_capture:
                capture_started = time.perf_counter()
                screenshot = cam.grab()
                if screenshot is None:
                    time.sleep(0.001)
                    continue

                input_state = snapshot_key_states()
                input_state.update(snapshot_mouse_click_states())

                raw_dx, raw_dy, scroll_up_count, scroll_down_count = raw_mouse.get_and_reset()
                input_state.update(snapshot_mouse_scroll_states(scroll_up_count, scroll_down_count))
                input_sample_time = time.perf_counter()

                screenshot = cv2.resize(screenshot, (frame_size[0], frame_size[1]))

                if DRAW_CURSOR_OVERLAY:
                    mx, my = pdi.position()
                    flags, hcursor, _, _ = get_cursor_info()
                    if flags == 1 and hcursor:
                        screenshot = draw_cursor_on_image(screenshot, mx, my, hcursor)

                frame_now = 0.5 * (capture_started + input_sample_time)
                frame_entry = {
                    "time": frame_now,
                    "frame": screenshot.copy(),
                    "input_state": dict(input_state),
                    "raw_dx": int(raw_dx),
                    "raw_dy": int(raw_dy),
                }

                if rare_tracking_focused:
                    capture_history.append(frame_entry)
                    trim_capture_history(frame_now)
                    transitions = detect_rare_key_transitions(previous_input_state, input_state)
                    if transitions:
                        handle_rare_key_trigger(frame_now, transitions)

                    previous_input_state = dict(input_state)

                with recording_lock:
                    is_collecting = collecting_data
                    active_auto_clip_end = auto_clip_end_time

                if is_collecting:
                    if not append_frame_to_active_clip(frame_entry):
                        stop_recording()
                        print("Stopped: FFmpeg pipe closed unexpectedly.")
                        continue

                if track_rare_keys and is_collecting and active_auto_clip_end is not None and frame_now >= float(active_auto_clip_end):
                    stop_info = stop_recording()
                    if stop_info["was_collecting"]:
                        print(f"Auto clip saved: {stop_info['base_path']}.mp4")
                    continue

                if capture_deadline is None or frame_now - capture_deadline > target_frame_time:
                    capture_deadline = frame_now
                capture_deadline += target_frame_time
                sleep_until(capture_deadline)
            else:
                capture_deadline = None
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording()
        print("Terminated.")


if __name__ == "__main__":
    main()
