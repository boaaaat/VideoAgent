import os
from typing import Dict, List, Optional


selected_game = "greenville"

key_names: Dict[str, List[str]] = {
    "doors": ["w", "a", "s", "d", "e", "1", "2", "3", "Key.shift", "Key.space"],
    "flee_the_facility": ["w", "a", "s", "d", "e", "Key.space", "Key.shift"],
    "lol": ["q", "w", "e", "r", "a", "f", "d", "c", "v", "b", "p", "x", "1", "2", "3", "4", "5", "Key.space", "Key.ctrl_l", "Key.tab"],
    "arc_raiders": ["w", "a", "s", "d", "e", "f", "h", "q", "v", "r", "x", "m", "g", "1", "2", "3", "Key.space", "Key.shift", "Key.ctrl_l", "Key.tab", "Key.alt_l"], #, "Key.esc"]
    "greenville": ['w', 'a', 's', 'd', 'e', 'q', 'c', 'z'],
    "greenville_test": ['w', 'a', 's', 'd', 'e', 'q', 'c', 'z']
}

mouse_button_names: Dict[str, List[str]] = {
    "doors": ["left_click", "right_click"],
    "flee_the_facility": ["left_click", "right_click"],
    "lol": ["left_click", "right_click"],
    "arc_raiders": ["left_click", "right_click", "middle_click", "scroll_up", "scroll_down"],
    'greenville': [],
    'greenville_test': []
}


def normalize_game_name(game_name: Optional[str] = None) -> str:
    name = selected_game if game_name is None else str(game_name)
    if name not in key_names:
        raise KeyError(f"Unknown game {name!r}. Available games: {sorted(key_names)}")
    if name not in mouse_button_names:
        raise KeyError(f"Missing mouse_button_names entry for game {name!r}.")
    return name


def get_key_names(game_name: Optional[str] = None) -> List[str]:
    return list(key_names[normalize_game_name(game_name)])


def get_mouse_button_names(game_name: Optional[str] = None) -> List[str]:
    return list(mouse_button_names[normalize_game_name(game_name)])


def game_data_root(game_name: Optional[str] = None, root: str = "data") -> str:
    return os.path.join(root, normalize_game_name(game_name))
