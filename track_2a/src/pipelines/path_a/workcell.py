"""The workcell's settings, loaded from ``workcell.ini``.

Every value particular to this camera, table, object set and model choice
lives in ``workcell.ini``; the code holds only the logic. ``CELL`` is the
loaded file, shared by every module.

The Hack Apertus template's variables override the model settings:
LLM_NAME (the model), LLM_BASE_URL (the endpoint) and LLM_API_KEY. They
are read from the environment, else from the project's .env file.
"""
import configparser
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_PATH = Path(__file__).with_name("workcell.ini")
# track_2a/.env, three folders above src/pipelines/path_a/.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def api_key() -> str | None:
    """Return the endpoint key: LLM_API_KEY, else APERTUS_API_KEY."""
    return (os.environ.get("LLM_API_KEY")
            or os.environ.get("APERTUS_API_KEY"))


@dataclass(frozen=True)
class Workcell:
    """The settings of one workcell; see ``workcell.ini`` for each value."""

    # [apertus]
    model: str
    base_url: str
    single_call_prompt: str
    single_call_retries: int
    single_call_max_tokens: int
    agent_prompts: dict
    agent_retries: int
    agent_max_tokens: int
    parallel_requests: int
    # [sam3]
    sam3_model: str
    sam3_revision: str
    sam3_threshold: float
    sam3_mask_threshold: float
    sam3_prompts: tuple
    # [depth]
    depth_model: str
    depth_window_px: int
    # label tables
    labels: dict
    ids: dict
    label_tiers: dict
    containers: frozenset
    round_kinds: frozenset
    not_pickable: dict
    # [table]
    min_on_table: float
    mat_max_tray_overlap: float
    mat_max_brightness: int
    mat_open_px: int
    mat_min_share: float
    rail_min_grey: int
    # [objects]
    min_area_px: int
    same_iou: float
    inside: float
    saucer_max_mm: float
    plate_middle_share: float
    # [leftovers]
    leftover_margin_px: int
    leftover_open_px: int
    leftover_min_area_px: int
    # [stacking]
    on_min_cover: float
    touch_px: int
    maybe_min_contact_px: float
    maybe_min_higher: float
    # [hands]
    hand_reach_px: int
    hand_min_overlap_px: int
    # [colours], [colour hues]
    pale_max_saturation: float
    white_min_value: float
    black_max_value: float
    hues: tuple
    # [planning]
    uncertain_support: str
    # [apertus vision experiment]
    known_names: tuple


def _lines(value: str) -> tuple[str, ...]:
    """Split a multi-line INI value into its non-empty lines."""
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def load(path: Path = DEFAULT_PATH) -> Workcell:
    """Read a workcell's settings from an INI file.

    Args:
        path: The settings file; ``workcell.ini`` next to this module by
            default.

    Returns:
        The workcell's settings.

    Raises:
        FileNotFoundError: The settings file does not exist.
    """
    ini = configparser.ConfigParser(interpolation=None)
    if not ini.read(path):
        raise FileNotFoundError(path)

    apertus, sam3, depth = ini["apertus"], ini["sam3"], ini["depth"]
    table, objects, leftovers = ini["table"], ini["objects"], ini["leftovers"]
    stacking, hands, colours = ini["stacking"], ini["hands"], ini["colours"]
    agent_prompts = {role: apertus[f"{role}_prompt"]
                     for role in ("safety", "reference", "constraint")}
    return Workcell(
        model=os.environ.get("LLM_NAME") or apertus["model"],
        base_url=os.environ.get("LLM_BASE_URL") or apertus["base_url"],
        single_call_prompt=apertus["single_call_prompt"],
        single_call_retries=apertus.getint("single_call_retries"),
        single_call_max_tokens=apertus.getint("single_call_max_tokens"),
        agent_prompts=agent_prompts,
        agent_retries=apertus.getint("agent_retries"),
        agent_max_tokens=apertus.getint("agent_max_tokens"),
        parallel_requests=apertus.getint("parallel_requests"),
        sam3_model=sam3["model"],
        sam3_revision=sam3["revision"],
        sam3_threshold=sam3.getfloat("threshold"),
        sam3_mask_threshold=sam3.getfloat("mask_threshold"),
        sam3_prompts=_lines(sam3["prompts"]),
        depth_model=depth["model"],
        depth_window_px=depth.getint("window_px"),
        labels=dict(ini["labels"]),
        ids=dict(ini["ids"]),
        label_tiers={label: int(tier)
                     for label, tier in ini["label tiers"].items()},
        containers=frozenset(_lines(ini["kinds"]["containers"])),
        round_kinds=frozenset(_lines(ini["kinds"]["round"])),
        not_pickable=dict(ini["not pickable"]),
        min_on_table=table.getfloat("min_on_table"),
        mat_max_tray_overlap=table.getfloat("mat_max_tray_overlap"),
        mat_max_brightness=table.getint("mat_max_brightness"),
        mat_open_px=table.getint("mat_open_px"),
        mat_min_share=table.getfloat("mat_min_share"),
        rail_min_grey=table.getint("rail_min_grey"),
        min_area_px=objects.getint("min_area_px"),
        same_iou=objects.getfloat("same_iou"),
        inside=objects.getfloat("inside"),
        saucer_max_mm=objects.getfloat("saucer_max_mm"),
        plate_middle_share=objects.getfloat("plate_middle_share"),
        leftover_margin_px=leftovers.getint("margin_px"),
        leftover_open_px=leftovers.getint("open_px"),
        leftover_min_area_px=leftovers.getint("min_area_px"),
        on_min_cover=stacking.getfloat("on_min_cover"),
        touch_px=stacking.getint("touch_px"),
        maybe_min_contact_px=stacking.getfloat("maybe_min_contact_px"),
        maybe_min_higher=stacking.getfloat("maybe_min_higher"),
        hand_reach_px=hands.getint("reach_px"),
        hand_min_overlap_px=hands.getint("min_overlap_px"),
        pale_max_saturation=colours.getfloat("pale_max_saturation"),
        white_min_value=colours.getfloat("white_min_value"),
        black_max_value=colours.getfloat("black_max_value"),
        hues=tuple((name, int(top))
                   for name, top in ini["colour hues"].items()),
        uncertain_support=ini["planning"]["uncertain_support"],
        known_names=_lines(ini["apertus vision experiment"]["known"]),
    )


CELL = load()
