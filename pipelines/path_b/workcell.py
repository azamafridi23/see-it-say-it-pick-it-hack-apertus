"""The workcell's settings, and the record of one detected object.

``Workcell`` holds every value particular to this camera, table and object
set, loaded from ``workcell.ini``. ``DetectedObject`` is one object found
by Apertus, with the facts OpenCV measures about it.
"""
import configparser
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("workcell.ini")


@dataclass(frozen=True)
class Workcell:
    """The settings of one workcell; see ``workcell.ini`` for each value."""

    kinds: tuple[str, ...]
    max_box_area: float
    edge_margin: float
    y_scale: float
    y_shift_px: float
    pale_max_saturation: int
    hues: tuple[tuple[str, int], ...]
    stack_min_overlap: float
    stack_colour_margin: float
    mat_max_brightness: int
    part_kinds: tuple[str, ...]
    small_part_max_mm: float
    grey_part_min_aspect: float
    unpickable: frozenset[str]
    hand_gap_cm: float


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
    settings = configparser.ConfigParser()
    if not settings.read(path):
        raise FileNotFoundError(path)

    detection, parts = settings["detection"], settings["parts"]
    stacking, planner = settings["stacking"], settings["planner"]
    return Workcell(
        kinds=_lines(detection["kinds"]),
        max_box_area=detection.getfloat("max_box_area"),
        edge_margin=detection.getfloat("edge_margin"),
        y_scale=detection.getfloat("y_scale"),
        y_shift_px=detection.getfloat("y_shift_px"),
        pale_max_saturation=settings["colours"].getint(
            "pale_max_saturation"),
        hues=tuple((name, int(top))
                   for name, top in settings["colour hues"].items()),
        stack_min_overlap=stacking.getfloat("min_overlap"),
        stack_colour_margin=stacking.getfloat("colour_margin"),
        mat_max_brightness=settings["trays"].getint("mat_max_brightness"),
        part_kinds=_lines(parts["kinds"]),
        small_part_max_mm=parts.getfloat("small_max_mm"),
        grey_part_min_aspect=parts.getfloat("grey_min_aspect"),
        unpickable=frozenset(_lines(planner["unpickable"])),
        hand_gap_cm=planner.getfloat("hand_gap_cm"),
    )


@dataclass
class DetectedObject:
    """One object Apertus found, with the facts OpenCV measures.

    Attributes:
        id: Its number, from 1, left to right.
        kind: One of the workcell's kinds.
        box: ``[x1, y1, x2, y2]`` in whole pixels, corrected.
        pixel: ``[u, v]``, the box centre: the target pixel.
        colour: A colour name, once measured.
        width_mm: Its width on the table, once measured.
        gaps: ``{object id: cm}`` to every other object, once measured.
        on: The ids of the objects it lies on top of.
    """

    id: int
    kind: str
    box: list[int]
    pixel: list[int]
    colour: str | None = None
    width_mm: int | None = None
    gaps: dict[int, float] = field(default_factory=dict)
    on: list[int] = field(default_factory=list)
