from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.pyplot import figure

from square_marcher import SquareMarcher

from square_marcher import NumericType
from typing import (
    Optional,
    overload
)


@overload
def generate_frames(
    square_marcher: SquareMarcher,
    speed: NumericType,
    *,
    z_start: NumericType,
    z_speed: NumericType,
    frames_count: int,
    fig: Optional[Figure] = None
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    """
    Utility function for creating frames to be used in ``matplotlib.animation.ArtistAnimation``

    ## Parameters:
    ``square_marcher``: a ``SquareMarcher`` object configured to be run.
    ``z_start``
    """
@overload
def generate_frames(
    square_marcher: SquareMarcher,
    speed: NumericType,
    *zs: NumericType,
    fig: Optional[Figure] = None,
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    """
    Overloaded 2
    """

def generate_frames(
    square_marcher: SquareMarcher,
    speed: NumericType,
    *args: NumericType,
    z_start: NumericType = None,
    z_speed: NumericType = None,
    frames_count: int = None,
    fig: Optional[Figure] = None
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    if not isinstance(square_marcher, SquareMarcher):
        raise TypeError(f'square_marcher must be of type SquareMarcher: {square_marcher=}')
    if not isinstance(speed, NumericType):
        raise TypeError(f"'speed' must be of type {NumericType}: {speed}")
    
    if args:
        if not (z_start is None and z_speed is None and frames_count is None):
            raise TypeError("generate_frames() received too many arguments")
        zs = args
    else:
        missing = []
        if z_start is None: missing.append("'z_start'")
        if z_speed is None: missing.append("'z_speed'")
        if frames_count is None: missing.append("'frames_count'")
        if missing:
            raise TypeError(f"generate_frames() missing {len(missing)} required keyword arguments: {', '.join(missing)}")
        
        if not isinstance(z_start, NumericType):
            raise TypeError(f"'z_start must be of type {NumericType}: {z_start}")
        if not isinstance(z_speed, NumericType):
            raise TypeError(f"'z_speed must be of type {NumericType}: {z_speed}")
        if not isinstance(frames_count, int):
            raise TypeError(f"'frames_count must be of type int: {frames_count}")
        elif frames_count < 1:
            raise ValueError(f"'frames_count' must be positive: {frames_count}")
        
        zs = (z_start + z_speed * i for i in range(frames_count))

    if not fig is None:
        if not isinstance(fig, Figure):
            raise TypeError(f"'fig must be of type matplotlib.figure.Figure: {fig}")
    else:
        fig = figure(figsize=(6, 6))
        fig.tight_layout()
        fig.subplots_adjust(0, 0, 1, 1, None, None)


    ax = fig.add_subplot(111)
    ax.axis('off')
    frames = []
    for z in zs:
        _, artists = square_marcher.run(ax, z, speed, animated=True)
        frames.append(artists)
    return fig, frames