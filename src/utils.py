from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.pyplot import figure

from square_marcher import SquareMarcher

from square_marcher import NumericType
from typing import Optional

def generate_frames(square_marcher: SquareMarcher,
                    z_start: NumericType, z_speed: NumericType, speed: NumericType,
                    frames_count: int,
                    fig: Optional[Figure] = None
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    
    if fig is None:
        fig = figure(figsize=(6, 6))
        fig.tight_layout()
        fig.subplots_adjust(0, 0, 1, 1, None, None)
    ax = fig.add_subplot(111)
    ax.axis('off')

    frames = []
    for _ in range(frames_count):
        _, artists = square_marcher.run(ax, z_start, speed, animated=True)
        frames.append(artists)
        z_start += z_speed
    return fig, frames