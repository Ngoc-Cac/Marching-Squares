import numpy as np

from matplotlib.image import imread
from matplotlib.pyplot import figure

from marching_squares.algo import draw_contours
from marching_squares.square_marcher import SquareMarcher

# These are typing stuff and are not involved in the code
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from marching_squares import NumericType
from typing import (
    Literal,
    Optional,
    overload
)


_GRAYSCALE_COEFS = 0.2989, 0.5870, 0.1140

@overload
def generate_frames(
    square_marcher: SquareMarcher,
    speed: NumericType,
    *,
    z_start: NumericType,
    z_speed: NumericType,
    frame_counts: int,
    fig: Optional[Figure] = None
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    """
    Utility function for creating frames to be used in ``matplotlib.animation.ArtistAnimation``.
    The frames will be created from a z value starting at ``z_start`` and incrementnig by ``z_speed``\
        for each ``frame_counts``.

    ## Parameters:
    ``square_marcher: SquareMarcher``
    
        a ``SquareMarcher`` object already configured to be run

    ``speed: int | float``

        the x and y speed when generting the nosimap. Larger value of speed will cause\
            the noisemap to be noisier.

    ``z_start: int | float``
        
        the starting z value

    ``z_speed: int | float``
        
        the increment to go up by each frame. For large increment, the change through each frame\
            might be very chaotic

    ``frame_counts: int``
        
        the number of frames in the animation, must be positive

    ``fig: matplotlib.figure.Figure | None``

        the figure on which the animation is drawn on. If None is given,\
            a 6in x 6in ``matplotlib.figure.Figure`` is created.

    # Returns
    The ``matplotlib.figure.Figure`` that was used to draw on and a list of lists\
        of ``matplotlib.artist.Artist``. This list can be passed straight into a\
        ``matplotlib.animation.ArtistAnimation``. Each sub-list of this list contains\
        the artists to be drawn on that frame.
    """
@overload
def generate_frames(
    square_marcher: SquareMarcher,
    speed: NumericType,
    *zs: NumericType,
    fig: Optional[Figure] = None,
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    """
    Utility function for creating frames to be used in ``matplotlib.animation.ArtistAnimation``

    ## Parameters:
    ``square_marcher: SquareMarcher``
    
        a ``SquareMarcher`` object already configured to be run

    ``speed: int | float``

        the x and y speed when generting the nosimap. Larger value of speed will cause\
            the noisemap to be noisier.

    ``*zs: int | float``

        the z values on each frame used to generate the noisemap

    ``fig: matplotlib.figure.Figure | None``

        the figure on which the animation is drawn on. If None is given,\
            a 6in x 6in ``matplotlib.figure.Figure`` is created.

    # Returns
    The ``matplotlib.figure.Figure`` that was used to draw on and a list of lists\
        of ``matplotlib.artist.Artist``. This list can be passed straight into a\
        ``matplotlib.animation.ArtistAnimation``. Each sub-list of this list contains\
        the artists to be drawn on that frame.
    """

def generate_frames(
    square_marcher: SquareMarcher,
    speed: NumericType,
    *args: NumericType,
    z_start: NumericType = None,
    z_speed: NumericType = None,
    frame_counts: int = None,
    fig: Optional[Figure] = None
) -> tuple[Figure, list[list[AxesImage, Line2D]]]:
    if not isinstance(square_marcher, SquareMarcher):
        raise TypeError(f'square_marcher must be of type SquareMarcher: {square_marcher=}')
    if not isinstance(speed, NumericType):
        raise TypeError(f"'speed' must be of type {NumericType}: {speed}")
    
    if args:
        if not (z_start is None and z_speed is None and frame_counts is None):
            raise TypeError("generate_frames() received too many arguments")
        zs = args
    else:
        missing = []
        if z_start is None: missing.append("'z_start'")
        if z_speed is None: missing.append("'z_speed'")
        if frame_counts is None: missing.append("'frame_counts'")
        if missing:
            raise TypeError(f"generate_frames() missing {len(missing)} required keyword arguments: {', '.join(missing)}")
        
        if not isinstance(z_start, NumericType):
            raise TypeError(f"'z_start must be of type {NumericType}: {z_start}")
        if not isinstance(z_speed, NumericType):
            raise TypeError(f"'z_speed must be of type {NumericType}: {z_speed}")
        if not isinstance(frame_counts, int):
            raise TypeError(f"'frame_counts must be of type int: {frame_counts}")
        elif frame_counts < 1:
            raise ValueError(f"'frame_counts' must be positive: {frame_counts}")
        
        zs = (z_start + z_speed * i for i in range(frame_counts))

    if not fig is None:
        if not isinstance(fig, Figure):
            raise TypeError(f"'fig must be of type matplotlib.figure.Figure: {fig}")
    else:
        fig = figure(figsize=(6, 6))
        fig.tight_layout()
        fig.subplots_adjust(0, 0, 1, 1, None, None)


    ax = fig.add_subplot(111)
    ax.axis('off')
    frames = [square_marcher.run(ax, z, speed, animated=True)[1] for z in zs]
    return fig, frames


@overload
def contours_from_image(
    filepath: str,
    threshold: Literal['midpoint', 'average'] = 'midpoint',
    lerping: bool = True
):
    """
    """
@overload
def contours_from_image(
    filepath: str,
    threshold: int,
    lerping: bool = True
):
    """
    """
@overload
def contours_from_image(
    filepath: str,
    threshold: float,
    lerping: bool = True
):
    """
    """

def contours_from_image(
    filepath: str,
    threshold: Literal['midpoint', 'average'] | int | float = 'midpoint',
    lerping: bool = True
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    if not isinstance(filepath, str):
        raise TypeError('filepath must be a string')
    if not isinstance(lerping, bool):
        raise TypeError('lerping must be a bool')
    if isinstance(threshold, str):
        if not threshold in ('midpoint', 'average'):
            raise ValueError("threshold must be a string literal of 'midpoint' or 'average'")
    elif isinstance(threshold, int):
        if threshold > 100 or threshold < 0:
            raise ValueError('threshold must be an int between 0 and 100 if percentile is used')
        threshold_value = threshold
        threshold = 'percentile'
    elif isinstance(threshold, float):
        if threshold > 1 or threshold < 0:
            raise ValueError('If custom threshold is used, threshold value must be between 0 and 1')
        threshold_value = threshold
        threshold = 'custom'
    else: raise TypeError('Unrecognised type for threshold argument')

    rgb_img = imread(filepath) / 255
    grayscale = np.dot(rgb_img[...,:3], _GRAYSCALE_COEFS)

    if threshold == 'midpoint':
        threshold_value = (grayscale.min() + grayscale.max()) / 2
    elif threshold == 'average':
        threshold_value = grayscale.mean()
    elif threshold == 'percentile':
        threshold_value = np.percentile(grayscale, threshold_value)

    return grayscale, draw_contours(grayscale, threshold_value, lerping)