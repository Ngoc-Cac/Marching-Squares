import random, sys


import numpy as np

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from pynoise.noisemodule import Perlin


from typing import (
    Optional,
    TypeAlias,
    Union,
)

NumericType: TypeAlias = Union[int, float]


# the origin is at the top left corner, just like in any graphics drawing
def _contour_line(coordinates: tuple[int, int],
                  tleft: bool, tright: bool,
                  bright: bool, bleft: bool)\
    -> tuple[tuple[float, float, float, float]]:
    """
    Drawing contour lines based on the four corners
    """
    case_no = tleft * 8 + tright * 4 + bright * 2 + bleft
    x, y = coordinates
    if case_no == 0 or case_no == 15:
        return (), ()
    elif case_no == 1 or case_no == 14:
        return (x + .5, x, y + 1, y + .5), ()
    elif case_no == 2 or case_no == 13:
        return (x + 1, x + .5, y + .5, y + 1), ()
    elif case_no == 3 or case_no == 12:
        return (x + 1, x, y + .5, y + .5), ()
    elif case_no == 4 or case_no == 11:
        return (x + .5, x + 1, y, y + .5), ()
    elif case_no == 5:
        return (x + 1, x + .5, y + .5, y + 1), (x, x + .5, y + .5, y)
    elif case_no == 6 or case_no == 9:
        return (x + .5, x + .5, y, y + 1), ()
    elif case_no == 7 or case_no == 8:
        return (x, x + .5, y + .5, y), ()
    elif case_no == 10:
        return (x + .5, x + 1, y, y + .5), (x + .5, x, y + 1, y + .5)
    
def _draw_contour(grid: np.ndarray[NumericType],
                  threshold: NumericType)\
    -> list[tuple[float, float]]:
    """
    The marching square algorithm in its entirety.\\
    This is supposed to be a multi-purpose function for internal use.

    The function will return a list of edges. Each element of the list is\
        a tuple of two coordinates.\\
    **This list should be read in increments of 2**.\
        The first tuple is the `(x_from, x_to)` coordinates of the edge.\
        The second tuple is the `(y_from, y_to)` coordinates of the edge.

    This is configured for convenience of matplotlib plotting. If you have no intention\
        of post-processing the edges, and just want to visualise the edges using `ax.plot`\
        or similar means, see the below example::
        
        import matplotlib.pyplot as plt
        import numpy as np
        from square_marcher import _draw_contour
        grid = np.array([[0, 0, 0, 1],
                         [0, 1, 1, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 0]])
        edges = _draw_contour(grid, 0.5)
        plt.plot(*edges, color='blue')

    ## Parameters:
    ``grid``: a `numpy.ndarray` of shape `(rows, cols)`. Each element should be a scalar\\
    ``threshold``: a number representing the threshold to seperate the scalars. The scalars\
        will be divided into two groups using the threshold: one being greater than the threshold\
        and one being less than or equal to the threshold.

    ## Returns
    A list of tuples of floats. For more information, see above.
    """
    lines = []
    rows, cols = grid.shape
    mask = grid > threshold
    for i in range(rows - 1):
        for j in range(cols - 1):
            line1, line2 = _contour_line((j, i),
                                         mask[i    , j    ],
                                         mask[i    , j + 1],
                                         mask[i + 1, j + 1],
                                         mask[i + 1, j    ])
            if line1:
                lines.append(line1[:2])
                lines.append(line1[2:])
            if line2:
                lines.append(line2[:2])
                lines.append(line2[2:])
    return lines

class SquareMarcher():
    __slots__ = (
        '_dim',
        '_grid',
        '_perlin_model',
        '_prng',
    )
    def __init__(self, dimension: tuple[int, int], seed: Optional[int] = int):
        if not isinstance(dimension, tuple):
            raise TypeError('Dimension must be a tuple of two ints.')
        elif len(dimension) != 2 or any((not isinstance(num, int) or num < 1) for num in dimension):
            raise ValueError('Number of rows and columns must be positive!')
        if not isinstance(seed, int):
            raise TypeError('Random seed must be an int')

        self._dim = dimension
        if seed is None: seed = random.randint(0, sys.maxsize)
        
        self._initialize_grid()
        self._prng = random.Random(seed)
        self._perlin_model = Perlin(octaves=2, seed=seed)

    
    @property
    def dimension(self) -> tuple[int, int]:
        return self._dim
    @dimension.setter
    def dimension(self, new_value: tuple[int, int]):
        if not isinstance(new_value, tuple):
            raise TypeError('Dimension must be a tuple of two ints.')
        elif len(new_value) != 2 or any((not isinstance(num, int) or num < 1) for num in new_value):
            raise ValueError('Number of rows and columns must be positive!')
        self._dim = new_value
        self._initialize_grid()

    @property
    def grid(self) -> np.ndarray[float]:
        return self._grid
    
    @property
    def octaves(self):
        return self._perlin_model.octaves
    @octaves.setter
    def octaves(self, value: int):
        if not isinstance(value, int):
            raise TypeError('octaves must be a positive int')
        elif value < 1:
            raise ValueError('octaves must be a positive int')
        self._perlin_model.octaves = value

    @property
    def seed(self) -> int:
        return self._perlin_model.seed
    @seed.setter
    def seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError('seed must be an int')
        self._prng.seed(value)
        self._perlin_model.seed = value


    def _generate_noisemap(self, z: NumericType,
                           speed: Optional[NumericType] = None):
        if speed is None: speed = 1 / max(self._dim)
        y = 0
        for i in range(self._dim[0]):
            x = 0
            for j in range(self._dim[1]):
                value = self._perlin_model.get_value(x, y, z)
                x += speed

                self._grid[i, j] = value
            y += speed

    def _initialize_grid(self):
        self._grid = np.zeros(self._dim, dtype=float)


    def run(self,
        ax: Optional[Axes],
        z: Optional[NumericType] = None,
        speed: Optional[NumericType] = None, *,
        line_color: tuple[float, float, float] = (1, 1, .5608),
        cmap: str = 'Greys',
        dot_marker: str = 'o',
        animated: bool = False
    ) -> tuple[Axes, Optional[list[AxesImage, Line2D]]]:
        """
        Run Marching Squares on a Perlin noisemap generated with the current configuration.

        ## Parameters:
        ``ax``: a ``matplotlib.axes.Axes`` object to plot on.

        ``z``: the z-level to generate Perlin noise. If None is given, a uniform random\
            number between 0-1 and chosen instead.

        ``speed``: a number specifying how much to increment x and y when moving along\
            the plane levelled at z. A big increment will lead to a noisier noisemap.

        ``line_color``: the color of contour lines. This defaults to ``(1, 1, 0.5608)``

        ``cmap``: the colormap to render the noisemap. This defaults to ``'Greys'``.

            *Note: this only takes effect for grids of dimension higher than 20.*\\
            *For dimension 20 and below, noisemap is rendered as grid of points.*

        ``dot_marker``: matplotlib's point marker used for rendering the noisemap.

            *Note: this only takes effect for grids of dimension 20 and below. See above.*

        ``animated``: whether or not to plot the artists with animated artists.\\
        Animated objects are not drawn to canvas unless artist.set_visible(True) is\
            called. This is useful when you need to create animation with\
            ``matplotlib.animation.ArtistAnimation`` or similar means.\\
        If ``animated=True``, the function will return back a list of ``matplotlib.artist.Artists``\
            to be drawn on the given ``matplotlib.axes.Axes``.

        ## Returns
        A tuple of ``matplotlib.axes.Axes`` and optional list of ``matplotlib.artist.Artists``\
            if ``animated=True``.
        """
        if z is None: z = self._prng.random()
        self._generate_noisemap(z, speed)

        threshold = (self._grid.max() + self._grid.min()) / 2

        ax.set_xlim(0, self._dim[1] - 1)
        ax.set_ylim(self._dim[0] - 1, 0)

        if not animated and (self._dim[0] < 20 or self._dim[1] < 20):
            gmin, gmax = self._grid.min(), self._grid.max()
            color_mat = 1 - (self._grid - gmin) / (gmax - gmin)

            ax.get_figure().set_facecolor('0.7')
            for i, row in enumerate(color_mat):
                for j, val in enumerate(row):
                    ax.plot([j], [i], dot_marker, color=str(val), mec='0',
                            animated=animated)
        else:
            ax_img = ax.imshow(self._grid, cmap=cmap, animated=animated)

        lines = ax.plot(*_draw_contour(self._grid, threshold),
                        color=line_color, animated=animated)
        if animated:
            lines.append(ax_img)
            return ax, lines 
        else:
            return ax, None