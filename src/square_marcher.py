import random, sys

import numpy as np

from pynoise.noisemodule import Perlin

from marching_squares_algo import draw_contours


from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType
from typing import (
    Literal,
    Optional,
    TypeAlias,
    Union,
)

NumericType: TypeAlias = Union[int, float]


class SquareMarcher():
    __slots__ = (
        '_dim',
        '_grid',
        '_perlin_model',
        '_prng',
    )
    def __init__(self, dimension: tuple[int, int], seed: Optional[int] = None):
        if not isinstance(dimension, tuple):
            raise TypeError('Dimension must be a tuple of two ints.')
        elif len(dimension) != 2 or any((not isinstance(num, int) or num < 1) for num in dimension):
            raise ValueError('Number of rows and columns must be positive!')
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        elif not isinstance(seed, int):
            raise TypeError('Random seed must be an int')

        self._dim = dimension
        
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
        thresholding_method: Literal['midpoint', 'average'] = 'midpoint',
        line_color: ColorType = (1, 1, .5608),
        cmap: str | Colormap = 'Greys',
        dot_marker: str = 'o',
        animated: bool = False
    ) -> tuple[Axes, Optional[list[AxesImage, Line2D]]]:
        """
        Run Marching Squares on a Perlin noisemap generated with the current configuration.

        ## Parameters:
        ``ax: matplotlib.axes.Axes | None``
            
            a ``matplotlib.axes.Axes`` object to plot on.
        
        ``z: int | float``
        
            the z-level to generate Perlin noise. If None is given, a uniform random\
            number between 0-1 and chosen instead.

        ``thresholding_method: Literal['midpoint', 'average']``

            The method for choosing a threshold.\\
            ``'midpoint'`` method takes the midpoint between the minimum\
                and the maximum value in the grid.\\
            ``'average'`` method takes the arithmetic mean across all values.

        ``speed: int | float``

            a number specifying how much to increment x and y when moving along\
            the plane levelled at z. A big increment will lead to a noisier noisemap.

        ``line_color: ColorType``
        
            the color of contour lines, should be a matplotlib's [color format](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def).\
            This defaults to ``(1, 1, 0.5608)``

        ``cmap: str | Colormap``
        
            the colormap to render the noisemap, should be a matplotlib's colormap.\
                This defaults to ``'Greys'``.\\
            *Note: this only takes effect for grids of dimension higher than 20.*
            *For dimension 20 and below, noisemap is rendered as grid of points.*

        ``dot_marker: str``
        
            matplotlib's point marker used for rendering the noisemap.\\
            *Note: this only takes effect for grids of dimension 20 and below. See above.*

        ``animated: bool``
            
            whether or not to plot the artists with animated artists.\
                This defualts to ``False``.\\
            Animated objects are not drawn to canvas unless artist.set_visible(True) is\
                called. This is useful when you need to create animation with\
            ``matplotlib.animation.ArtistAnimation`` or similar means.\\
            If ``animated=True``, the function will return back a list of\
                ``matplotlib.artist.Artists`` to be drawn on the given ``matplotlib.axes.Axes``.

        ## Returns
        A tuple of ``matplotlib.axes.Axes`` and optional list of ``matplotlib.artist.Artists``\
            if ``animated=True``.
        """
        if z is None: z = self._prng.random()
        self._generate_noisemap(z, speed)

        if thresholding_method == 'midpoint':
            threshold = (self._grid.max() + self._grid.min()) / 2
        elif thresholding_method == 'average':
            threshold = self._grid.mean()

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

        lines = ax.plot(*draw_contours(self._grid, threshold),
                        color=line_color, animated=animated)
        if animated:
            lines.append(ax_img)
            return ax, lines 
        else:
            return ax, None