import random, sys

import numpy as np
try: import opensimplex
except ImportError: opensimplex = None

from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from pynoise.noisemodule import (
    Perlin,
    Voronoi
)

from marching_squares.algo import draw_contours

# This is typing stuff, not involved in the actual code
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.typing import ColorType
from marching_squares import NumericType
from typing import (
    Literal,
    Optional,
)

class SquareMarcher():
    """
    This is a class wrapper for the Marching Squares algorithm running on a randomly generated noisemap\
        using 3D Noise Generator.
    """
    __slots__ = (
        '_dim',
        '_grid',
        '_lerping',
        '_qth',
        '_thres_method',
    )
    def __init__(self,
                 dimension: tuple[int, int],
                 threshold_method: Literal['midpoint', 'average'] | int = 'midpoint',
                 lerping: bool = False,
    ):
        """
        Initialize SquareMarcher object. This is a Marching Squares algorithm that runs\
            on a noisemap generated with 3d Gradient Noise Generator.

        ## Parameters:
        ``dimension: tuple[int, int]``

            the dimension of the noisemap in pixels.

        ``seed: Optional[int]``

            the seed for any prng used by the SquareMarcher, including the Perlin Noise generator.\
                This defaults to None and a random seed will be generated

        ``threshold_method: Literal['midpoint', 'average'] | int``

            the thresholding method to use on the noisemap. This defaults to ``'midpoint'`` and \
                the mid-range value betwen the max and min value in the noise map will use.\\
            If ``'average'`` is specified instead, the arithmetic mean across all values will be used.\\
            An integer ``q`` in the range [0, 100] can be specified as well, in which case,\
                the q-th percentile will be used as the threshold.

        ``lerping: bool``

            whether or not to use linear interpolation to find the endpoint of the contour lines.\
                Using linear interpolation will lead to smoother contour lines along regions. This\
                defaults to False.

        ## Raises
        Various TypeError and ValueError if you didn't read the docstring carefully.
        """
        if not isinstance(dimension, tuple):
            raise TypeError('Dimension must be a tuple of two ints.')
        elif len(dimension) != 2 or any((not isinstance(num, int) or num < 1) for num in dimension):
            raise ValueError("'dimension' must be a tuple of two positive integers")
        if isinstance(threshold_method, int):
            if threshold_method > 100 or threshold_method < 0:
                raise ValueError('Percentage of percentile must be in the range [0, 100]')
            self._qth = threshold_method
            threshold_method = 'percentile'
        elif not threshold_method in ['midpoint', 'average']:
            raise ValueError(f"Expected 'threshold_method' to be 'midpoint' or 'average' or a percentile percentage: {threshold_method}")
        if not isinstance(lerping, bool):
            raise TypeError("'lerping' must be a bool")

        self._dim = dimension
        self._thres_method: Literal['midpoint', 'average', 'percentile'] = threshold_method
        self._lerping = lerping
        
        self._initialize_grid()

    
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
    def grid(self) -> NDArray[np.float64]:
        return self._grid
    
    @property
    def lerping(self) -> bool:
        return self._lerping
    @lerping.setter
    def lerping(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("'lerping' must be a bool")
        self._lerping = value

    @property
    def threshold_method(self) -> Literal['midpoint', 'average', 'percentile']:
        return self._thres_method
    @threshold_method.setter
    def threshold_method(self, value: Literal['midpoint', 'average'] | int):
        if isinstance(value, int):
            if value > 100 or value < 0:
                raise ValueError('Percentage of percentile must be in the range [0, 100]')
            self._qth = value
            value = 'percentile'
        elif not value in ['midpoint', 'average']:
            raise ValueError(f"Expected 'threshold_method' to be 'midpoint' or 'average' or a percentile percentage: {value}")
        self._thres_method = value


    def generate_scalar_field(self, *args):
        raise NotImplementedError('generate_scalar_field() not implemented')

    def _initialize_grid(self):
        self._grid = np.zeros(self._dim, dtype=np.float64)


    def run(self,
        ax: Axes,
        *generate_args,
        line_color: ColorType = (1, 1, .5608),
        cmap: str | Colormap = 'Greys',
        dot_marker: str = 'o',
        animated: bool = False
    ) -> tuple[Axes, Optional[list[AxesImage, Line2D]]]:
        """
        Run Marching Squares on a Perlin noisemap generated with the current configuration.

        ## Parameters:
        ``ax: matplotlib.axes.Axes``
            
            a ``matplotlib.axes.Axes`` object to plot on.
        
        ``z: int | float``
        
            the z-level to generate Perlin noise. If None is given, a uniform random\
            number between 0-1 and chosen instead.

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
        if len(generate_args):
            self.generate_scalar_field(*generate_args)

        if self._thres_method == 'midpoint':
            threshold = (self._grid.max() + self._grid.min()) / 2
        elif self._thres_method == 'average':
            threshold = self._grid.mean()
        elif self._thres_method == 'percentile':
            threshold = np.percentile(self._grid, self._qth)

        contours = draw_contours(self._grid, threshold, lerp=self._lerping)


        ax.set_xlim(0, self._dim[1] - 1)
        ax.set_ylim(self._dim[0] - 1, 0)

        if not animated and (self._dim[0] < 20 or self._dim[1] < 20):
            gmin, gmax = self._grid.min(), self._grid.max()
            color_mat = 1 - (self._grid - gmin) / (gmax - gmin)

            ax.get_figure().set_facecolor('0.7')
            for i, row in enumerate(color_mat):
                for j, val in enumerate(row):
                    ax.add_line(Line2D([j], [i], color=str(val), marker=dot_marker, markeredgecolor='0'))
        else:
            ax_img = AxesImage(ax, cmap=cmap, extent=(-.5, self._dim[1] - .5,
                                                      self._dim[0] -.5, -.5))
            ax_img.set_data(self._grid)
            ax.add_image(ax_img)
        for i in range(0, len(contours), 2):
            ax.add_line(Line2D(contours[i], contours[i + 1], color=line_color))

        if animated:
            lines = list(ax.lines)
            lines.append(ax_img)
            return ax, lines
        else:
            ax.get_figure().canvas.draw()
            return ax, None

class PerlinMarcher(SquareMarcher):
    """
    This class uses 3D Perlin Noise to generate a noisemap and then run Marching Squares on the noisemap.
    """
    __slots__ = (
        '_noise_module',
        '_prng'
    )
    def __init__(self,
                 dimension: tuple[int, int],
                 seed: Optional[int] = None,
                 threshold_method: Literal['midpoint', 'average'] | int = 'midpoint',
                 lerping: bool = False,
                 frequency: NumericType = 1,
                 lacunarity: NumericType = 2,
                 octaves: int = 2,
                 persistence: NumericType = 0.25
    ):
        """
        Initialize PerlinMarcher object. This is a Marching Squares algorithm that runs\
            on a noisemap generated with 3D Perlin Noise.\\
        For information on modifying the Noise Generator, see the pynoise's documention:\
            https://pynoise.readthedocs.io/en/latest/tutorial4.html

        ## Parameters:
        ``dimension: tuple[int, int]``

            the dimension of the noisemap in pixels.

        ``seed: Optional[int]``

            the seed for any prng used by the SquareMarcher, including the Perlin Noise generator.\
                This defaults to None and a random seed will be generated

        ``threshold_method: Literal['midpoint', 'average'] | int``

            the thresholding method to use on the noisemap. This defaults to ``'midpoint'`` and \
                the mid-range value betwen the max and min value in the noise map will use.\\
            If ``'average'`` is specified instead, the arithmetic mean across all values will be used.\\
            An integer ``q`` in the range [0, 100] can be specified as well, in which case,\
                the q-th percentile will be used as the threshold.

        ``lerping: bool``

            whether or not to use linear interpolation to find the endpoint of the contour lines.\
                Using linear interpolation will lead to smoother contour lines along regions. This\
                defaults to False.

        ``frequency: NumericType``

            From the docs: `"The frequency determines how many changes along a unit length.\
                Increasing the frequency adds to the number of interesting features\
                in the noise, which also making each feature smaller\
                as more are packed into a given area."`

        ``lacunarity: NumericType``

            lacunarity affects the smoothness of the noisemap. For higher values, the noisemap\
                generally have a finer, coarser grain texture, with small features sticking out.

        ``octaves: int``

            this is how many times the noise are summed up. Like lacunarity, this gives rougher\
                texture for higher values. However, this parameter is also dependent on ``persistence``.

        ``persistence: NumericType``

            Persistence can be used to control the effect of ``octaves``. With values less than 1,\
                the persistence mitigate effects of passes from later octaves.\\
            From the docs: `"Persistence determines how quickly the amplitudes decrease between\
                each octave..."`

        ## Raises
        Various TypeError and ValueError if you didn't read the docstring carefully.
        """
        super().__init__(dimension, threshold_method, lerping)

        if seed is None:
            seed = random.randint(0, sys.maxsize)
        elif not isinstance(seed, int):
            raise TypeError('Random seed must be an int')
        if not isinstance(frequency, NumericType):
            raise TypeError('frequency must be an int or float')
        if not isinstance(lacunarity, NumericType):
            raise TypeError('lacunarity must be an int or float')
        if not isinstance(octaves, int):
            raise TypeError('octaves must be a positive int')
        elif octaves < 1:
            raise ValueError('octaves must be a positive int')
        if not isinstance(persistence, NumericType):
            raise TypeError('persistence must be an int or float')

        self._noise_module = Perlin(frequency, lacunarity, octaves, persistence, seed)
        self._prng = random.Random(seed)

    
    @property
    def frequency(self) -> NumericType:
        """
        From pynoise's documentation:\
        
            `"The frequency determines how many changes along a unit length.\
            Increasing the frequency adds to the number of interesting features\
            in the noise, which also making each feature smaller\
            as more are packed into a given area."`
        """
        return self._noise_module.frequency
    @frequency.setter
    def frequency(self, value: NumericType):
        if not isinstance(value, NumericType):
            raise TypeError('frequency must be an int or float')
        self._noise_module.frequency = value

    @property
    def lacunarity(self) -> NumericType:
        """
        lacunarity affects the smoothness of the noisemap. For higher values, the noisemap\
                generally have a finer, coarser grain texture, with small features sticking out.
        """
        return self._noise_module.lacunarity
    @lacunarity.setter
    def lacunarity(self, value: NumericType):
        if not isinstance(value, NumericType):
            raise TypeError('lacunarity must be an int or float')
        self._noise_module.lacunarity = value

    @property
    def octaves(self) -> int:
        """
        this is how many times the noise are summed up. Like lacunarity, this gives rougher\
                texture for higher values. However, this parameter is also dependent on ``persistence``.
        """
        return self._noise_module.octaves
    @octaves.setter
    def octaves(self, value: int):
        if not isinstance(value, int):
            raise TypeError('octaves must be a positive int')
        elif value < 1:
            raise ValueError('octaves must be a positive int')
        self._noise_module.octaves = value

    @property
    def persistence(self) -> NumericType:
        """
        Persistence can be used to control the effect of ``octaves``. With values less than 1,\
                the persistence mitigate effects of passes from later octaves.\\
        From the docs: `"Persistence determines how quickly the amplitudes decrease between\
            each octave..."`
        """
        return self._noise_module.persistence
    @persistence.setter
    def persistence(self, value: NumericType):
        if not isinstance(value, NumericType):
            raise TypeError('persistence must be an int or float')
        self._noise_module.persistence = value

    @property
    def seed(self) -> int:
        return self._noise_module.seed
    @seed.setter
    def seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError('seed must be an int')
        self._noise_module.seed = value
        self._prng.seed(value)

    def generate_scalar_field(self, z: Optional[NumericType] = None,
                               speed: Optional[NumericType] = None):
        if z is None: z = self._prng.random() * 100
        if speed is None: speed = 1 / max(self._dim)
        for i in range(self._dim[0]):
            for j in range(self._dim[1]):
                value = self._noise_module.get_value(i * speed, j * speed, z)
                self._grid[i, j] = value

class VoronoiMarcher(SquareMarcher):
    """
    This is a class wrapper for the Marching Squares algorithm running on a randomly generated noisemap\
        using 3D Voronoi Noise.
    """
    __slots__ = (
        '_noise_module',
        '_prng'
    )
    def __init__(self,
                 dimension: tuple[int, int],
                 seed: Optional[int] = None,
                 threshold_method: Literal['midpoint', 'average'] | int = 'midpoint',
                 lerping: bool = False,
                 displacement: NumericType = 1,
                 enable_distance: bool = False,
                 frequency: NumericType = 1
    ):
        """
        Initialize VoronoiMarcher object. This is a Marching Squares algorithm that runs\
            on a noisemap generated with 3D Voronoi Noise.

        ## Parameters:
        ``dimension: tuple[int, int]``

            the dimension of the noisemap in pixels.

        ``seed: Optional[int]``

            the seed for any prng used by the SquareMarcher, including the Perlin Noise generator.\
                This defaults to None and a random seed will be generated

        ``threshold_method: Literal['midpoint', 'average'] | int``

            the thresholding method to use on the noisemap. This defaults to ``'midpoint'`` and \
                the mid-range value betwen the max and min value in the noise map will use.\\
            If ``'average'`` is specified instead, the arithmetic mean across all values will be used.\\
            An integer ``q`` in the range [0, 100] can be specified as well, in which case,\
                the q-th percentile will be used as the threshold.

        ``lerping: bool``

            whether or not to use linear interpolation to find the endpoint of the contour lines.\
                Using linear interpolation will lead to smoother contour lines along regions. This\
                defaults to False.

        ``displacement: NumericType``

            this determines how large or small the range of values each seed can take on.\
                Every seed is assigned a random value +/- ``displacement``

        ``enable_distance: bool``

            whether or not the information on the distance to the nearest seed is included in\
                each point. If ``True``, each point in the Voronoi cells will increase in value\
                the further the distance to the nearest seed gets.

        ``frequency: NumericType``

            this changes the distance between each seed placed in the grid. The higher the values\
                the closer the seeds are placed.

        ## Raises
        Various TypeError and ValueError if you didn't read the docstring carefully.
        """
        super().__init__(dimension, threshold_method, lerping)

        if seed is None:
            seed = random.randint(0, sys.maxsize)
        elif not isinstance(seed, int):
            raise TypeError('Random seed must be an int')
        if not isinstance(frequency, NumericType):
            raise TypeError('frequency must be an int or float')
        if not isinstance(enable_distance, bool):
            raise TypeError('enable_distance must be a bool')
        if not isinstance(displacement, NumericType):
            raise TypeError('displacement must be an int or float')

        self._noise_module = Voronoi(displacement, enable_distance, frequency)
        self._prng = random.Random(seed)

    
    @property
    def frequency(self) -> NumericType:
        """
        this changes the distance between each seed placed in the grid. The higher the values\
                the closer the seeds are placed.
        """
        return self._noise_module.frequency
    @frequency.setter
    def frequency(self, value: NumericType):
        if not isinstance(value, NumericType):
            raise TypeError('frequency must be an int or float')
        self._noise_module.frequency = value
    
    @property
    def enable_distance(self) -> bool:
        """
        whether or not the information on the distance to the nearest seed is included in\
            each point. If ``True``, each point in the Voronoi cells will increase in value\
            the further the distance to the nearest seed gets.
        """
        return self._noise_module.enable_distance
    @enable_distance.setter
    def enable_distance(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError('enable_distance must be a bool')
        self._noise_module.enable_distance = value

    @property
    def displacement(self) -> NumericType:
        """
        this determines how large or small the range of values each seed can take on.\
            Every seed is assigned a random value +/- ``displacement``
        """
        return self._noise_module.displacement
    @displacement.setter
    def displacement(self, value: NumericType):
        if not isinstance(value, NumericType):
            raise TypeError('displacement must be an int or float')
        self._noise_module.displacement = value

    @property
    def seed(self) -> int:
        return self._noise_module.seed
    @seed.setter
    def seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError('seed must be an int')
        self._noise_module.seed = value
        self._prng.seed(value)

    def generate_scalar_field(self, z: Optional[NumericType] = None,
                               speed: Optional[NumericType] = None):
        if z is None: z = self._prng.random() * 100
        if speed is None: speed = 1 / max(self._dim)
        for i in range(self._dim[0]):
            for j in range(self._dim[1]):
                value = self._noise_module.get_value(i * speed, j * speed, z)
                self._grid[i, j] = value

class OpenSimplexMarcher(SquareMarcher):
    __slots__ = (
        '_prng',
        '_xs',
        '_ys'
    )
    def __init__(self,
                 dimension: tuple[int, int],
                 seed = None,
                 threshold_method: Literal['midpoint', 'average'] | int = 'midpoint',
                 lerping: bool = False):
        if opensimplex is None:
            raise ImportError('Could not import opensimplex library, this could be because the library has not beend installed')
        super().__init__(dimension, threshold_method, lerping)
        
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        elif not isinstance(seed, int):
            raise TypeError('Random seed must be an int')
        opensimplex.seed(seed)
        self._prng = random.Random(seed)

    @property
    def seed(self) -> int:
        return opensimplex.get_seed()
    @seed.setter
    def seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError('seed must be an int')
        opensimplex.seed(value)


    def generate_scalar_field(self, z: Optional[NumericType] = None,
                               speed: Optional[NumericType] = None):
        if z is None: z = self._prng.random() * 100
        if speed is None: speed = 1 / max(self._dim)

        xs = self._xs * speed
        ys = self._ys * speed
        zs = np.array([z], dtype=np.float64)

        self._grid = opensimplex.noise3array(xs, ys, zs)[0]

    def _initialize_grid(self):
        super()._initialize_grid()
        self._xs = np.arange(self._dim[1], dtype=np.float64)
        self._ys = np.arange(self._dim[0], dtype=np.float64)
