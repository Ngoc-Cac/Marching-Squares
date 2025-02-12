import numpy as np

# This is typing stuff, not involved in the actual code
from marching_squares import NumericType


# Note for the whole file:
# The origin is at the TOP LEFT corner, just like in any graphics drawing
def edge_lerp(t: NumericType, v_1: NumericType, v_2: NumericType) -> float:
    """
    Linear interpolation between two corners' values in Marching Square according to\
        a threshold ``t``.
    Think of this as drawing a line between ``(v_1, 0)`` and ``(v_2, 1)`` and then\
        evaluate that line at the value ``t``. This gives the offest on the edge.

    ## Parameters:
    ``t: NumericType``

        the parameter at which to evaluate the interpolation.

    ``v_1: NumericType``

        the value of first corner. This should be the corner where there is no\
            offset from the the top left corner.

    ``v_2: NumericType``

        the value of the second corner. This is the corner with the offset.

    ## Return
    A value representing the offset on the edge.

    ## Raises
    ZeroDivisionError if ``v_1 == v_2``
    """
    return (t - v_1) / (v_2 - v_1)

def _rough_contouring(coordinates: tuple[int, int],
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
    
def _smooth_contouring(coordinates: tuple[int, int],
                       tleft: bool, tright: bool,
                       bright: bool, bleft: bool,
                       grid: np.ndarray[NumericType],
                       threshold: NumericType)\
    -> tuple[tuple[float, float, float, float]]:
    """
    Drawing contour lines based on the four corners' states with linear interpolation\
        using their values to find the intersection.
    """
    case_no = tleft * 8 + tright * 4 + bright * 2 + bleft
    x, y = coordinates
    if case_no == 0 or case_no == 15:
        return (), ()
    elif case_no == 1 or case_no == 14:
        return (x + edge_lerp(threshold, grid[y + 1, x], grid[y + 1, x + 1]), x,
                y + 1, y + edge_lerp(threshold, grid[y, x], grid[y + 1, x])),\
               ()
    elif case_no == 2 or case_no == 13:
        return (x + 1, x + edge_lerp(threshold, grid[y + 1, x], grid[y + 1, x + 1]),
                y + edge_lerp(threshold, grid[y, x + 1], grid[y + 1, x + 1]), y + 1),\
               ()
    elif case_no == 3 or case_no == 12:
        return (x + 1, x,
                y + edge_lerp(threshold, grid[y, x + 1], grid[y + 1, x + 1]),
                y + edge_lerp(threshold, grid[y, x], grid[y + 1, x])),\
               ()
    elif case_no == 4 or case_no == 11:
        return (x + edge_lerp(threshold, grid[y, x], grid[y, x + 1]), x + 1,
                y, y + edge_lerp(threshold, grid[y, x + 1], grid[y + 1, x + 1])),\
               ()
    elif case_no == 6 or case_no == 9:
        return (x + edge_lerp(threshold, grid[y, x], grid[y, x + 1]),
                x + edge_lerp(threshold, grid[y + 1, x], grid[y + 1, x + 1]),
                y, y + 1),\
               ()
    elif case_no == 7 or case_no == 8:
        return (x, x + edge_lerp(threshold, grid[y, x], grid[y, x + 1]),
                y + edge_lerp(threshold, grid[y, x], grid[y + 1, x]), y),\
               ()
    elif case_no == 5:
        return (x + 1, x + edge_lerp(threshold, grid[y + 1, x], grid[y + 1, x + 1]),
                y + edge_lerp(threshold, grid[y, x + 1], grid[y + 1, x + 1]), y + 1),\
               (x, x + edge_lerp(threshold, grid[y, x], grid[y, x + 1]),
                y + edge_lerp(threshold, grid[y, x], grid[y + 1, x]), y)
    elif case_no == 10:
        return (x + edge_lerp(threshold, grid[y, x], grid[y, x + 1]), x + 1,
                y, y + edge_lerp(threshold, grid[y, x + 1], grid[y + 1, x + 1])),\
               (x + edge_lerp(threshold, grid[y + 1, x], grid[y + 1, x + 1]), x,
                y + 1, y + edge_lerp(threshold, grid[y, x], grid[y + 1, x]))
    

def draw_contours(grid: np.ndarray[NumericType],
                  threshold: NumericType,
                  lerp: bool = False)\
    -> list[tuple[float, float]]:
    """
    The Marching Squares algorithm in its entirety.\\
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
    ``grid: numpy.ndarray[int | float]``
        
        a `numpy.ndarray` of shape `(rows, cols)`. Each element should be a scalar

    ``threshold: int | float``
    
        a number representing the threshold to seperate the scalars. The scalars\
        will be divided into two groups using the threshold: one being greater than the threshold\
        and one being less than or equal to the threshold.

    ``lerp: bool``

        whether to use linear interpolation to determine the coordinates of the edge.\
            If linear interpolation is not used, the coordinates of the edge is at the\
            midpoint between two corners. This defaults to False.

    ## Returns
    A list of tuples of floats. For more information, see above.
    """
    lines = []
    rows, cols = grid.shape
    mask = grid > threshold
    contouring = (lambda *args: _smooth_contouring(*args, grid, threshold)) if lerp else _rough_contouring
    for i in range(rows - 1):
        for j in range(cols - 1):
            line1, line2 = contouring((j, i),
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