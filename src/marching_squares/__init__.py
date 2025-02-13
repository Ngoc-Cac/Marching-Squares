from pynoise.noisemodule import NoiseModule
from typing import (
    TypeAlias,
    TypeVar,
    Union,
)

# Custom type
NoiseModuleType = TypeVar('NoiseModuleType', bound=NoiseModule)
NumericType: TypeAlias = Union[int, float]