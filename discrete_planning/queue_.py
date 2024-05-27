from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt


class Queue:
    def __init__(self, number_of_dimensions: int, maximum_length: int) -> None:
        self._elements = np.zeros((maximum_length, number_of_dimensions))

    def put(self, elements: typing.Union[float, npt.NDArray[np.float_]]) -> None:
        np.append(self._elements, elements)

    def pop(self) -> npt.NDArray[np.float_]:
        element = self._elements[0, :]
        self._elements = self._elements[1:, :]
        return element

    def as_numpy(self) -> None:
        pass

    @staticmethod
    def from_numpy() -> None:
        pass
