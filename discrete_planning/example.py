from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

import base


class ConfigurationSpace:
    """
    Describes the configuration space of the problem

    The space is constrained to the XY plane [0, 1) and rotations [0, 2pi). The points are
    discretized according to the parameters given on construction.

    Continuous points are assigned to the nearest discrete point by absolute distance e.g. if
    n_x_points = 10, then x=0.22 will be mapped to x=0.2 for the purposes of configurations.
    """

    def __init__(self, n_x_points: int, n_y_points: int, n_psi_points: int) -> None:
        self._x_points = np.linspace(0.0, 1.0, n_x_points)
        self._y_points = np.linspace(0.0, 1.0, n_y_points)
        self._psi_points = np.linspace(0.0, 2 * np.pi, n_psi_points)

    def to_float(self, points: npt.NDArray[np.uint]) -> npt.NDArray[np.float_]:
        return np.array(
            [
                self._x_points[points[:, 0]],
                self._y_points[points[:, 1]],
                self.psi_to_float(points[:, 2]),
            ]
        ).transpose()

    def psi_to_float(self, psis: npt.NDArray[np.uint]) -> npt.NDArray[np.float_]:
        return self._psi_points[psis]


class InputSpace:
    """
    Describes the input space

    The input space is simple. It is a 2-D point (s, r) containing s E {-1, 0, +1} where s is the
    command to reverse, neutral, and forward, respectively, and r E {psi : (x, y, psi) E C} where r
    is an angle contained in the discrete configuration space.
    """

    def __init__(self, config_space: ConfigurationSpace) -> None:
        self._config_space = config_space

    def to_float(self, points: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
        return np.array(
            points[:, 0].astype(np.float_),
            self._config_space.psi_to_float(points[:, 1].astype(np.uint)),
        ).transpose()


class RigidBody2D:
    """
    Describes a simple 2D robot

    The robot may rotate freely about the z-axis, but may only move forward in the direction it is
    pointing. The robot may move up to one discrete point at a time e.g. (0.00, 0.00, 0) ->
    (0.01, 0.00, pi/4). Translation is first applied and then rotation.
    """

    Configuration: typing.TypeAlias = npt.NDArray[np.float_]
    Input: typing.TypeAlias = npt.NDArray[np.float_]

    def __init__(self, x: float, y: float, psi: float) -> None:
        self._config = np.array([x, y, psi])

    def inputs(self, config: Configuration) -> Input:
        return self._config

    def transition(self, config: Configuration, input: Input) -> Configuration:
        return config

    @property
    def configuration(self) -> Configuration:
        return self._config


class ExampleProblem:
    """
    Simple problem to showcase search algorithms

    The search space is bound to [0, 1) in R^2 and [0, 2pi) in S^1. The robot must reach
    (0.9, 0.9) at any rotation.

    The space is discretized to the 2nd decimal place for R^2 e.g. (0.18, o.45) and into 8
    directions for S^1: (0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi). This yields
    a total of ( 100 ^ 2 ) * ( 8 ) = 8,000,000 possible configurations.
    """

    def __init__(self) -> None:
        pass

    @property
    def goal_region(self) -> RigidBody2D.Configuration:
        return np.array([0, 0, 0])


if __name__ == "__main__":
    body = RigidBody2D(0, 0, 0)
    problem = ExampleProblem()
    base.BreadthFirstForwardSearchAlgorithm(
        body.configuration, problem.goal_region, body.inputs, body.transition
    )
