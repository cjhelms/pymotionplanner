from __future__ import annotations

import argparse
import dataclasses
import random

import matplotlib.pyplot as plt
import numpy as np

import discrete


class HolonomicRobot2D:
    def __init__(self, do_shuffle_inputs: bool) -> None:
        self._do_shuffle_inputs = do_shuffle_inputs

    def transition(self, state: State2D, input: HolonomicInput2D) -> State2D:
        return State2D(state.x + input.dx, state.y + input.dy)

    def get_inputs(self, state: State2D) -> list[HolonomicInput2D]:
        inputs = [
            HolonomicInput2D.GoNorth(),
            HolonomicInput2D.GoNortheast(),
            HolonomicInput2D.GoEast(),
            HolonomicInput2D.GoSoutheast(),
            HolonomicInput2D.GoSouth(),
            HolonomicInput2D.GoSouthwest(),
            HolonomicInput2D.GoWest(),
            HolonomicInput2D.GoNorthwest(),
        ]
        if self._do_shuffle_inputs:
            random.shuffle(inputs)
        return inputs


@dataclasses.dataclass
class State2D:
    x: int
    y: int


def distance_between(a: State2D, b: State2D) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


@dataclasses.dataclass
class HolonomicInput2D:
    dx: int
    dy: int

    @staticmethod
    def GoNorth() -> HolonomicInput2D:
        return HolonomicInput2D(1, 0)

    @staticmethod
    def GoNortheast() -> HolonomicInput2D:
        return HolonomicInput2D(1, 1)

    @staticmethod
    def GoEast() -> HolonomicInput2D:
        return HolonomicInput2D(0, 1)

    @staticmethod
    def GoSoutheast() -> HolonomicInput2D:
        return HolonomicInput2D(-1, 1)

    @staticmethod
    def GoSouth() -> HolonomicInput2D:
        return HolonomicInput2D(-1, 0)

    @staticmethod
    def GoSouthwest() -> HolonomicInput2D:
        return HolonomicInput2D(-1, -1)

    @staticmethod
    def GoWest() -> HolonomicInput2D:
        return HolonomicInput2D(0, -1)

    @staticmethod
    def GoNorthwest() -> HolonomicInput2D:
        return HolonomicInput2D(1, -1)


class RectangularOccupancyGrid2D:
    def __init__(self, northwest_corner: State2D, obstacles: list[Obstacle2D]) -> None:
        self._lookup_table = np.zeros((northwest_corner.x, northwest_corner.y))
        for o in obstacles:
            half_spaces = [
                (o.vertices[i], o.vertices[(i + 1) % len(o.vertices)])
                for i in range(len(o.vertices))
            ]
            for i in range(self._lookup_table.shape[0]):
                for j in range(self._lookup_table.shape[1]):

                    def point_i_j_is_left_of(
                        half_space: tuple[State2D, State2D]
                    ) -> bool:
                        hs_origin = np.array([half_space[0].x, half_space[0].y])
                        hs_vector = (
                            np.array([half_space[1].x, half_space[1].y]) - hs_origin
                        )
                        EPSILON = 0.01
                        return bool(
                            np.cross(hs_vector, np.array([i, j]) - hs_origin) < EPSILON
                        )

                    if all(point_i_j_is_left_of(hs) for hs in half_spaces):
                        self._lookup_table[i, j] = 1

    def is_occupied(self, state: State2D) -> bool:
        if (
            state.x >= self._lookup_table.shape[0]
            or state.x < 0
            or state.y >= self._lookup_table.shape[1]
            or state.y < 0
        ):
            return True
        return bool(self._lookup_table[state.x][state.y])

    @property
    def total_spaces(self) -> int:
        return self._lookup_table.size


@dataclasses.dataclass
class Obstacle2D:
    vertices: list[State2D]


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algorithm",
        required=True,
        choices=["breadth-first", "depth-first", "dijkstra", "astar"],
        help="algorithm to be used for planning",
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        default=False,
        help="Shuffle inputs randomly before putting onto queue",
    )
    return parser.parse_args()


def make_settings(do_shuffle_inputs: bool) -> Settings:
    initial_state = State2D(1, 9)
    goal_state = State2D(18, 1)
    world_northwest_corner = State2D(30, 30)
    obstacles = [
        Obstacle2D([State2D(12, 0), State2D(12, 15), State2D(15, 15), State2D(15, 0)]),
        Obstacle2D(
            [State2D(15, 12), State2D(15, 15), State2D(28, 15), State2D(28, 12)]
        ),
        Obstacle2D(
            [State2D(15, 15), State2D(15, 25), State2D(18, 25), State2D(18, 15)]
        ),
        Obstacle2D(
            [State2D(23, 18), State2D(23, 30), State2D(24, 30), State2D(24, 18)]
        ),
        Obstacle2D([State2D(3, 20), State2D(3, 21), State2D(13, 21), State2D(13, 20)]),
    ]
    return Settings(
        do_shuffle_inputs, initial_state, goal_state, world_northwest_corner, obstacles
    )


@dataclasses.dataclass
class Settings:
    do_shuffle_inputs: bool
    initial_state: State2D
    goal_state: State2D
    world_northwest_corner: State2D
    obstacles: list[Obstacle2D]

    def make_tuple(
        self,
    ) -> tuple[HolonomicRobot2D, State2D, State2D, RectangularOccupancyGrid2D]:
        return (
            HolonomicRobot2D(self.do_shuffle_inputs),
            self.initial_state,
            self.goal_state,
            RectangularOccupancyGrid2D(self.world_northwest_corner, self.obstacles),
        )


def make_motion_planner(algorithm: str) -> discrete.ForwardSearchAlgorithm:
    if algorithm == "breadth-first":
        return discrete.BreadthFirstMotionPlanner(*settings.make_tuple())
    elif algorithm == "depth-first":
        return discrete.DepthFirstMotionPlanner(*settings.make_tuple())
    elif algorithm == "dijkstra":
        return discrete.DijkstraMotionPlanner(distance_between, *settings.make_tuple())
    elif algorithm == "astar":
        return discrete.AStarMotionPlanner(
            lambda state: distance_between(settings.goal_state, state),
            distance_between,
            *settings.make_tuple(),
        )
    print(f"Unrecognized planner: {algorithm}")
    exit(1)


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    settings = make_settings(arguments.random)
    motion_planner = make_motion_planner(arguments.algorithm)
    motion_plan, visited_nodes = motion_planner.search()
    if motion_plan is None:
        print("No plan found!")
        exit(1)
    for i, o in enumerate(settings.obstacles):
        pairs = [
            (o.vertices[i], o.vertices[(i + 1) % len(o.vertices)])
            for i in range(len(o.vertices))
        ]
        for j, p in enumerate(pairs):
            v0, v1 = p
            (line,) = plt.plot([v0.y, v1.y], [v0.x, v1.x], "-k", linewidth=2)
            if i == 0 and j == 0:
                line.set_label("Obstacle boundary")
    plt.plot(
        motion_plan[0].state.y,
        motion_plan[0].state.x,
        "bx",
        markersize=10,
        label="Initial state",
    )
    plt.plot(
        motion_plan[-1].state.y,
        motion_plan[-1].state.x,
        "bo",
        markersize=10,
        label=f"Final (goal) state, metadata={motion_plan[-1].metadata}",
    )
    plt.plot(
        [node.state.y for node in motion_plan],
        [node.state.x for node in motion_plan],
        "-b",
        label="Motion plan",
    )
    plt.scatter(
        [node.state.y for node in visited_nodes],
        [node.state.x for node in visited_nodes],
        c="r",
        marker="x",
        label="Encountered state",
    )
    plt.xlim([-1, settings.world_northwest_corner.y])
    plt.ylim([-1, settings.world_northwest_corner.x])
    plt.grid()
    plt.legend()
    plt.show()
