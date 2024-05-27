from __future__ import annotations

import argparse
import dataclasses

import matplotlib.pyplot as plt
import numpy as np

import discrete


@dataclasses.dataclass
class Input:
    dx: int
    dy: int

    @staticmethod
    def GoNorth() -> Input:
        return Input(1, 0)

    @staticmethod
    def GoNortheast() -> Input:
        return Input(1, 1)

    @staticmethod
    def GoEast() -> Input:
        return Input(0, 1)

    @staticmethod
    def GoSoutheast() -> Input:
        return Input(-1, 1)

    @staticmethod
    def GoSouth() -> Input:
        return Input(-1, 0)

    @staticmethod
    def GoSouthwest() -> Input:
        return Input(-1, -1)

    @staticmethod
    def GoWest() -> Input:
        return Input(0, -1)

    @staticmethod
    def GoNorthwest() -> Input:
        return Input(1, -1)


@dataclasses.dataclass
class HolonomicState2D:
    x: int
    y: int

    def transition(self, input: Input) -> HolonomicState2D:
        return HolonomicState2D(self.x + input.dx, self.y + input.dy)

    @property
    def inputs(self) -> list[Input]:
        return [
            Input.GoNorth(),
            Input.GoNortheast(),
            Input.GoEast(),
            Input.GoSoutheast(),
            Input.GoSouth(),
            Input.GoSouthwest(),
            Input.GoWest(),
            Input.GoNorthwest(),
        ]


def distance_to(a: HolonomicState2D, b: HolonomicState2D) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


@dataclasses.dataclass
class Obstacle:
    vertices: list[HolonomicState2D]


class RectangularOccupancyGrid:
    def __init__(
        self,
        northwest_corner: HolonomicState2D,
        obstacles: list[Obstacle],
    ) -> None:
        self._lookup_table = np.zeros((northwest_corner.x, northwest_corner.y))
        for o in obstacles:
            half_spaces = [
                (o.vertices[i], o.vertices[(i + 1) % len(o.vertices)])
                for i in range(len(o.vertices))
            ]
            for i in range(self._lookup_table.shape[0]):
                for j in range(self._lookup_table.shape[1]):

                    def point_i_j_is_left_of(
                        half_space: tuple[HolonomicState2D, HolonomicState2D]
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

    def is_occupied(self, state: HolonomicState2D) -> bool:
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


if __name__ == "__main__":
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
    arguments = parser.parse_args()
    initial_state = HolonomicState2D(1, 9)
    goal_state = HolonomicState2D(18, 1)
    northwest_corner = HolonomicState2D(30, 30)
    obstacles = [
        Obstacle(
            [
                HolonomicState2D(12, 0),
                HolonomicState2D(12, 15),
                HolonomicState2D(15, 15),
                HolonomicState2D(15, 0),
            ]
        ),
        Obstacle(
            [
                HolonomicState2D(15, 12),
                HolonomicState2D(15, 15),
                HolonomicState2D(28, 15),
                HolonomicState2D(28, 12),
            ]
        ),
        Obstacle(
            [
                HolonomicState2D(15, 15),
                HolonomicState2D(15, 25),
                HolonomicState2D(18, 25),
                HolonomicState2D(18, 15),
            ]
        ),
        Obstacle(
            [
                HolonomicState2D(23, 18),
                HolonomicState2D(23, 30),
                HolonomicState2D(24, 30),
                HolonomicState2D(24, 18),
            ]
        ),
        Obstacle(
            [
                HolonomicState2D(3, 20),
                HolonomicState2D(3, 21),
                HolonomicState2D(13, 21),
                HolonomicState2D(13, 20),
            ]
        ),
    ]
    occupancy_grid = RectangularOccupancyGrid(northwest_corner, obstacles)
    if arguments.algorithm == "breadth-first":
        motion_planner = discrete.BreadthFirstMotionPlanner(
            initial_state, goal_state, occupancy_grid, arguments.random
        )
    elif arguments.algorithm == "depth-first":
        motion_planner = discrete.DepthFirstMotionPlanner(
            initial_state, goal_state, occupancy_grid, arguments.random
        )
    elif arguments.algorithm == "dijkstra":
        motion_planner = discrete.DijkstraMotionPlanner(
            distance_to, initial_state, goal_state, occupancy_grid, arguments.random
        )
    elif arguments.algorithm == "astar":
        motion_planner = discrete.AStarMotionPlanner(
            distance_to, initial_state, goal_state, occupancy_grid, arguments.random
        )
    else:
        print(f"Unrecognized planner: {arguments.algorithm}")
        exit(1)
    motion_plan, visited = motion_planner.search()
    if motion_plan is None:
        print("No plan found!")
        exit(1)
    for i, o in enumerate(obstacles):
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
        [node.state.y for node in visited],
        [node.state.x for node in visited],
        c="r",
        marker="x",
        label="Encountered state",
    )
    plt.xlim([-1, northwest_corner.y])
    plt.ylim([-1, northwest_corner.x])
    plt.grid()
    plt.legend()
    plt.show()
