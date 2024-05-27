"""
Simple implementation and demo of some discrete motion planning algorithms:

  - Breadth-first search
  - Depth-first search
  - Dijkstra's algorithm
  - A*

Environment (example):

   ┌───────────┐
   │ ^     ┌┐  │  ^ := Start
   │ ┌─┐   ││  │  # := Goal
   │ │ └─┐ └┘  │  o := Origin
  x▲ │ ┌─┘     │  ─ := Obstacle boundary
   │ └─┘     # │
   o──►────────┘
     y

Axis shown is the "world frame".

The obstacles shall be represented using an occupancy grid. That is, since the space is already
discretized into (in this instance) R^2, we simply encode the cells which correspond to obstacles
in a look-up table such that if p E R^2 maps to "True" then p is occupied and is not a valid space
for the robot to occupy. Encoding the example above where "1" is "True" and "0" is "False" yields:
    ┌                           ┐
  6 │ 1 1 1 1 1 1 1 1 1 1 1 1 1 │
  5 │ 1 0 0 0 0 0 0 0 1 1 0 0 1 │
  4 │ 1 0 1 1 1 0 0 0 1 1 0 0 1 │
  3 │ 1 0 1 1 1 1 1 0 1 1 0 0 1 │
  2 │ 1 0 1 1 1 1 1 0 0 0 0 0 1 │
  1 │ 1 0 1 1 1 0 0 0 0 0 0 0 1 │
  0 │ 1 1 1 1 1 1 1 1 1 1 1 1 1 │
    └                           ┘
      0 1 2 3 ...

Robot:

    ▲ x
    │
  ┌─│─┐
  │ o────► y
  └───┘

Axis shown is the "body frame" which moves about in the world frame. Coincidentally, since the
robot does not rotate, by choosing to align the body and world frame, we can coerce the
transformation matrix between the two frames to be the identity matrix. We shall do this by
convention.

The robot is free to move in both x and y dimensions independently.

The robot may only move one cell at a time (i.e. the robot cannot skip cells during a transition).
"""

from __future__ import annotations

import abc
import argparse
import dataclasses
import queue
import random
import typing

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import typing_extensions


@dataclasses.dataclass
class Input:
    dx: int
    dy: int

    @staticmethod
    def GoNorth() -> Input:
        return Input(1, 0)

    @staticmethod
    def GoEast() -> Input:
        return Input(0, 1)

    @staticmethod
    def GoSouth() -> Input:
        return Input(-1, 0)

    @staticmethod
    def GoWest() -> Input:
        return Input(0, -1)


@dataclasses.dataclass
class State:
    x: int
    y: int

    def transition(self, input: Input) -> State:
        return State(self.x + input.dx, self.y + input.dy)

    @property
    def inputs(self) -> list[Input]:
        return [Input.GoNorth(), Input.GoEast(), Input.GoSouth(), Input.GoWest()]


@dataclasses.dataclass
class ParentedState:
    state: State
    parent_id: int

    @staticmethod
    def make_with_no_parent(state: State) -> ParentedState:
        return ParentedState(state, -1)

    def has_parent(self) -> bool:
        return self.parent_id != -1


@dataclasses.dataclass
class Obstacle:
    vertices: list[State]


class RectangularOccupancyGrid:
    def __init__(
        self,
        northwest_corner: State,
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

                    def point_i_j_is_left_of(half_space: tuple[State, State]) -> bool:
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

    def is_occupied(self, state: State) -> bool:
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


class CompatibleQueue(typing.Protocol):
    def put(self, state: ParentedState) -> None: ...
    def pop(self) -> ParentedState: ...
    def is_empty(self) -> bool: ...


class CompatibleOccupancyGrid(typing.Protocol):
    def is_occupied(self, state: State) -> bool: ...
    @property
    def total_spaces(self) -> int: ...


class ForwardSearchAlgorithm:
    """
    All motion planning algorithms adhere to the following pattern:

      q ◄─ Queue()
      Insert x_0 into q
      While q is not empty:
        x_i ◄─ Pop q
        u_0, ..., u_n ◄─ All inputs applicable from x_i
        For u_j in u_0, ..., u_n:
          x_i_j ◄─ Transition x_i given u_j
          If x_i_j has not yet been to_be_visited and is not in an occupied cell:
            Record x_i_j as an to_be_visited state with parent x_i
            If x_i_j is the goal:
              Traverse backwards to x_0 through ancestry of x_i_j
              Return reverse of traversal as motion plan
            Put x_i_j onto q
      No solution found => return no motion plan

    The only significant difference between different discrete planners is in how the queue is
    ordered when x_i_j is put onto q. There are some other differences (e.g. Dijkstra's records
    scores for each state visited), but these additional differences typically only exist to sort
    the queue.
    """

    def __init__(
        self,
        queue: CompatibleQueue,
        initial_state: State,
        goal: State,
        occupancy_grid: CompatibleOccupancyGrid,
        do_shuffle_inputs: bool,
    ) -> None:
        self._queue = queue
        self._goal = goal
        self._occupancy_grid = occupancy_grid
        self._do_shuffle_inputs = do_shuffle_inputs
        self._motion_plan: typing.Optional[list[State]] = None
        first_to_be_visisted = ParentedState.make_with_no_parent(initial_state)
        self._queue.put(first_to_be_visisted)
        self._encountered = [first_to_be_visisted]
        self._visited_encountered_indices: list[int] = []

    def search(self) -> tuple[typing.Optional[list[State]], list[State]]:
        if self._queue.is_empty():
            # The queue is only empty if the search has already been run
            return (self._motion_plan, [e.state for e in self._encountered])
        iteration = 0
        with tqdm.tqdm(total=self._occupancy_grid.total_spaces) as progress_bar:
            while not self._queue.is_empty():
                visiting = self._queue.pop()
                visiting_encountered_index = self._encountered.index(visiting)
                if any(
                    [
                        visiting.state == self._encountered[i].state
                        for i in self._visited_encountered_indices
                    ]
                ):
                    continue
                progress_bar.update(1)
                iteration += 1
                if iteration > self._occupancy_grid.total_spaces:
                    raise RuntimeError("Algorithm ran too long!")
                self._visited_encountered_indices.append(visiting_encountered_index)
                inputs = visiting.state.inputs
                if self._do_shuffle_inputs:
                    random.shuffle(inputs)
                for input in inputs:
                    to_be_visited = ParentedState(
                        visiting.state.transition(input), visiting_encountered_index
                    )
                    if self._occupancy_grid.is_occupied(to_be_visited.state) or any(
                        [
                            to_be_visited.state == self._encountered[i].state
                            for i in self._visited_encountered_indices
                        ]
                    ):
                        continue
                    self._encountered.append(to_be_visited)
                    if to_be_visited.state == self._goal:
                        plan: list[State] = [to_be_visited.state]
                        current = to_be_visited
                        while current.has_parent():
                            current = self._encountered[current.parent_id]
                            plan.append(current.state)
                        plan.reverse()
                        self._motion_plan = plan
                        return (self._motion_plan, [e.state for e in self._encountered])
                    self._queue.put(to_be_visited)
        return (
            None,
            [encountered.state for encountered in self._encountered],
        )


class MotionPlanner(abc.ABC):
    def __init__(
        self,
        initial_state: State,
        goal: State,
        occupancy_grid: CompatibleOccupancyGrid,
        do_shuffle_inputs: bool,
    ) -> None:
        self._forward_search_algorithm = ForwardSearchAlgorithm(
            self._make_queue(), initial_state, goal, occupancy_grid, do_shuffle_inputs
        )

    @abc.abstractmethod
    def _make_queue(self) -> CompatibleQueue:
        raise NotImplementedError()

    def search(self) -> tuple[typing.Optional[list[State]], list[State]]:
        return self._forward_search_algorithm.search()


@typing.final
class BreadthFirstMotionPlanner(MotionPlanner):
    class _StandardQueueWrapper:
        def __init__(self) -> None:
            self._queue = queue.Queue()

        def put(self, state: ParentedState) -> None:
            self._queue.put(state)

        def pop(self) -> ParentedState:
            return self._queue.get_nowait()

        def is_empty(self) -> bool:
            return self._queue.empty()

    @typing_extensions.override
    def _make_queue(self) -> CompatibleQueue:
        return self._StandardQueueWrapper()


@typing.final
class DepthFirstMotionPlanner(MotionPlanner):
    class _Stack:
        def __init__(self) -> None:
            self._stack: list[ParentedState] = []

        def put(self, state: ParentedState) -> None:
            self._stack.append(state)

        def pop(self) -> ParentedState:
            if len(self._stack) == 0:
                raise RuntimeError("Stack is empty!")
            state = self._stack[-1]
            self._stack = self._stack[:-1]
            return state

        def is_empty(self) -> bool:
            return len(self._stack) == 0

    @typing_extensions.override
    def _make_queue(self) -> CompatibleQueue:
        return self._Stack()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algorithm",
        required=True,
        choices=["breadth-first", "depth-first"],
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
    initial_state = State(1, 9)
    goal_state = State(18, 1)
    northwest_corner = State(30, 30)
    obstacles = [
        Obstacle([State(12, 0), State(12, 15), State(15, 15), State(15, 0)]),
        Obstacle([State(15, 12), State(15, 15), State(28, 15), State(28, 12)]),
        Obstacle([State(15, 15), State(15, 28), State(18, 28), State(18, 15)]),
        Obstacle([State(20, 18), State(20, 30), State(23, 30), State(23, 18)]),
    ]
    occupancy_grid = RectangularOccupancyGrid(northwest_corner, obstacles)
    if arguments.algorithm == "breadth-first":
        planner = BreadthFirstMotionPlanner(
            initial_state, goal_state, occupancy_grid, arguments.random
        )
    elif arguments.algorithm == "depth-first":
        planner = DepthFirstMotionPlanner(
            initial_state, goal_state, occupancy_grid, arguments.random
        )
    else:
        print(f"Unrecognized planner: {arguments.algorithm}")
        exit(1)
    plan, encountered_states = planner.search()
    if plan is None:
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
    plt.plot(plan[0].y, plan[0].x, "bx", markersize=10, label="Initial state")
    plt.plot(plan[-1].y, plan[-1].x, "bo", markersize=10, label="Goal state")
    plt.plot(
        [state.y for state in plan],
        [state.x for state in plan],
        "-b",
        label="Motion plan",
    )
    plt.scatter(
        [state.y for state in encountered_states],
        [state.x for state in encountered_states],
        c="r",
        marker="x",
        label="Encountered state",
    )
    plt.xlim([-1, northwest_corner.y])
    plt.ylim([-1, northwest_corner.x])
    plt.grid()
    plt.legend()
    plt.show()
