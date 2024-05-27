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
import dataclasses
import queue
import random
import typing

import tqdm
import typing_extensions

InputT_contra = typing.TypeVar("InputT_contra", contravariant=True)


class CompatibleState(typing.Generic[InputT_contra], typing.Protocol):
    def transition(self, input: InputT_contra) -> typing_extensions.Self: ...


StateT = typing.TypeVar("StateT", bound=CompatibleState[typing.Any])
MetadataT = typing.TypeVar("MetadataT")


@dataclasses.dataclass
class Node(typing.Generic[StateT, MetadataT]):
    state: StateT
    parent_node: typing.Optional[Node[StateT, MetadataT]]
    metadata: MetadataT


class CompatibleQueue(typing.Generic[MetadataT, StateT], typing.Protocol):
    def put(self, node: Node[StateT, MetadataT]) -> None: ...
    def update(self, node: Node[StateT, MetadataT]) -> None: ...
    def pop(self) -> Node[StateT, MetadataT]: ...
    def is_empty(self) -> bool: ...


StateT_contra = typing.TypeVar(
    "StateT_contra", bound=CompatibleState[typing.Any], contravariant=True
)


class CompatibleOccupancyGrid(typing.Generic[StateT_contra], typing.Protocol):
    def is_occupied(self, state: StateT_contra) -> bool: ...
    @property
    def total_spaces(self) -> int: ...


class ForwardSearchAlgorithm(typing.Generic[StateT, MetadataT]):
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
        metadata_class: typing.Type[MetadataT],
        initial_state: StateT,
        goal: StateT,
        occupancy_grid: CompatibleOccupancyGrid,
        do_shuffle_inputs: bool,
    ) -> None:
        self._queue = queue
        self._metadata_class = metadata_class
        self._state_class = type(initial_state)
        self._goal = goal
        self._occupancy_grid = occupancy_grid
        self._do_shuffle_inputs = do_shuffle_inputs
        self._motion_plan: typing.Optional[list[Node[StateT, MetadataT]]] = None
        first_to_be_visisted = Node(initial_state, None, self._metadata_class())
        self._queue.put(first_to_be_visisted)
        self._encountered = [first_to_be_visisted]
        self._visited: list[Node[StateT, MetadataT]] = []

    def search(
        self,
    ) -> tuple[
        typing.Optional[list[Node[StateT, MetadataT]]], list[Node[StateT, MetadataT]]
    ]:
        if self._queue.is_empty():
            # The queue is only empty if the search has already been run
            return (self._motion_plan, self._visited)
        iteration = 0
        with tqdm.tqdm(total=self._occupancy_grid.total_spaces) as progress_bar:
            while not self._queue.is_empty():
                visiting = self._queue.pop()
                if visiting in self._visited:
                    continue
                # The progress bar shows how many nodes out of all possible have been processed, so
                # the algorithm may successfully terminate before the progress bar shows 100% and,
                # further, reaching 100% likely will correspond to failure to find a motion plan
                progress_bar.update(1)
                iteration += 1
                if iteration > self._occupancy_grid.total_spaces:
                    raise RuntimeError("Algorithm ran too long!")
                self._visited.append(visiting)
                if visiting.state == self._goal:
                    plan: list[Node[StateT, MetadataT]] = [visiting]
                    current = visiting
                    while current.parent_node is not None:
                        current = current.parent_node
                        plan.append(current)
                    plan.reverse()
                    self._motion_plan = plan
                    return (self._motion_plan, self._visited)
                inputs = visiting.state.inputs
                if self._do_shuffle_inputs:
                    random.shuffle(inputs)
                for input in inputs:
                    to_be_visited = Node(
                        visiting.state.transition(input),
                        visiting,
                        self._metadata_class(),
                    )
                    if self._occupancy_grid.is_occupied(
                        to_be_visited.state
                    ) or to_be_visited.state in [v.state for v in self._visited]:
                        continue
                    if to_be_visited.state in [e.state for e in self._encountered]:
                        self._queue.update(to_be_visited)
                        continue
                    self._encountered.append(to_be_visited)
                    self._queue.put(to_be_visited)
        return (None, self._visited)


class MotionPlanner(typing.Generic[StateT, MetadataT], abc.ABC):
    def __init__(
        self,
        initial_state: StateT,
        goal: StateT,
        occupancy_grid: CompatibleOccupancyGrid,
        do_shuffle_inputs: bool,
    ) -> None:
        self._forward_search_algorithm = ForwardSearchAlgorithm(
            self._make_queue(),
            self._metadata_class,
            initial_state,
            goal,
            occupancy_grid,
            do_shuffle_inputs,
        )

    @abc.abstractmethod
    def _make_queue(self) -> CompatibleQueue:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _metadata_class(self) -> typing.Type[MetadataT]:
        raise NotImplementedError()

    def search(
        self,
    ) -> tuple[
        typing.Optional[list[Node[StateT, MetadataT]]],
        list[Node[StateT, MetadataT]],
    ]:
        return self._forward_search_algorithm.search()


class NoMetadata:
    def __str__(self) -> str:
        return "None"


class NoMetadataMotionPlanner(MotionPlanner[StateT, NoMetadata], abc.ABC):
    @property
    @typing_extensions.override
    def _metadata_class(self) -> typing.Type[NoMetadata]:
        return NoMetadata


@typing.final
class BreadthFirstMotionPlanner(NoMetadataMotionPlanner[StateT]):
    class _StandardQueueWrapper:
        def __init__(self) -> None:
            self._queue = queue.Queue()

        def put(self, node: Node[StateT, NoMetadata]) -> None:
            self._queue.put(node)

        def update(self, node: Node[StateT, NoMetadata]) -> None:
            pass  # Do nothing

        def pop(self) -> Node[StateT, NoMetadata]:
            return self._queue.get_nowait()

        def is_empty(self) -> bool:
            return self._queue.empty()

    @typing_extensions.override
    def _make_queue(self) -> BreadthFirstMotionPlanner._StandardQueueWrapper:
        return self._StandardQueueWrapper()


@typing.final
class DepthFirstMotionPlanner(NoMetadataMotionPlanner[StateT]):
    class _Stack:
        def __init__(self) -> None:
            self._stack = []

        def put(self, node: Node[StateT, NoMetadata]) -> None:
            self._stack.append(node)

        def update(self, node: Node[StateT, NoMetadata]) -> None:
            pass  # Do nothing

        def pop(self) -> Node[StateT, NoMetadata]:
            if len(self._stack) == 0:
                raise RuntimeError("Stack is empty!")
            return self._stack.pop(-1)

        def is_empty(self) -> bool:
            return len(self._stack) == 0

    @typing_extensions.override
    def _make_queue(self) -> DepthFirstMotionPlanner._Stack:
        return self._Stack()


class CostBasedMotionPlanner(MotionPlanner[StateT, float], abc.ABC):
    def __init__(
        self, distance_to: typing.Callable[[StateT, StateT], float], *args, **kwargs
    ) -> None:
        self._distance_to = distance_to
        super().__init__(*args, **kwargs)

    class _PriorityQueue:
        def __init__(
            self, compute_cost: typing.Callable[[Node[StateT, float]], None]
        ) -> None:
            self._queue = []
            self._compute_cost = compute_cost

        def put(self, node: Node[StateT, float]) -> None:
            self._compute_cost(node)
            self._put(node)

        def _put(self, node: Node[StateT, float]) -> None:
            self._queue.append(node)
            self._queue.sort(key=lambda x: x.metadata)

        def update(self, node: Node[StateT, float]) -> None:
            queue_index = [n.state for n in self._queue].index(node.state)
            self._compute_cost(node)
            if self._queue[queue_index].metadata > node.metadata:
                self._queue.pop(queue_index)
                self._put(node)

        def pop(self) -> Node[StateT, float]:
            if len(self._queue) == 0:
                raise RuntimeError("Queue is empty!")
            return self._queue.pop(0)

        def is_empty(self) -> bool:
            return len(self._queue) == 0

    @typing_extensions.override
    def _make_queue(self) -> CostBasedMotionPlanner._PriorityQueue:
        return self._PriorityQueue(self._compute_cost)

    @abc.abstractmethod
    def _compute_cost(self, node: Node[StateT, float]) -> None:
        raise NotImplementedError()

    @property
    @typing_extensions.override
    def _metadata_class(self) -> typing.Type[float]:
        return float


@typing.final
class DijkstraMotionPlanner(CostBasedMotionPlanner[StateT]):
    @typing_extensions.override
    def _compute_cost(self, node: Node[StateT, float]) -> None:
        if node.parent_node is None:
            node.metadata = 0.0
        else:
            node.metadata = node.parent_node.metadata + self._distance_to(
                node.parent_node.state, node.state
            )


@typing.final
class AStarMotionPlanner(CostBasedMotionPlanner[StateT]):
    def __init__(self, *args, **kwargs) -> None:
        self._goal: StateT = args[2]
        super().__init__(*args, **kwargs)

    @typing_extensions.override
    def _compute_cost(self, node: Node[StateT, float]) -> None:
        if node.parent_node is None:
            node.metadata = 0.0
        else:
            node.metadata = (
                node.parent_node.metadata
                + self._distance_to(node.parent_node.state, node.state)
                + self._distance_to(self._goal, node.state)
            )
