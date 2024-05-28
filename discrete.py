"""
Implementations of some discrete motion planning algorithms

The algorithms are implemented in the *MotionPlanner classes. See relevant class for more details
on specific usage.

Generally, to use a planner, you must define the robot, which tells the planner how the robot moves
and transforms in space, the initial and goal states for the robot, and an occupancy grid so the
planner knows what states are occupied by obstacles.

Currently, the occupancy grid is restricted to be a uniform grid with aligned rows and columns.
Therefore, the only possible movements are up, down, left, right, and diagonal. Further, the
planners do not check for collisions along a "path" generated during a transition, so robots should
transition in a way such that the new state is no more than 1 grid square away from the original:

  x x x x x            LEGEND
  x ^ ^ ^ x    ──────────────────────
  x ^ o ^ x    x := Invalid new state
  x ^ ^ ^ x    ^ := Valid new state
  x x x x x    o := Original state

The planners do not assume any structure on the state or input spaces, so they can be anything
(R^2, R^2 X S^1, etc...), nor do they make any assumptions about how complex or simple transitions
between states are (linear, nonlinear, etc...). These decisions are left to the user, although it is
worth noting that performance is heavily determined by these decisions!
"""

from __future__ import annotations

import abc
import dataclasses
import queue
import typing

import tqdm
import typing_extensions

StateT = typing.TypeVar("StateT")
InputT = typing.TypeVar("InputT")


class CompatibleRobot(typing.Generic[StateT, InputT], typing.Protocol):
    def transition(self, state: StateT, input: InputT) -> StateT: ...
    def get_inputs(self, state: StateT) -> list[InputT]: ...


RobotT = typing.TypeVar("RobotT", bound=CompatibleRobot[typing.Any, typing.Any])
MetadataT = typing.TypeVar("MetadataT")


class ForwardSearchAlgorithm(typing.Generic[StateT, InputT, MetadataT], abc.ABC):
    """
    All motion planning algorithms adhere to the following pattern:

      Given: x := State, u := Input

      q ◄─ Queue()
      Insert x_0 into q
      While q is not empty:
        x_i ◄─ Pop q
        If x_i is the goal:
          Traverse backwards to x_0 through ancestry of x_i
          Return reverse of traversal as motion plan
        u_0, ..., u_n ◄─ All inputs applicable from x_i
        For u_j in u_0, ..., u_n:
          x_i_j ◄─ Transition x_i given u_j
          If x_i_j has not yet been encountered and is not in an occupied cell:
            Record x_i_j as an to_be_visited state with parent x_i
            Put x_i_j onto q
      No solution found => return no motion plan

    The only significant difference between different discrete planners is in how the queue is
    ordered when x_i_j is put onto q. There are some other differences (e.g. Dijkstra's records
    scores for each state visited), but these additional differences typically only exist to sort
    the queue.
    """

    def __init__(
        self,
        robot: CompatibleRobot[StateT, InputT],
        initial_state: StateT,
        goal: StateT,
        occupancy_grid: CompatibleOccupancyGrid,
    ) -> None:
        self.__robot = robot
        self.__goal = goal
        self.__occupancy_grid = occupancy_grid
        self.__to_be_visited = self._make_node_data_structure()
        self.__motion_plan: typing.Optional[list[Node[StateT, MetadataT]]] = None
        first_to_be_visisted = Node(initial_state, None, self._metadata_class())
        self.__to_be_visited.put(first_to_be_visisted)
        self.__encountered = [first_to_be_visisted]
        self.__visited: list[Node[StateT, MetadataT]] = []

    @abc.abstractmethod
    def _make_node_data_structure(
        self,
    ) -> CompatibleNodeDataStructure[MetadataT, StateT]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _metadata_class(self) -> typing.Type[MetadataT]:
        raise NotImplementedError()

    @typing.final
    def search(self) -> SearchResult[StateT, MetadataT]:
        if self.__to_be_visited.is_empty():
            # The queue is only empty if the search has already been run
            return SearchResult(self.__motion_plan, self.__visited)
        iteration = 0
        with tqdm.tqdm(total=self.__occupancy_grid.total_spaces) as progress_bar:
            while not self.__to_be_visited.is_empty():
                visiting = self.__to_be_visited.pop()
                if visiting in self.__visited:
                    continue
                # The progress bar shows how many nodes out of all possible have been processed, so
                # the algorithm may successfully terminate before the progress bar shows 100% and,
                # further, reaching 100% likely will correspond to failure to find a motion plan
                progress_bar.update(1)
                iteration += 1
                if iteration > self.__occupancy_grid.total_spaces:
                    raise RuntimeError("Algorithm ran too long!")
                self.__visited.append(visiting)
                if visiting.state == self.__goal:
                    plan: list[Node[StateT, MetadataT]] = [visiting]
                    current = visiting
                    while current.parent_node is not None:
                        current = current.parent_node
                        plan.append(current)
                    plan.reverse()
                    self.__motion_plan = plan
                    return SearchResult(self.__motion_plan, self.__visited)
                inputs = self.__robot.get_inputs(visiting.state)
                for input in inputs:
                    to_be_visited = Node(
                        self.__robot.transition(visiting.state, input),
                        visiting,
                        self._metadata_class(),
                    )
                    if self.__occupancy_grid.is_occupied(
                        to_be_visited.state
                    ) or to_be_visited.state in [v.state for v in self.__visited]:
                        continue
                    if to_be_visited.state in [e.state for e in self.__encountered]:
                        self.__to_be_visited.update(to_be_visited)
                        continue
                    self.__encountered.append(to_be_visited)
                    self.__to_be_visited.put(to_be_visited)
        return SearchResult(None, self.__visited)


@dataclasses.dataclass
class SearchResult(typing.Generic[StateT, MetadataT]):
    motion_plan: typing.Optional[list[Node[StateT, MetadataT]]]
    visited: list[Node[StateT, MetadataT]]


class CompatibleNodeDataStructure(typing.Generic[MetadataT, StateT], typing.Protocol):
    def put(self, node: Node[StateT, MetadataT]) -> None: ...
    def update(self, node: Node[StateT, MetadataT]) -> None: ...
    def pop(self) -> Node[StateT, MetadataT]: ...
    def is_empty(self) -> bool: ...


StateT_contra = typing.TypeVar("StateT_contra", contravariant=True)


class CompatibleOccupancyGrid(typing.Generic[StateT_contra], typing.Protocol):
    def is_occupied(self, state: StateT_contra) -> bool: ...
    @property
    def total_spaces(self) -> int: ...


@dataclasses.dataclass
class Node(typing.Generic[StateT, MetadataT]):
    state: StateT
    parent_node: typing.Optional[Node[StateT, MetadataT]]
    metadata: MetadataT


class NoMetadata:
    def __str__(self) -> str:
        return "None"


@typing.final
class BreadthFirstMotionPlanner(ForwardSearchAlgorithm[StateT, InputT, NoMetadata]):
    @typing_extensions.override
    def _make_node_data_structure(self) -> NodeQueue[StateT]:
        return NodeQueue()

    @property
    @typing_extensions.override
    def _metadata_class(self) -> typing.Type[NoMetadata]:
        return NoMetadata


class NoUpdateNodeDataStructure(typing.Generic[StateT, MetadataT]):
    def update(self, node: Node[StateT, MetadataT]) -> None:
        self.__do_nothing()

    def __do_nothing(self) -> None:
        pass


class NodeQueue(NoUpdateNodeDataStructure[StateT, NoMetadata]):
    def __init__(self) -> None:
        self._queue = queue.Queue()

    def put(self, node: Node[StateT, NoMetadata]) -> None:
        self._queue.put(node)

    def pop(self) -> Node[StateT, NoMetadata]:
        return self._queue.get_nowait()

    def is_empty(self) -> bool:
        return self._queue.empty()


@typing.final
class DepthFirstMotionPlanner(ForwardSearchAlgorithm[StateT, InputT, NoMetadata]):
    @typing_extensions.override
    def _make_node_data_structure(self) -> NodeStack[StateT]:
        return NodeStack()

    @property
    @typing_extensions.override
    def _metadata_class(self) -> typing.Type[NoMetadata]:
        return NoMetadata


class NodeStack(NoUpdateNodeDataStructure[StateT, NoMetadata]):
    def __init__(self) -> None:
        self._stack = []

    def put(self, node: Node[StateT, NoMetadata]) -> None:
        self._stack.append(node)

    def pop(self) -> Node[StateT, NoMetadata]:
        if len(self._stack) == 0:
            raise RuntimeError("Stack is empty!")
        return self._stack.pop(-1)

    def is_empty(self) -> bool:
        return len(self._stack) == 0


@typing.final
class DijkstraMotionPlanner(ForwardSearchAlgorithm[StateT, InputT, float]):
    def __init__(
        self, cost_to_come: typing.Callable[[StateT, StateT], float], *args, **kwargs
    ) -> None:
        self._queue = DijkstraPriorityNodeQueue(cost_to_come)
        super().__init__(*args, **kwargs)

    @typing_extensions.override
    def _make_node_data_structure(self) -> DijkstraPriorityNodeQueue[StateT]:
        return self._queue

    @property
    @typing_extensions.override
    def _metadata_class(self) -> typing.Type[float]:
        return float


class PriorityNodeQueue(typing.Generic[StateT], abc.ABC):
    def __init__(self) -> None:
        self._queue: list[Node[StateT, float]] = []

    def put(self, node: Node[StateT, float]) -> None:
        node.metadata = self._compute_cost(node)
        self._put(node)

    @abc.abstractmethod
    def _compute_cost(self, node: Node[StateT, float]) -> float:
        raise NotImplementedError()

    def _put(self, node: Node[StateT, float]) -> None:
        self._queue.append(node)
        self._queue.sort(key=lambda x: x.metadata)

    def update(self, node: Node[StateT, float]) -> None:
        queue_index = [n.state for n in self._queue].index(node.state)
        node.metadata = self._compute_cost(node)
        if self._queue[queue_index].metadata > node.metadata:
            self._queue.pop(queue_index)
            self._put(node)

    def pop(self) -> Node[StateT, float]:
        if len(self._queue) == 0:
            raise RuntimeError("Queue is empty!")
        return self._queue.pop(0)

    def is_empty(self) -> bool:
        return len(self._queue) == 0


class DijkstraPriorityNodeQueue(PriorityNodeQueue[StateT]):
    def __init__(self, cost_to_come: typing.Callable[[StateT, StateT], float]) -> None:
        self._cost_to_come = cost_to_come
        super().__init__()

    @typing_extensions.override
    def _compute_cost(self, node: Node[StateT, float]) -> float:
        if node.parent_node is None:
            return 0.0
        return node.parent_node.metadata + self._cost_to_come(
            node.parent_node.state, node.state
        )


@typing.final
class AStarMotionPlanner(ForwardSearchAlgorithm[StateT, InputT, float]):
    def __init__(
        self,
        cost_to_go: typing.Callable[[StateT], float],
        cost_to_come: typing.Callable[[StateT, StateT], float],
        *args,
        **kwargs,
    ) -> None:
        self._queue = AStarPriorityNodeQueue(cost_to_go, cost_to_come)
        super().__init__(*args, **kwargs)

    @typing_extensions.override
    def _make_node_data_structure(self) -> AStarPriorityNodeQueue[StateT]:
        return self._queue

    @property
    @typing_extensions.override
    def _metadata_class(self) -> typing.Type[float]:
        return float


@typing.final
class AStarPriorityNodeQueue(DijkstraPriorityNodeQueue[StateT]):
    def __init__(
        self, cost_to_go: typing.Callable[[StateT], float], *args, **kwargs
    ) -> None:
        self._cost_to_go = cost_to_go
        super().__init__(*args, **kwargs)

    @typing_extensions.override
    def _compute_cost(self, node: Node[StateT, float]) -> float:
        cost = super()._compute_cost(node)
        if cost > 0:
            return cost + self._cost_to_go(node.state)
        return cost
