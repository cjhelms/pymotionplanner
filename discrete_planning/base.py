import abc
import typing

import numpy as np
import numpy.typing as npt


class CompatibleDataStructure(typing.Protocol):
    def put(self, element: int) -> None: ...
    def pop(self) -> int: ...
    def __len__(self) -> int: ...


_T = typing.TypeVar("_T", bound=CompatibleDataStructure)


class ForwardSearchAlgorithm(typing.Generic[_T], abc.ABC):
    _NO_PARENT_INDEX = -1

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        goal_region: npt.NDArray[np.float_],
        get_inputs: typing.Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
        transition: typing.Callable[
            [npt.NDArray[np.float_], npt.NDArray[np.float_]], npt.NDArray[np.float_]
        ],
        maximum_n_iterations=1000,
    ) -> None:
        """
        Arguments:
            initial_state: D-dimensional array containing initial state
            goal_region: G-by-D array containing G D-dimensional points containing the states for
                which one must be reached by the found path
            get_inputs: Given D-dimensional state, returns K-by-U array containing K U-dimensional
                inputs (need not be ordered) which may be applied to given state
            transition: Given D-dimensional state and U dimensional input, applies input to state
                and returns resulting D-dimensional new state
            maximum_n_iterations: Maximum iterations allowed before failure is declared
        """
        self.__goal_region = goal_region
        self.__maximum_n_iterations = maximum_n_iterations
        self._get_inputs = get_inputs
        self._transition = transition
        self.__states = np.zeros((maximum_n_iterations, initial_state.shape[0]))
        self.__states[0, :] = initial_state
        self.__n_states_discovered = 1
        self._unvisisted_state_indices = self._make_unvisited_data_structure()
        self.__parent_indices = np.zeros(maximum_n_iterations).astype(np.int_)
        self.__parent_indices[0] = self._NO_PARENT_INDEX
        self.__inputs = np.zeros((maximum_n_iterations, initial_state.shape[0]))
        self.__iteration = 0
        self._post_init()

    @abc.abstractmethod
    def _make_unvisited_data_structure(self) -> _T:
        """
        Return is used as the data structure to store state indices
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _post_init(self) -> None:
        """
        Do any work here which you want done immediately after initialization
        """
        raise NotImplementedError()

    def search(self) -> typing.Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Executes forward search algorithm

        Exact behavior (algorithm) is determined by child class. Empty arrays means that no solution
        exists.

        Returns:
            1. N-by-D array containing the ordered N D-dimensional states of the path
            2. (N+1)-by-U array containing the ordered N U-dimensional inputs where the first input
                is the input to apply right now and the next N inputs are the inputs to apply to
                the corresponding N output states
        """
        while len(self._unvisisted_state_indices) > 0:
            self.__iteration += 1
            if self.__iteration > self.__maximum_n_iterations:
                raise MaximumIterationError()
            self._sort_unvisited()
            state_index, state = self.__pop_unvisited()
            if self.__state_array_contains(self.__goal_region, state):
                return self.__get_solution(state_index)
            for input in self._get_inputs(state):
                next_state = self._transition(state, input)
                if not self.__state_array_contains(self.__states, next_state):
                    self.__handle_next_state(next_state, state_index)
        return (np.array([]), np.array([]))

    @abc.abstractmethod
    def _sort_unvisited(self) -> None:
        """
        Should sort the unvisited indices for the next iteration of the forward search
        """
        raise NotImplementedError()

    def __pop_unvisited(self) -> tuple[int, npt.NDArray[np.float_]]:
        state_index = self._unvisisted_state_indices.pop()
        return (state_index, self.__states[state_index])

    def __state_array_contains(
        self, state_array: npt.NDArray[np.float_], state: npt.NDArray[np.float_]
    ) -> bool:
        EPSILON = 0.01
        return bool(np.any(np.all(np.abs(state_array - state) < EPSILON, 1)))

    def __get_solution(
        self, state_index: int
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        states = self.__states[state_index]
        inputs = self.__states[state_index]
        i = self.__parent_indices[state_index]
        while i != self._NO_PARENT_INDEX:
            states = np.append(states, self.__states[i])
            inputs = np.append(inputs, self.__inputs[i])
            i = self.__parent_indices[i]
        return (self.__states, self.__inputs)

    def __handle_next_state(
        self, next_state: npt.NDArray[np.float_], parent_state_index: int
    ) -> None:
        self.__states[self.__n_states_discovered] = next_state
        self.__parent_indices[self.__n_states_discovered] = parent_state_index
        self._unvisisted_state_indices.put(self.__n_states_discovered)
        self.__n_states_discovered += 1


class MaximumIterationError(Exception):
    pass


class IntDataStructure(abc.ABC):
    def __init__(self) -> None:
        self.data: npt.NDArray[np.int_] = np.array([]).astype(np.int_)

    def put(self, element: int) -> None:
        self.data = np.append(self.data, element)

    @abc.abstractmethod
    def pop(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.data.shape[0]


class IntQueue(IntDataStructure):
    @typing.override
    def pop(self) -> int:
        if self.data.shape[0] == 0:
            raise EmptyDataStructureError()
        element = self.data[0]
        if self.data.shape[0] > 1:
            self.data = self.data[1:]
        else:
            self.data = np.array([]).astype(np.int_)
        return element


class EmptyDataStructureError(Exception):
    pass


class IntStack(IntDataStructure):
    def pop(self) -> int:
        if self.data.shape[0] == 0:
            raise EmptyDataStructureError()
        element = self.data[-1]
        self.data = self.data[:-1]
        return element


@typing.final
class BreadthFirstForwardSearchAlgorithm(ForwardSearchAlgorithm[IntQueue]):
    @typing.override
    def _post_init(self) -> None:
        _do_nothing()

    @typing.override
    def _make_unvisited_data_structure(self) -> IntQueue:
        return IntQueue()

    @typing.override
    def _sort_unvisited(self) -> None:
        _do_nothing()


def _do_nothing() -> None:
    pass


@typing.final
class DepthFirstForwardSearchAlgorithm(ForwardSearchAlgorithm[IntStack]):
    @typing.override
    def _post_init(self) -> None:
        _do_nothing()

    @typing.override
    def _make_unvisited_data_structure(self) -> IntStack:
        return IntStack()

    @typing.override
    def _sort_unvisited(self) -> None:
        _do_nothing()


@typing.final
class DijkstrasForwardSearchAlgorithm(ForwardSearchAlgorithm[IntStack]):
    @typing.override
    def _post_init(self) -> None:
        # TODO
        pass

    @typing.override
    def _make_unvisited_data_structure(self) -> IntStack:
        return IntStack()

    @typing.override
    def _sort_unvisited(self) -> None:
        # TODO
        pass


@typing.final
class AStarForwardSearchAlgorithm(ForwardSearchAlgorithm[IntStack]):
    @typing.override
    def _post_init(self) -> None:
        # TODO
        pass

    @typing.override
    def _make_unvisited_data_structure(self) -> IntStack:
        return IntStack()

    @typing.override
    def _sort_unvisited(self) -> None:
        # TODO
        pass
