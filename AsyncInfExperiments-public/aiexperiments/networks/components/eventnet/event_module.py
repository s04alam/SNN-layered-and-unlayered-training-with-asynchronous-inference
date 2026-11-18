from abc import ABC
from typing import Tuple, List
import numpy as np

from aiexperiments.inference.event_utils import UnindexedEventWithValue


# refractory_period to None and event_count_thresh = 20 results in much fewer spikes with same accuracy
class EventModuleParamters:
    def __init__(
        self,
        y_init: float = 0,
        y_thresh: float = 0.3,
        y_reset: float = 0,
        refractory_period: int = 0,  # note that 0 and None are not the same, None means a neuron can spike multiple times in the same timestep
        timestep_size: int = 0, # used for numerical integration, 0 means analytical integration
        log=True,
    ):
        self.y_init = y_init
        self.y_thresh = y_thresh
        self.y_reset = y_reset
        self.refractory_period = refractory_period
        self.timestep_size = timestep_size
        self.log = log

class EventModule(ABC):

    """
    Abstract base class for event-based modules.
    """
    def __init__(
        self,
        params: EventModuleParamters,
    ):
        self.y_init = params.y_init
        self.y_thresh = params.y_thresh
        self.y_reset = params.y_reset
        self.refractory_period = params.refractory_period
        self.timestep_size = params.timestep_size
        self.log = params.log

        if (
            self.refractory_period is not None
            and self.timestep_size > 0
            and self.refractory_period % self.timestep_size != 0
        ):
            raise Exception("Timestep size must be a multiple of the refractory period")
        self.reset_state()

    def reset_state(self) -> None:
        self.ts = np.array([0], dtype=int)
        self.ys = np.array([self.y_init], dtype=float)
        self.refractory_until = -1
        self.n_steps = 1
        if self.log:
            self.threshold_timings = []
            self.external_input_timings = []
            self.external_input_amounts = []

    # returns threshold crossing times
    def external_input(self, t: int, y_add: float, source_index: int) -> List[UnindexedEventWithValue]:
        if self.timestep_size > 0 and t % self.timestep_size != 0:
            print("Warning: external input time is not a multiple of the timestep size")
        output_events = []
        if t > self.refractory_until:
            if self.log:
                self.external_input_timings.append(t)
                self.external_input_amounts.append(y_add)
            # Make sure the time is not in the past
            if t < self.ts[-1]:
                raise Exception("Cannot revert time to the past")
            # Integrate to the time of the external input
            time_forward = t - self.ts[-1]
            if time_forward > 0:
                self.forward(time_forward)
            # Add the external input
            self.ys[-1] += y_add
            if self.exceeds_threshold(self.ys[-1]):
                output_events.append(UnindexedEventWithValue(t, 1))
                self.on_event(t)
            # self.clean_memory()
        return output_events

    # returns threshold crossing times
    def forward(self, t: int):
        t_target = self.ts[-1] + t
        while self.ts[-1] < t_target:
            t0 = self.ts[-1]
            if self.timestep_size <= 0:
                ts, ys, success = self.integrate_analytically(t0, t_target, self.ys[-1])
            else:
                ts, ys, success = self.integrate_numerically(t0, t_target, self.ys[-1])
            if not success:
                raise Exception("Integration failed")
            if len(ts) > 0:
                self.ts = np.append(self.ts, ts)
                self.ys = np.append(self.ys, ys)
                self.n_steps += len(ts)
            else:
                break # This means that the timestep size is too small: forwarding makes no progress

    def on_event(self, threshold_time: int) -> None:
        if self.log:
            self.threshold_timings.append(threshold_time)
        self.ys[-1] = self.y_reset
        if self.refractory_period is not None:
            # note: even if the refractory period is 0, this prevents spiking in the same timestep
            self.refractory_until = threshold_time + self.refractory_period

    # Removes all timesteps and values except the last one.
    def clean_memory(self) -> None:
        self.ts = self.ts[-1:]
        self.ys = self.ys[-1:]

    def integrate_numerically(
        self, t0: int, t_end: int, y0: float
    ) -> Tuple[List[int], List[float], bool]:
        steps = int((t_end - t0) // self.timestep_size)
        t_out = np.linspace(t0+self.timestep_size, t_end, steps)
        y_out = np.zeros(steps)
        y_last = y0
        for i in range(steps):
            y_out[i] = self.numerical_step(y_last)
            y_last = y_out[i]
        return t_out, y_out, True

    def integrate_analytically(
        self, t0: int, t_end: int, y0: float
    ) -> Tuple[List[int], List[float], bool]:
        t_out = [t_end]
        y_out = [self.analytical(t_end-t0, y0)]
        return t_out, y_out, True

    def analytical(self, t: int, y0: float) -> float:
        raise NotImplementedError

    def numerical_step(self, y0: float) -> float:
        raise NotImplementedError

    def exceeds_threshold(self, y: float) -> bool:
        return self.y_thresh is not None and y >= self.y_thresh
