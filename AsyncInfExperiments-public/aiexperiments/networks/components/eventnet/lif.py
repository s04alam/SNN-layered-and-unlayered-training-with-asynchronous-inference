from typing import List
from aiexperiments.networks.components.eventnet.event_module import (
    EventModule,
    EventModuleParamters,
)
import numpy as np

from aiexperiments.inference.event_utils import UnindexedEventWithValue


class LIFParameters(EventModuleParamters):
    def __init__(self, tau_membrane=20.0, **kwargs):
        super().__init__(**kwargs)
        self.tau_membrane = tau_membrane
        if self.timestep_size > 0:
            self.beta = np.exp(-self.timestep_size / self.tau_membrane)


class LIF(EventModule):
    def __init__(self, params: LIFParameters):
        super().__init__(params)
        self.tau_membrane = params.tau_membrane
        if params.timestep_size > 0:
            self.beta = params.beta

    def analytical(self, t: int, y0: float) -> float:
        return y0 * np.exp(-t / self.tau_membrane)

    def numerical_step(self, y0: float) -> float:
        return y0 * self.beta


class BarrierLIFParameters(LIFParameters):
    def __init__(
        self, min_barrier_events_for_spike, send_barrier_events=True, ignore_spike_as_barrier_message=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.send_barrier_events = send_barrier_events
        self.min_barrier_events_for_spike = min_barrier_events_for_spike
        self.ignore_spike_as_barrier_message = ignore_spike_as_barrier_message


class BarrierLIF(LIF):
    def __init__(self, params: BarrierLIFParameters):
        super().__init__(params)
        self.send_barrier_events = params.send_barrier_events
        self.min_barrier_events_for_spike = params.min_barrier_events_for_spike
        self.ignore_spike_as_barrier_message = params.ignore_spike_as_barrier_message

    def reset_state(self) -> None:
        super().reset_state()
        self.on_event_barrier_event_count = 0
        self.barrier_events_received_from = set()
        self.last_barrier_message_sent_time = -1
        self.last_barrier_message_received_time = -1

    def external_input(self, t: int, y_add: float, source_index: int) -> List[UnindexedEventWithValue]:
        # reset barrier event count if time has passed since last barrier message
        if t > self.last_barrier_message_received_time:
            self.barrier_events_received_from = set()
        self.last_barrier_message_received_time = t

        if not self.ignore_spike_as_barrier_message or y_add == 0:
            self.barrier_events_received_from.add(source_index)
        output_events = super().external_input(t, y_add, source_index)
        if (
            len(output_events) == 0
            and self.send_barrier_events
            and t > self.last_barrier_message_sent_time
            and len(self.barrier_events_received_from) >= self.min_barrier_events_for_spike
        ):
            output_events = [UnindexedEventWithValue(t, 0)]  # zero valued event
        if len(output_events) > 0:
            self.last_barrier_message_sent_time = t # a normal spike will reset this too, since a normal spike functions as a barrier event as well
        return output_events

    def on_event(self, threshold_time: int) -> None:
        self.on_event_barrier_event_count = len(self.barrier_events_received_from)
        self.barrier_events_received_from = set()
        return super().on_event(threshold_time)

    def exceeds_threshold(self, y: float) -> bool:
        return super().exceeds_threshold(y) and (
            len(self.barrier_events_received_from) >= self.min_barrier_events_for_spike
        )
    