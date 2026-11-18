import random
from typing import Dict, List, Tuple, Union
import torch
from aiexperiments.base_engine import StaticParameters
from aiexperiments.constants import (
    DatasetProvider,
    LossFunctionType,
    EventSchedulingMethod,
    EventOrderingMethod,
)
from aiexperiments.datasets.transforms import convert_tensor_to_named_tuple
from aiexperiments.inference.base_inference import (
    BaseInferenceEngine,
    InferenceParameters,
)
from aiexperiments.networks.components.eventnet.event_layer import DenseLIFLayer
from aiexperiments.networks.components.eventnet.lif import BarrierLIF
from aiexperiments.visualizing.metrics import ce_count_loss_numpy, ce_max_membrane_loss_numpy, ce_rate_loss_numpy
from aiexperiments.visualizing.recording.event_recorder import (
    EventBatchResult,
    EventRecorder,
)
from aiexperiments.inference.event_utils import (
    EventWithValue,
    classification_events_to_output_recordings,
    classification_events_to_array,
    membrane_potential_recordings_to_output_recordings,
)
from aiexperiments.networks.base_network import (
    BaseEventDrivenNetwork,
    BaseNetworkBuilder,
    BaseNetworkBuilderParameters,
)
import numpy as np
import heapq
from itertools import chain

# from concurrent.futures import TimeoutError
# from pebble import ProcessPool, ProcessExpired
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class EventDrivenInferenceParameters(InferenceParameters):
    def __init__(
        self,
        ordering_method: EventOrderingMethod,
        scheduling_method: EventSchedulingMethod,
        loss_function_type: LossFunctionType,
        instantly_forward_input: bool = False,
        add_input_barrier: bool = False,
        destroy_input=False,  # TODO: remove this
        **kwargs,
    ):
        super().__init__(
            dataset_provider=DatasetProvider.TONIC_EVENTS, pre_build_network=kwargs["batch_size"] == 1, **kwargs
        )
        self.ordering_method = ordering_method
        self.loss_function_type = loss_function_type
        self.destroy_input = destroy_input
        self.scheduling_method = scheduling_method
        self.instantly_forward_input = instantly_forward_input
        self.add_input_barrier = add_input_barrier
        if self.add_input_barrier and not self.instantly_forward_input:
            print("WARNING: Input barrier messages are only supported with instant forwarding of input events.")
            self.instantly_forward_input = True
        if loss_function_type == LossFunctionType.RATE:
            self.loss_function = ce_rate_loss_numpy()
        elif loss_function_type == LossFunctionType.COUNT:
            self.loss_function = ce_count_loss_numpy()
        elif (
            loss_function_type == LossFunctionType.MAX_MEMBRANE or loss_function_type == LossFunctionType.LAST_MEMBRANE
        ):
            self.loss_function = (
                ce_max_membrane_loss_numpy()
            )  # does not matter if it is last or max membrane, since this is not handled in the loss function for event driven inference
        else:
            raise ValueError(f"Loss function {loss_function_type} not supported.")


class EventInHeap:
    def __init__(
        self,
        target_layer_idx: int,
        event: EventWithValue,
        ordering_method: EventOrderingMethod,
        is_first_half: bool,
        priority: float,
    ):
        self.target_layer_idx = target_layer_idx
        self.event = event
        self.ordering_method = ordering_method
        self.half_idx = 1 - int(is_first_half)
        self.priority = priority

    def __lt__(self, other):
        if self.ordering_method == EventOrderingMethod.PER_LAYER:
            return (self.target_layer_idx, self.priority) < (
                other.target_layer_idx,
                other.priority,
            )
        elif self.ordering_method == EventOrderingMethod.STRICT_PER_LAYER:
            return (self.target_layer_idx, self.event.index, self.priority) < (
                other.target_layer_idx,
                other.event.index,
                other.priority,
            )
        elif self.ordering_method == EventOrderingMethod.PER_HALF:
            return (self.half_idx, self.priority) < (other.half_idx, other.priority)
        else:
            return self.priority < other.priority


class SimpleEventInHeap(EventInHeap):
    def __init__(self, target_layer_idx, event, priority):
        super().__init__(target_layer_idx, event, EventOrderingMethod.NONE, True, priority)


class SyncGroupDict:
    def __init__(self, ordering_method: EventOrderingMethod):
        self.ordering_method = ordering_method
        self.sync_groups = {}

    def append(self, target_layer_idx: int, event: EventWithValue, is_first_half: bool):
        if self.ordering_method == EventOrderingMethod.PER_LAYER:
            sync_group = (target_layer_idx,)
        elif self.ordering_method == EventOrderingMethod.STRICT_PER_LAYER:
            sync_group = (target_layer_idx, event.index)
        elif self.ordering_method == EventOrderingMethod.PER_HALF:
            sync_group = (1 - int(is_first_half),)
        else:
            sync_group = (0,)
        if self.sync_groups.get(sync_group) is None:
            self.sync_groups[sync_group] = []
        self.sync_groups[sync_group].append((target_layer_idx, event))

    def pop(self) -> Tuple[int, EventWithValue]:
        sync_group = min(self.sync_groups.keys())  # find sync group with lowest index
        pop_index = random.randrange(len(self.sync_groups[sync_group]))  # random index within sync group
        target_and_event = self.sync_groups[sync_group].pop(pop_index)
        if len(self.sync_groups[sync_group]) == 0:
            del self.sync_groups[sync_group]
        return target_and_event

    def clear(self):
        for sync_group_key in list(self.sync_groups.keys()):
            self.sync_groups[sync_group_key].clear()
            del self.sync_groups[sync_group_key]

    def __len__(self):
        return sum([len(sync_group) for sync_group in self.sync_groups.values()])


class StatefulEventDrivenInferenceEngine:
    ts_dict: Dict[int, Union[List[EventInHeap], SyncGroupDict]]
    priority_events: Dict[int, List[SimpleEventInHeap]]
    ts_current: int

    def __init__(self, params: EventDrivenInferenceParameters, network: BaseEventDrivenNetwork):
        self.ordering_method = params.ordering_method
        self.scheduling_method = params.scheduling_method
        self.network = network
        self.ts_dict = {}
        self.ts_current = 0
        self.spike_count = 0  # With barrier messages
        self.true_spike_count = 0  # Without barrier messages
        self.true_spike_count_network_only = 0  # Without barrier messages and without input layer
        self.disordered_spike_count = 0
        self.classification_latencies = []
        self.correct_classification_latencies = []
        self.last_classification_time = -1
        self.last_correct_classification_time = -1
        self.priorities_given = []
        self.instantly_forward_input = params.instantly_forward_input
        self.add_input_barrier = params.add_input_barrier
        # events which must be forwarded instantly, indexed by timestamp (keys)
        self.priority_events = {}

    def append_event(self, target_layer_idx: int, event: EventWithValue, event_priority_if_input=0):
        # Input events, which must be forwarded instantly if specified
        if self.instantly_forward_input and target_layer_idx == 0:
            if self.priority_events.get(event.time) is None:
                self.priority_events[event.time] = []
            heapq.heappush(
                self.priority_events[event.time],
                SimpleEventInHeap(target_layer_idx, event, event_priority_if_input),
            )
            self.spike_count += 1
            if event.value != 0:
                self.true_spike_count += 1
            return  # do not add to queue

        # Normal events
        if self.ts_dict.get(event.time) is None:
            if self.scheduling_method == EventSchedulingMethod.RANDOM_SAMPLING:
                self.ts_dict[event.time] = SyncGroupDict(self.ordering_method)
            else:
                self.ts_dict[event.time] = []
        self.spike_count += 1
        if event.value != 0:
            self.true_spike_count += 1
            if target_layer_idx > 0:
                self.true_spike_count_network_only += 1

        is_first_half = target_layer_idx < len(self.network.layers) // 2
        if self.scheduling_method == EventSchedulingMethod.RANDOM_SAMPLING:
            self.ts_dict[event.time].append(target_layer_idx, event, is_first_half)
        else:
            # Calculate priority
            if self.scheduling_method == EventSchedulingMethod.DISTORTED_SPIKE_COUNT:
                self.disordered_spike_count += np.random.normal(
                    1, 1.337
                )  # based on data from https://arxiv.org/pdf/2003.09696.pdf
                # Calculate priority (lower = earlier)
                ## Based ond data from https://arxiv.org/pdf/2003.09696.pdf
                ## The idea as follows: we have a spike, which has an ISI (inter-spike interval) and a queue delay
                ## ISI is based on figure 22
                isi_in_spike = np.random.normal(0, 1.337)
                priority = self.disordered_spike_count + isi_in_spike
            elif self.scheduling_method == EventSchedulingMethod.RANDOM_PRIORITY:
                priority = random.random()
            elif self.scheduling_method == EventSchedulingMethod.SPIKE_COUNT:
                priority = self.spike_count
            elif self.scheduling_method == EventSchedulingMethod.EVENT_CURRENT:
                priority = -self.network.layers[target_layer_idx].weights[:, event.index].sum().item()
            self.priorities_given.append(priority)  # log

            # Add event to queue
            heapq.heappush(
                self.ts_dict[event.time],
                EventInHeap(
                    target_layer_idx,
                    event,
                    self.ordering_method,
                    is_first_half,
                    priority,
                ),
            )

    def get_next_event(self) -> Tuple[int, EventWithValue]:
        # Make sure the timestamp is not empty
        ts_current_events = self.ts_dict.get(self.ts_current)
        ts_current_events_priority = self.priority_events.get(self.ts_current)
        while (ts_current_events_priority is None or len(ts_current_events_priority) == 0) and (
            ts_current_events is None or len(ts_current_events) == 0
        ):
            # If no events for current timestamp, get next timestamp
            # First make sure that empty keys are removed from dict
            if ts_current_events_priority is not None:
                del self.priority_events[self.ts_current]
            if ts_current_events is not None:
                del self.ts_dict[self.ts_current]
            # If no events left, return None
            if len(self.priority_events.keys()) == 0 and len(self.ts_dict.keys()) == 0:
                return None
            # Find next events
            ts_keys = list(self.ts_dict.keys())
            ts_priority_keys = list(self.priority_events.keys())
            if len(ts_keys) == 0:
                self.ts_current = min(ts_priority_keys)
            elif len(ts_priority_keys) == 0:
                self.ts_current = min(ts_keys)
            else:
                self.ts_current = min(min(ts_keys), min(ts_priority_keys))
            ts_current_events = self.ts_dict.get(self.ts_current)
            ts_current_events_priority = self.priority_events.get(self.ts_current)

        # Return next event for current timestamp
        if ts_current_events_priority is not None and len(ts_current_events_priority) > 0:
            event_in_queue = heapq.heappop(ts_current_events_priority)
            return (event_in_queue.target_layer_idx, event_in_queue.event)
        else:
            if self.scheduling_method == EventSchedulingMethod.RANDOM_SAMPLING:
                return ts_current_events.pop()
            else:
                event_in_queue = heapq.heappop(ts_current_events)
                return (event_in_queue.target_layer_idx, event_in_queue.event)

    def infer(
        self, events: List[EventWithValue], true_label: int = None
    ) -> Tuple[List[EventWithValue], List[Tuple[int, EventWithValue]], List[List[Tuple[int, EventWithValue]]]]:
        membrane_event_log = []
        event_order_log = []  # list of (layer_idx, event)
        for event in events:
            if event.value != 0:
                self.append_event(
                    0, event, event_priority_if_input=np.random.rand()
                )  # input events, appended to first layer
        classification_events: List[EventWithValue] = []  # list of events, sorted by timestamp
        # Add input barrier messages (with lower priority than input spikes)
        if self.add_input_barrier and isinstance(self.network.layers[0], DenseLIFLayer):
            for t in self.priority_events.keys():
                # for i in range(len(self.network.layers[0].neurons)):
                self.append_event(0, EventWithValue(0, t, 0), event_priority_if_input=10)

        # n_output_events_seen = {}
        while True:
            next_event = self.get_next_event()
            if next_event is None:  # or next_event[1].time > 10000: # TODO: remove after or
                break

            layer_idx, event = next_event
            layer = self.network.layers[layer_idx]

            out_events = layer.forward([event])
            # if layer_idx == 0 and event.time >= 32 and event.time < 40:
            # print(event.time, event.value, self.network.layers[0].neurons[3].ys, self.network.layers[0].neurons[3].ts)
            # print(layer_idx, event, out_events) # TODO: remove
            np.random.shuffle(out_events)  # shuffle output events to avoid bias
            if event.value != 0:
                event_order_log.append((layer_idx, event))
                # Track membrane potential changes
                membrane_changes = []
                for i_layer in range(len(self.network.layers)):
                    for i, neuron in enumerate(self.network.layers[i_layer].neurons):
                        membrane_changes.append((i_layer, EventWithValue(i, event.time, neuron.ys[-1])))
                membrane_event_log.append(membrane_changes)
            if layer_idx == len(self.network.layers) - 1:
                classification_events.extend(out_events)
                for classification_event in out_events:
                    if classification_event.value == 0:
                        continue
                    # Log event
                    event_order_log.append(((layer_idx) + 1, classification_event))
                    # Compute latency
                    origin_neuron = layer.neurons[classification_event.index]
                    if isinstance(origin_neuron, BarrierLIF):
                        # Any classification
                        if classification_event.time > self.last_classification_time:
                            self.classification_latencies.append(
                                #origin_neuron.on_event_barrier_event_count
                                self.true_spike_count_network_only
                            )
                            self.last_classification_time = classification_event.time
                        # Correct classification
                        if (
                            classification_event.index == true_label
                            and classification_event.time > self.last_correct_classification_time
                        ):
                            self.correct_classification_latencies.append(
                                #origin_neuron.on_event_barrier_event_count
                                self.true_spike_count_network_only
                            )
                            self.last_correct_classification_time = classification_event.time
            else:
                for out_event in out_events:
                    self.append_event(layer_idx + 1, out_event)

        # sort output events by timestamp
        classification_events.sort(key=lambda x: x.time)
        return classification_events, event_order_log, membrane_event_log


class EventDrivenInferenceEngine(BaseInferenceEngine):
    def __init__(self, static_params: StaticParameters):
        super().__init__(static_params)

    def build_network(
        self,
        params: EventDrivenInferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> BaseEventDrivenNetwork:
        return net_builder.build_event_driven_network(net_builder_params)

    def init_recorder(
        self,
        params: EventDrivenInferenceParameters,
        net_builder_params: BaseNetworkBuilderParameters,
    ):
        return EventRecorder(
            track_order_recordings=params.track_order_recordings,
            track_membrane_recordings=params.track_membrane_recordings,
            n_bins=params.n_order_histogram_bins,
            n_inputs=net_builder_params.get_n_inputs(),
        )

    def infer_sample(
        net: BaseEventDrivenNetwork,
        params: EventDrivenInferenceParameters,
        net_builder_params: BaseNetworkBuilderParameters,
        events: np.ndarray,  # shape [n_events, 4]
        target: int,
    ) -> EventBatchResult:
        converted_events = convert_tensor_to_named_tuple(
            events,
            params.sensor_size[0],
            params.sensor_size[1],
            params.timestep_size,
        )
        target_array = np.array([target], dtype=int)
        net.reset_state()
        engine = StatefulEventDrivenInferenceEngine(params, net)
        classification_events, event_order_recordings, membrane_event_log = engine.infer(converted_events, target)

        # Acquire membrane potential for classification (if needed)
        if params.loss_function_type == LossFunctionType.LAST_MEMBRANE:
            mem_potential_for_classification = [neuron.ys[-1] for neuron in net.layers[-1].neurons]
        elif params.loss_function_type == LossFunctionType.MAX_MEMBRANE:
            mem_potential_for_classification = [max(neuron.ys) for neuron in net.layers[-1].neurons]

        # Compute loss
        if params.loss_function_type == LossFunctionType.RATE or params.loss_function_type == LossFunctionType.COUNT:
            events_as_array = classification_events_to_array(
                classification_events,
                params.timestep_size,
                params.fix_sample_length_to,
                net_builder_params.get_neurons_per_layer()[-1],
            )
            loss = params.loss_function(events_as_array, target_array)
            inferred_label = events_as_array.sum(axis=0).argmax(axis=1)[0]
            output_recordings = classification_events_to_output_recordings(
                classification_events,
                params.timestep_size,
                params.fix_sample_length_to,
                net_builder_params.get_neurons_per_layer()[-1],
            )
        elif (
            params.loss_function_type == LossFunctionType.LAST_MEMBRANE
            or params.loss_function_type == LossFunctionType.MAX_MEMBRANE
        ):
            inferred_label = np.argmax(mem_potential_for_classification)
            mem_potential_for_classification_array = np.array([[mem_potential_for_classification]], dtype=float)
            loss = params.loss_function(mem_potential_for_classification_array, target_array)
            output_recordings = membrane_potential_recordings_to_output_recordings(
                [neuron.ts for neuron in net.layers[-1].neurons],
                [neuron.ys for neuron in net.layers[-1].neurons],
                params.timestep_size,
                params.fix_sample_length_to,
            )

        # get like output_recordings for membrane potential, but now for all layers
        # target shape torch.Tensor(n_event_steps, n_neurons)
        # such that for each step (triggered by an event), contains the membrane potential of each neuron
        # note that there is always one membrane potential change per event
        n_neurons = sum(net_builder_params.get_neurons_per_layer())
        n_input_neurons = net_builder_params.get_neurons_per_layer()[0]
        membrane_recordings = np.zeros((len(membrane_event_log), n_neurons), dtype=float)
        membrane_timesteps = np.zeros(len(membrane_event_log), dtype=int)
        for i_event, membrane_events in enumerate(membrane_event_log):
            if i_event > 0:
                previous_membrane_potentials = np.copy(membrane_recordings[i_event - 1])
            else:
                previous_membrane_potentials = np.zeros(n_neurons, dtype=float)
            for layer_idx, membrane_event in membrane_events:
                network_neuron_idx = membrane_event.index + sum(net_builder_params.get_neurons_per_layer()[:layer_idx])
                previous_membrane_potentials[network_neuron_idx] = membrane_event.value
            membrane_recordings[i_event] = previous_membrane_potentials
            membrane_timesteps[i_event] = membrane_events[0][1].time

        # per neuron, retrueve the input currents that were necessary to trigger a spike
        # then compute average number of input currents and absolute value of input currents
        n_currents_for_spike = np.zeros(n_neurons)
        excitatory_currents_for_spike = np.zeros(n_neurons)
        inhibitory_currents_for_spike = np.zeros(n_neurons)
        currents_histogram = np.zeros((n_neurons, 100))  # TODO: make this a parameter
        currents_for_spike_histogram = np.zeros((n_neurons, 100))  # TODO: make this a parameter
        for i_neuron, neuron in enumerate([neuron for layer in net.layers for neuron in layer.neurons]):
            if i_neuron < n_input_neurons:
                continue # First layer is skipped (this is not a great solution, but it is a quick fix)
            nonzero_external_input_amounts = [
                input_amount for input_amount in neuron.external_input_amounts if input_amount != 0
            ]
            currents_histogram_i, _ = np.histogram(nonzero_external_input_amounts, bins=100, range=(-0.5, 0.5))
            currents_histogram[i_neuron] += currents_histogram_i
            if len(neuron.threshold_timings) > 0:
                currents_for_spike_histogram[i_neuron] += currents_histogram_i
                for threshold_t in neuron.threshold_timings:
                    input_currents = []
                    for i, input_amount in enumerate(nonzero_external_input_amounts):
                        if input_amount != 0:
                            if (
                                neuron.external_input_timings[i] == threshold_t
                            ):  # these are before the spike, after the spike (in the same timestamp) the neuron is in refractory period
                                input_currents.append(input_amount)
                    # compute metrics
                    n_currents_for_spike[i_neuron] += len(input_currents)
                    excitatory_currents_for_spike[i_neuron] += sum(
                        [input_current for input_current in input_currents if input_current > 0]
                    )
                    inhibitory_currents_for_spike[i_neuron] += sum(
                        [input_current for input_current in input_currents if input_current < 0]
                    )
                n_currents_for_spike[i_neuron] /= len(neuron.threshold_timings)
                excitatory_currents_for_spike[i_neuron] /= len(neuron.threshold_timings)
                inhibitory_currents_for_spike[i_neuron] /= len(neuron.threshold_timings)
        # Quick conversion to new format
        n_currents_to_spike = n_currents_for_spike[n_currents_for_spike > 0].flatten().tolist()
        sum_excitatory_currents_to_spike = excitatory_currents_for_spike[n_currents_for_spike > 0].flatten().tolist()
        sum_inhibitory_currents_to_spike = inhibitory_currents_for_spike[n_currents_for_spike > 0].flatten().tolist()
        return EventBatchResult(
            output_recordings=output_recordings,
            event_order_recordings=event_order_recordings,
            membrane_recordings=[membrane_recordings],
            membrane_timesteps=[membrane_timesteps],
            timestep_size=params.timestep_size,
            max_n_timesteps=params.fix_sample_length_to,
            neurons_per_layer=net_builder_params.get_neurons_per_layer(),
            n_inputs=np.prod(params.sensor_size),
            loss=loss,
            accuracy=(inferred_label == target),
            batch_size=1,
            spike_counts=np.array(
                [engine.true_spike_count_network_only], dtype=int
            ),  # this is an array because it is expected to be for a batch (of >= 1 samples)
            priorities_given=engine.priorities_given,
            classification_latencies=engine.classification_latencies,
            correct_classification_latencies=engine.correct_classification_latencies,
            currents_histogram=currents_histogram,
            currents_for_spike_histogram=currents_for_spike_histogram,
            labels=np.array([target], dtype=int),
            n_currents_to_spike=n_currents_to_spike,
            sum_excitatory_currents_to_spike=sum_excitatory_currents_to_spike,
            sum_inhibitory_currents_to_spike=sum_inhibitory_currents_to_spike,
        )

    def infer_sample_wrapper(args):
        return EventDrivenInferenceEngine.infer_sample(*args)

    def infer_batch(
        self,
        net: Union[BaseEventDrivenNetwork, None],
        params: EventDrivenInferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
        data,  # must be of shape [batch_size, n_events, 4], where 4 = [x, y, polarity, timestamp]
        targets: torch.Tensor,  # shape [batch_size]
    ) -> Union[EventBatchResult, List[EventBatchResult]]:
        if params.batch_size == 1:
            assert net is not None
        else:
            assert net is None

        # Convert data to numpy, if it is a tensor
        # torch is unstable with multiprocessing
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data[0], torch.Tensor):
            data = [data[i].cpu().numpy() for i in range(len(data))]

        def build_net():
            if net is None:
                return self.build_network(
                    params, net_builder, net_builder_params
                )  # rebuild network for every batch to allow parallel inference
            else:
                return net  # for batch size == 1, use the same network for every batch

        batch_args = [
            (build_net(), params, net_builder_params, data[i], targets[i].item()) for i in range(params.batch_size)
        ]

        if params.batch_size > 1:
            results = []
            with ProcessPoolExecutor(max_workers=params.batch_size) as pool:
                futures = []
                for batch_arg in batch_args:
                    future = pool.submit(EventDrivenInferenceEngine.infer_sample_wrapper, batch_arg)
                    futures.append(future)
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            del batch_args
            gc.collect()
            return results
        else:
            return EventDrivenInferenceEngine.infer_sample(*batch_args[0])
