from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Tuple


PLOT_LOSS_KEY = "l"
QUIT_KEY = "q"


# inference network extensions
# given as: "extension1,extension2-X,extension3", where X may be some variable
class NetworkExtension(ABC):
    str_repr: str

    def __init__(self, s):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def is_(s: str):
        pass


class BarrierLIFNetworkExtension(NetworkExtension):
    str_repr = "barrier_lif"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.barrier_size = int(
                split[1]
            )  # How many events (spikes or barrier messages) are required to trigger a spike or barrier message
        else:
            raise ValueError("BarrierLIFNetworkExtension requires barrier size to be set")
        if len(split) >= 3:
            self.send_barrier_events = bool(
                int(split[2])
            )  # Whether to send barrier messages. If False, only the spikes themselves are barrier messages.
        else:
            self.send_barrier_events = True

    def __str__(self):
        base_str = f"{BarrierLIFNetworkExtension.str_repr}/{self.barrier_size}"
        if not self.send_barrier_events:
            base_str += f"/{int(self.send_barrier_events)}"
        return base_str

    def is_(s: str):
        return s.startswith(BarrierLIFNetworkExtension.str_repr)


class InputBarrierNetworkExtension(NetworkExtension):
    str_repr = "input_barrier"

    def __str__(self):
        return f"{InputBarrierNetworkExtension.str_repr}"

    def is_(s: str):
        return s.startswith(InputBarrierNetworkExtension.str_repr)


class PreProcessingNetworkExtension(NetworkExtension):
    str_repr = "pp"

    def __str__(self):
        return f"{PreProcessingNetworkExtension.str_repr}"

    def is_(s: str):
        return s.startswith(PreProcessingNetworkExtension.str_repr)


class WTANetworkExtension(NetworkExtension):
    str_repr = "wta"

    def __str__(self):
        return f"{WTANetworkExtension.str_repr}"

    def is_(s: str):
        return s.startswith(WTANetworkExtension.str_repr)

    
class NoRefracNetworkExtension(NetworkExtension):
    str_repr = "no_refrac"

    def __str__(self):
        return f"{NoRefracNetworkExtension.str_repr}"

    def is_(s: str):
        return s.startswith(NoRefracNetworkExtension.str_repr)
    
class LogSpikeDynamicsNetworkExtension(NetworkExtension):
    str_repr = "log_spike_dynamics"

    def __str__(self):
        return f"{LogSpikeDynamicsNetworkExtension.str_repr}"

    def is_(s: str):
        return s.startswith(LogSpikeDynamicsNetworkExtension.str_repr)

    
class StopOnOutputNetworkExtension(NetworkExtension):
    str_repr = "stop_on_output"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.max_post_output_steps = int(split[1])
        else:
            raise ValueError("StopOnOutputNetworkExtension requires max_post_output_steps to be set")

    def __str__(self):
        return f"{StopOnOutputNetworkExtension.str_repr}/{self.max_post_output_steps}"

    def is_(s: str):
        return s.startswith(StopOnOutputNetworkExtension.str_repr)


class EarlyAccumulateNetworkExtension(NetworkExtension):
    str_repr = "early_accumulate"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.K = int(split[1])
        else:
            raise ValueError("EarlyAccumulateNetworkExtension requires K to be set")

    def __str__(self):
        return f"{EarlyAccumulateNetworkExtension.str_repr}/{self.K}"

    def is_(s: str):
        return s.startswith(EarlyAccumulateNetworkExtension.str_repr)


class InputDropoutNetworkExtension(NetworkExtension):
    str_repr = "input_dropout"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.dropout = float(split[1])
        else:
            raise ValueError("InputDropoutNetworkExtension requires dropout to be set")

    def __str__(self):
        return f"{InputDropoutNetworkExtension.str_repr}/{self.dropout}"

    def is_(s: str):
        return s.startswith(InputDropoutNetworkExtension.str_repr)

class NetworkDropoutNetworkExtension(NetworkExtension):
    str_repr = "network_dropout"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.dropout = float(split[1])
        else:
            raise ValueError("NetworkDropoutNetworkExtension requires dropout to be set")

    def __str__(self):
        return f"{NetworkDropoutNetworkExtension.str_repr}/{self.dropout}"

    def is_(s: str):
        return s.startswith(NetworkDropoutNetworkExtension.str_repr)


class FullDropoutNetworkExtension(NetworkExtension):
    str_repr = "full_dropout"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.dropout = float(split[1])
        else:
            raise ValueError("FullDropoutNetworkExtension requires dropout to be set")

    def __str__(self):
        return f"{FullDropoutNetworkExtension.str_repr}/{self.dropout}"

    def is_(s: str):
        return s.startswith(FullDropoutNetworkExtension.str_repr)
    
class RefractoryDropoutNetworkExtension(NetworkExtension):
    str_repr = "refrac_dropout"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.dropout = float(split[1])
        else:
            raise ValueError("RefractoryDropoutNetworkExtension requires dropout to be set")

    def __str__(self):
        return f"{RefractoryDropoutNetworkExtension.str_repr}/{self.dropout}"

    def is_(s: str):
        return s.startswith(RefractoryDropoutNetworkExtension.str_repr)


class PositiveSkewDropoutNetworkExtension(NetworkExtension):
    str_repr = "positive_skew_dropout"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 3:
            self.slope = float(split[1])
            self.threshold = float(split[2])
        else:
            raise ValueError("PositiveSkewDropoutNetworkExtension requires slope and threshold to be set")

    def __str__(self):
        return f"{PositiveSkewDropoutNetworkExtension.str_repr}/{self.slope}/{self.threshold}"

    def is_(s: str):
        return s.startswith(PositiveSkewDropoutNetworkExtension.str_repr)


class NegativeSkewDropoutNetworkExtension(NetworkExtension):
    str_repr = "negative_skew_dropout"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 3:
            self.slope = float(split[1])
            self.threshold = float(split[2])
        else:
            raise ValueError("NegativeSkewDropoutNetworkExtension requires slope and threshold to be set")

    def __str__(self):
        return f"{NegativeSkewDropoutNetworkExtension.str_repr}/{self.slope}/{self.threshold}"

    def is_(s: str):
        return s.startswith(NegativeSkewDropoutNetworkExtension.str_repr)


class FineTuneNetworkExtension(NetworkExtension):
    str_repr = "fine_tune"
    # "Fine tune" a model by loading its parameters from a file, and then training it again.

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.source_model_id = split[1]
        else:
            raise ValueError("FineTuneNetworkExtension requires source model id to be set")

    def __str__(self):
        return f"{FineTuneNetworkExtension.str_repr}/{self.source_model_id}"

    def is_(s: str):
        return s.startswith(FineTuneNetworkExtension.str_repr)


class SpikeSparsityLossNetworkExtension(NetworkExtension):
    str_repr = "spike_sparsity_loss"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.Lambda = float(split[1])
        else:
            raise ValueError("SpikeSparsityLossNetworkExtension requires Lambda to be set")

    def __str__(self):
        return f"{SpikeSparsityLossNetworkExtension.str_repr}/{self.Lambda}"

    def is_(s: str):
        return s.startswith(SpikeSparsityLossNetworkExtension.str_repr)


class EarlyStoppingNetworkExtension(NetworkExtension):
    str_repr = "early_stopping"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.n_steps = float(split[1])
        else:
            raise ValueError("EarlyStoppingNetworkExtension requires n_steps to be set")

    def __str__(self):
        return f"{EarlyStoppingNetworkExtension.str_repr}/{int(self.n_steps)}"

    def is_(s: str):
        return s.startswith(EarlyStoppingNetworkExtension.str_repr)

class AsyncTorchMode(Enum):
    RANDOM = 1
    MEMBRANE_POTENTIAL = 2
    OUTGOING_CURRENT = 3
    NEURON_INDEX = 4

    def from_string(s: str):
        if s == "random":
            return AsyncTorchMode.RANDOM
        elif s == "membrane_potential":
            return AsyncTorchMode.MEMBRANE_POTENTIAL
        elif s == "outgoing_current":
            return AsyncTorchMode.OUTGOING_CURRENT
        elif s == "neuron_index":
            return AsyncTorchMode.NEURON_INDEX
        else:
            raise ValueError("Unknown async torch mode")

class AsyncTorchNetworkExtension(NetworkExtension):
    str_repr = "async_torch"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 3:
            self.forward_group_size = int(split[1])
            self.mode = AsyncTorchMode.from_string(split[2])
        else:
            raise ValueError("AsyncTorchNetworkExtension requires forward group size and mode to be set")

    def __str__(self):
        return f"{AsyncTorchNetworkExtension.str_repr}/{self.forward_group_size}/{self.mode.name.lower()}"

    def is_(s: str):
        return s.startswith(AsyncTorchNetworkExtension.str_repr)


class SyncThresholdNetworkExtension(NetworkExtension):
    str_repr = "sync_threshold"
    # Examples:
    # sync_threshold/1 fixes the threshold to 1, no training
    # sync_threshold/1/0/1 fixes the threshold to 1, trainable
    # sync_threshold/1/0.1/1 samples the threshold from a normal distribution with mean 1 and std 0.1, trainable
    # sync_threshold/1/0.1/0 is invalid (sampling is only possible if trainable is True)

    # Note for inference testing barrier_lif/0/0 can be used to load sync threshold from file. Barrier message will not be sent (second 0 param).
    # barrier_lif/1/0 will fix sync threshold to 1, no loading from file.

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.mean = float(split[1])
        else:
            raise ValueError("TrainableSyncParamsNetworkExtension requires a (fixed) value for mean")
        if len(split) >= 3:
            self.std = float(split[2])
        else:
            self.std = 0.0
        if len(split) >= 4:
            self.trainable = bool(int(split[3]))
        else:
            self.trainable = False
            if self.std != 0:
                print(
                    "WARNING: SyncThresholdNetworkExtension: std is not 0, but trainable is False. This is probably not what you want."
                )
        if len(split) >= 5:
            self.trainable_input_layer = bool(int(split[4]))
        else:
            self.trainable_input_layer = False

    def __str__(self):
        return f"{SyncThresholdNetworkExtension.str_repr}/{self.mean}/{self.std}/{int(self.trainable)}/{int(self.trainable_input_layer)}"

    def is_(s: str):
        return s.startswith(SyncThresholdNetworkExtension.str_repr)

class MomentumNoiseNetworkExtension(NetworkExtension):
    str_repr = "momentum_noise"

    def __init__(self, s: str):
        super().__init__(s)
        split = s.split("/")
        if len(split) >= 2:
            self.lambda_ = float(split[1])
        else:
            raise ValueError("MomentumNoiseNetworkExtension requires lambda to be set")

    def __str__(self):
        return f"{MomentumNoiseNetworkExtension.str_repr}/{self.lambda_}"

    def is_(s: str):
        return s.startswith(MomentumNoiseNetworkExtension.str_repr)

def network_extension_from_string(s: str) -> NetworkExtension:
    if BarrierLIFNetworkExtension.is_(s):
        return BarrierLIFNetworkExtension(s)
    elif PreProcessingNetworkExtension.is_(s):
        return PreProcessingNetworkExtension(s)
    elif WTANetworkExtension.is_(s):
        return WTANetworkExtension(s)
    elif EarlyAccumulateNetworkExtension.is_(s):
        return EarlyAccumulateNetworkExtension(s)
    elif InputDropoutNetworkExtension.is_(s):
        return InputDropoutNetworkExtension(s)
    elif NetworkDropoutNetworkExtension.is_(s):
        return NetworkDropoutNetworkExtension(s)
    elif FullDropoutNetworkExtension.is_(s):
        return FullDropoutNetworkExtension(s)
    elif PositiveSkewDropoutNetworkExtension.is_(s):
        return PositiveSkewDropoutNetworkExtension(s)
    elif NegativeSkewDropoutNetworkExtension.is_(s):
        return NegativeSkewDropoutNetworkExtension(s)
    elif FineTuneNetworkExtension.is_(s):
        return FineTuneNetworkExtension(s)
    elif AsyncTorchNetworkExtension.is_(s):
        return AsyncTorchNetworkExtension(s)
    elif SyncThresholdNetworkExtension.is_(s):
        return SyncThresholdNetworkExtension(s)
    elif InputBarrierNetworkExtension.is_(s):
        return InputBarrierNetworkExtension(s)
    elif SpikeSparsityLossNetworkExtension.is_(s):
        return SpikeSparsityLossNetworkExtension(s)
    elif RefractoryDropoutNetworkExtension.is_(s):
        return RefractoryDropoutNetworkExtension(s)
    elif NoRefracNetworkExtension.is_(s):
        return NoRefracNetworkExtension(s)
    elif MomentumNoiseNetworkExtension.is_(s):
        return MomentumNoiseNetworkExtension(s)
    elif EarlyStoppingNetworkExtension.is_(s):
        return EarlyStoppingNetworkExtension(s)
    elif StopOnOutputNetworkExtension.is_(s):
        return StopOnOutputNetworkExtension(s)
    elif LogSpikeDynamicsNetworkExtension.is_(s):
        return LogSpikeDynamicsNetworkExtension(s)
    else:
        raise ValueError("Unknown network extension")


def network_extensions_from_string(s: str) -> List[NetworkExtension]:
    if s == "" or s == "nan":
        return []
    s_split = s.split(",")
    lst = []
    for s in s_split:
        lst.append(network_extension_from_string(s))
    return lst


def network_extensions_to_string(extensions: List[NetworkExtension], delimiter=",") -> str:
    if len(extensions) == 0:
        return ""
    s = ""
    extensions = sorted(extensions, key=lambda x: str(x))
    for extension in extensions:
        s += f"{str(extension)}" + delimiter
    return s[:-1]


# pass a class, not an instance
def is_network_extension_type_in_list(extensions: List[NetworkExtension], extension_type) -> bool:
    for extension in extensions:
        if isinstance(extension, extension_type):
            return True
    return False


# pass a class, not an instance
# returns the first extension of the given type, or None if none exists
def get_network_extension_from_list(extensions: List[NetworkExtension], extension_type) -> NetworkExtension:
    for extension in extensions:
        if isinstance(extension, extension_type):
            return extension
    return None


class NetworkBackend(Enum):
    SNNTORCH = 2


class EventOrderingMethod(Enum):
    NONE = 0  # no sorting, events are inserted at random position
    PER_LAYER = 1  # sort events per layer
    STRICT_PER_LAYER = 2  # sort events per layer, also sort events within a layer
    PER_HALF = 3  # sort events per half of the network

    def from_string(s: str):
        if s == "random" or s == "none":
            return EventOrderingMethod.NONE
        elif s == "layer" or s == "per_layer":
            return EventOrderingMethod.PER_LAYER
        elif s == "half" or s == "per_half":
            return EventOrderingMethod.PER_HALF
        elif s == "strict_layer" or s == "strict_per_layer":
            return EventOrderingMethod.STRICT_PER_LAYER
        else:
            raise ValueError("Unknown event ordering method")

    def to_simple_label(self):
        if self == EventOrderingMethod.NONE:
            return "N"
        elif self == EventOrderingMethod.PER_LAYER:
            return "L"
        elif self == EventOrderingMethod.PER_HALF:
            return "H"
        elif self == EventOrderingMethod.STRICT_PER_LAYER:
            return "SL"
        else:
            raise ValueError("Unknown event ordering method")


class EventSchedulingMethod(Enum):
    SPIKE_COUNT = 1
    RANDOM_PRIORITY = 2
    DISTORTED_SPIKE_COUNT = 3
    RANDOM_SAMPLING = 4
    EVENT_CURRENT = 5

    def from_string(s: str):
        if s == "spike_count":
            return EventSchedulingMethod.SPIKE_COUNT
        elif s == "random":
            return EventSchedulingMethod.RANDOM_PRIORITY
        elif s == "distorted_spike_count":
            return EventSchedulingMethod.DISTORTED_SPIKE_COUNT
        elif s == "random_sampling":
            return EventSchedulingMethod.RANDOM_SAMPLING
        elif s == "event_current":
            return EventSchedulingMethod.EVENT_CURRENT
        else:
            raise ValueError("Unknown event scheduling method")


class InferenceMethod(Enum):
    MATRIX = 1
    LOIHI2 = 2
    EVENT = 3

    def from_string(s: str):
        if s == "matrix":
            return InferenceMethod.MATRIX
        elif s == "loihi":
            return InferenceMethod.LOIHI2
        elif s == "event":
            return InferenceMethod.EVENT
        else:
            raise ValueError("Unknown inference method")
        
    def __str__(self):
        if self == InferenceMethod.MATRIX:
            return "matrix"
        elif self == InferenceMethod.LOIHI2:
            return "loihi"
        elif self == InferenceMethod.EVENT:
            return "event"
        else:
            raise ValueError("Unknown inference method")


class DatasetProvider(Enum):
    TONIC_FRAME = 1
    TONIC_EVENTS = 3


class LossFunctionType(Enum):
    RATE = 1
    TEMPORAL = 2
    LAST_MEMBRANE = 3
    MAX_MEMBRANE = 4
    COUNT = 5

    def from_string(s: str):
        if s == "rate":
            return LossFunctionType.RATE
        elif s == "temporal":
            return LossFunctionType.TEMPORAL
        elif s == "last_membrane":
            return LossFunctionType.LAST_MEMBRANE
        elif s == "max_membrane":
            return LossFunctionType.MAX_MEMBRANE
        elif s == "count":
            return LossFunctionType.COUNT
        else:
            raise ValueError("Unknown loss function type")

class OptimizerType(Enum):
    ADAM = 1
    ADAMAX = 2

    def from_string(s: str):
        if s == "adam":
            return OptimizerType.ADAM
        elif s == "adamax":
            return OptimizerType.ADAMAX
        else:
            raise ValueError("Unknown optimizer type")