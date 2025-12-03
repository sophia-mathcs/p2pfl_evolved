#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""PartialModelCommand command."""

from collections.abc import Callable

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.pre_send_model_command import PreSendModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class PartialModelCommand(Command):
    """PartialModelCommand."""

    def __init__(
        self,
        state: NodeState,
        stop: Callable[[], None],
        aggregator: Aggregator,
        comm_proto: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        """Initialize PartialModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.communication_protocol = comm_proto
        self.laerner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "partial_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: bytes | None = None,
        contributors: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None or contributors is None or num_samples is None:
            raise ValueError("Weights, contributors and weight are required")

        if self.state.round is not None:
            if round != self.state.round:
                logger.debug(
                    self.state.addr,
                    f"Model reception in a late round ({round} != {self.state.round}).",
                )
                return

            if len(self.state.train_set) == 0:
                logger.error(self.state.addr, "Model Reception when there is no trainset")
                return

            try:
                base_model = self.laerner.get_model()
                if isinstance(weights, (bytes, bytearray)):
                    params, extra_info = base_model.decode_parameters(weights)
                else:
                    params = weights
                    extra_info = {}
                if extra_info is None:
                    extra_info = {}
                current_round = self.state.round if self.state.round is not None else round
                if "version" not in extra_info:
                    extra_info["version"] = current_round
                version = extra_info["version"]
                try:
                    staleness = int(current_round) - int(version)
                except Exception:
                    staleness = 0
                if staleness < 0:
                    staleness = 0
                try:
                    raw_weight_float = float(num_samples)
                except Exception:
                    raw_weight_float = 1.0
                decay = 1.0 / (1.0 + float(staleness))
                effective_weight = int(raw_weight_float * decay)
                if effective_weight <= 0:
                    effective_weight = 1

                model = base_model.build_copy(
                    params=params,
                    num_samples=effective_weight,
                    contributors=list(contributors),
                    additional_info=extra_info,
                )
                self.state.neighbor_models[source] = model

                models_added = self.aggregator.add_model(model)
                if models_added != []:
                    self.communication_protocol.broadcast(
                        self.communication_protocol.build_msg(
                            ModelsAggregatedCommand.get_name(),
                            models_added,
                            round=self.state.round,
                        )
                    )
                else:
                    PreSendModelCommand.remove_hashed(self.state, self.get_name(), contributors, self.state.round)

            except DecodingParamsError:
                logger.error(self.state.addr, "Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "Models not matching.")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"Unknown error adding model: {e}")
                self.stop()

        else:
            logger.debug(self.state.addr, "Tried to add a model while learning is not running")
