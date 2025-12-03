#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Gossip model stage."""

from typing import Any

from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class GossipModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    def name() -> str:
        """Return the name of the stage."""
        return "GossipModelStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        learner: Learner | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if (
            state is None
            or aggregator is None
            or communication_protocol is None
            or learner is None
        ):
            raise Exception("Invalid parameters on GossipModelStage.")

        GossipModelStage.__gossip_model_difusion(state, communication_protocol, learner)
        return StageFactory.get_stage("RoundFinishedStage")

    @staticmethod
    def __gossip_model_difusion(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        """Diffuse the aggregated model using the gossip protocol."""
        logger.info(state.addr, "🗣️ Gossiping aggregated model.")
        fixed_round = state.round
        if fixed_round is None:
            raise Exception("Learner not initialized")

        # A node is a candidate if it has not yet received the model for this round
        def candidate_condition(node: str) -> bool:
            return state.nei_status[node] < fixed_round

        # Base set of candidates (used by both get_candidates_fn and status_fn)
        def base_candidates() -> list[str]:
            return [
                n
                for n in communication_protocol.get_neighbors(only_direct=True)
                if candidate_condition(n)
            ]

        # Function used by the gossiper to know who to contact next
        def get_candidates_fn() -> list[str]:
            """
            Preferential attachment on top of the original condition:
            among nodes that have not yet received the model, those with
            higher observed degree (more past interactions) are contacted
            first.
            """
            candidates = base_candidates()
            if not candidates:
                return candidates

            def score(addr: str) -> int:
                # Some CommunicationProtocol implementations may not implement
                # get_neighbor_degree; fall back to 0 in that case.
                get_deg = getattr(communication_protocol, "get_neighbor_degree", None)
                if callable(get_deg):
                    try:
                        return int(get_deg(addr))
                    except Exception:
                        return 0
                return 0

            # Higher degree first
            candidates.sort(key=score, reverse=True)
            return candidates

        # Function used only for monitoring / status
        def status_fn() -> Any:
            return base_candidates()

        # Build the message to send to a given neighbor
        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized")

            # 1. Get current model
            model = learner.get_model()

            # 2. Ensure additional_info exists and write version = current round
            info = getattr(model, "additional_info", None)
            if info is None:
                info = {}
            model.additional_info = info
            model.additional_info["version"] = state.round

            # 3. Serialize parameters (including additional_info["version"])
            encoded_model = model.encode_parameters()

            # 4. Build gossip weights message
            return (
                communication_protocol.build_weights(
                    FullModelCommand.get_name(),
                    state.round,
                    encoded_model,
                ),
                FullModelCommand.get_name(),
                state.round,
                [str(state.round)],
            )

        # Gossip
        communication_protocol.gossip_weights(
            lambda: check_early_stop(state, raise_exception=False),
            get_candidates_fn,
            status_fn,
            model_fn,
        )
