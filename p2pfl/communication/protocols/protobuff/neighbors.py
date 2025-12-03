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

"""Protocol agnostic neighbor management."""

import threading
import time
from collections.abc import Callable

from p2pfl.communication.protocols.exceptions import NeighborNotConnectedError
from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.management.logger import logger
from p2pfl.utils.node_component import NodeComponent


class Neighbors(NodeComponent):
    """Neighbor management class for agnostic communication protocol."""

    def __init__(self, build_client_fn: Callable[..., ProtobuffClient]) -> None:
        """Initialize the neighbor management class."""
        self.neis: dict[str, tuple[ProtobuffClient, float]] = {}
        self.neis_lock = threading.Lock()
        self.build_client_fn = build_client_fn
        self.nei_stats: dict[str, dict[str, float | int]] = {}

    def refresh_or_add(self, addr: str, time: float) -> None:
        """
        Refresh or add a neighbor.

        Args:
            addr: Address of the neighbor.
            time: Time of the last heartbeat.

        """
        with self.neis_lock:
            exist_nei = addr in self.neis
            if exist_nei:
                self.neis[addr] = (
                    self.neis[addr][0],
                    time,
                )
        if not exist_nei:
            self.add(addr, non_direct=True)

    def add(self, addr: str, non_direct: bool = False, handshake: bool = True) -> bool:
        """
        Add a neighbor to the neighbors list.

        Args:
            addr: Address of the neighbor to add.
            non_direct: Flag to add a non-direct neighbor.
            handshake: Flag to perform a handshake.

        Returns:
            True if the neighbor was added, False otherwise.

        """
        if addr == self.addr:
            logger.info(self.addr, "❌ Cannot add itself")
            return False

        with self.neis_lock:
            if self.exists(addr, only_direct=True):
                logger.info(self.addr, f"❌ Cannot add duplicates. {addr} already exists.")
                logger.debug(self.addr, f"Current neighbors: {self.neis.keys()}")
                return False

            try:
                client = self.build_client_fn(self.addr, addr)
                if not non_direct:
                    client.connect(handshake_msg=handshake)
                self.neis[addr] = (client, time.time())
                self.nei_stats[addr] = {"degree": 0}
            except Exception as e:
                logger.error(self.addr, f"❌ Cannot add {addr}: {e}")
                return False

            return True

    def remove(self, addr: str, disconnect_msg: bool = True) -> None:
        """
        Remove a neighbor from the neighbors list.

        Be careful, this method does not close the connection, is agnostic to the connection state.

        Args:
            addr: Address of the neighbor to remove.
            disconnect_msg: If a disconnect message is needed.

        """
        with self.neis_lock:
            if addr in self.neis:
                if self.neis[addr][0].is_connected() and not self.neis[addr][0].has_temporal_connection():
                    self.neis[addr][0].disconnect(disconnect_msg=disconnect_msg)
                del self.neis[addr]
                if addr in self.nei_stats:
                    del self.nei_stats[addr]

    def get(self, addr: str) -> ProtobuffClient:
        """
        Get a neighbor from the neighbors list.

        Args:
            addr: Address of the neighbor to get.

        Returns:
            The neighbor.

        """
        with self.neis_lock:
            try:
                return self.neis[addr][0]
            except KeyError:
                raise NeighborNotConnectedError(f"Neighbor {addr} not connected") from None

    def get_all(self, only_direct: bool = False) -> dict[str, tuple[ProtobuffClient, float]]:
        """
        Get all neighbors from the neighbors list.

        Args:
            only_direct: Flag to get only direct neighbors.

        """
        neis = self.neis.copy()
        if only_direct:
            return {k: v for k, v in neis.items() if v[0].is_connected() and not v[0].has_temporal_connection()}
        return neis

    def exists(self, addr: str, only_direct: bool = False) -> bool:
        """
        Check if a neighbor exists in the neighbors list.

        Args:
            addr: Address of the neighbor to check.
            only_direct: Flag to check only direct neighbors.

        """
        if only_direct:
            return addr in self.neis and self.neis[addr][0].is_connected() and not self.neis[addr][0].has_temporal_connection()
        return addr in self.neis

    def clear_neighbors(self) -> None:
        """Clear all neighbors."""
        while len(self.neis) > 0:
            self.remove(list(self.neis.keys())[0])

    def note_interaction(self, addr: str) -> None:
        if addr in self.nei_stats:
            deg = self.nei_stats[addr].get("degree", 0)
            self.nei_stats[addr]["degree"] = deg + 1

    def get_degree(self, addr: str) -> int:
        if addr in self.nei_stats:
            return int(self.nei_stats[addr].get("degree", 0))
        return 0

