# Peer-to-Peer Federated Learning Framework with Improved Communication Strategies and Failure Recovery Mechanism

This repository implements a peer-to-peer federated learning framework with enhanced communication protocols and failure recovery mechanisms. The framework is built on top of the [P2PFL library](https://github.com/p2pfl/p2pfl), extending it with intelligent communication strategies and robust fault tolerance capabilities.

## Overview

Building on the P2PFL framework, we extend it with improved communication and failure recovery mechanisms for practical deployment. This framework introduces two key enhancements:

- **Smart Communication Protocol** - A smarter gossip protocol that attaches version information to model updates, applies staleness weighted aggregation, maintains neighbor model caches, and steers propagation based on interaction frequency. These designs allow decentralized training to better handle asynchronous participants and heterogeneous update quality.

- **Failure Recovery Mechanism** - A recovery mechanism combining local and remote checkpoints, enabling failed nodes to rejoin training and restore model state. This enhancement improves training continuity and system reliability.

## Branches and Code Organization

The project is organized into three branches, each containing different implementations and experimental evaluations:

- **`main`** - Baseline implementation and experiments
- **`communication`** - Smart communication protocol implementation and convergence improvement experiments
- **`fault_tolerance`** - Failure recovery mechanism implementation and effectiveness experiments

### `main` - Baseline Implementation

The `main` branch contains the baseline implementation using the standard P2PFL library. This branch serves as a reference point for comparing the enhancements implemented in other branches.

### `communication` - Smart Communication Protocol

**All communication protocol related code is located in the `communication` branch.**

The `communication` branch implements a smarter gossip protocol that enhances decentralized training by better handling asynchronous participants and heterogeneous update quality. 

**All failure recovery related code is located in the `fault_tolerance` branch.**

The `fault_tolerance` branch implements a comprehensive recovery mechanism that combines local and remote checkpoints, enabling failed nodes to seamlessly rejoin training and restore their model state. This enhancement significantly improves training continuity and system reliability.

## Built on P2PFL Library

This framework is built on the P2PFL library, a general-purpose open-source library for executing decentralized federated learning systems using peer-to-peer networks and gossip protocols.

**Reference:**

> P2PFL: Peer-to-peer federated learning framework. (2024). GitHub repository. https://github.com/p2pfl/p2pfl

## Project Contributors

- [@lsnnnnnnnn](https://github.com/lsnnnnnnnn)
- [@zhangyunzhen2027](https://github.com/zhangyunzhen2027)
- [@sophia-mathcs](https://github.com/sophia-mathcs)


