# Communication-Layer Enhancements

A smarter **communication layer** on top of the original `p2pfl` framework, without changing the learning API or the FedAvg aggregator interface.

## Overview

We extend the original P2PFL system with the following capabilities:

- **Versioned model updates**  
  Every model update carries a `version` field (currently set to the global training round).

- **Staleness-aware FedAvg**  
  When a node receives a model update, its contribution to FedAvg is down-weighted if the update is stale (old version).

- **Neighbor model cache**  
  Each node caches the latest model received from every neighbor in its `NodeState`. This enables future **similarity-based neighbor selection**.

- **Degree-based preferential gossip**  
  The gossip protocol now tracks how often it communicates with each neighbor and uses this **interaction degree** to bias neighbor selection (a simple preferential-attachment-like behavior).

All of this happens **inside the communication layer** (commands + gossiper + neighbors + stages). The learning workflow and aggregator APIs remain unchanged.

---

## Key Features

### 1. Versioned Updates

- Before gossiping a model, the node sets:

  ```python
  model.additional_info["version"] = state.round


