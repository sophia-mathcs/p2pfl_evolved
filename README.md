# Adaptive Communication Protocol for P2P Federated Learning

## Overview

This project extends the original [P2PFL](https://github.com/p2pfl/p2pfl) framework with an **adaptive communication protocol** that intelligently manages model synchronization in peer-to-peer federated learning networks. The adaptive protocol enhances communication efficiency and model quality by dynamically adjusting communication strategies based on network conditions, model staleness, and neighbor interaction patterns.

## Key Features
c
### 1. Versioned Model Updates

Every model update in the network carries a **version field** that tracks the global training round. This enables nodes to:

- Track the freshness of received models
- Identify stale updates that may degrade model quality
- Make informed decisions about which models to prioritize

**Implementation:**
```python
# Model version is automatically set before gossiping
model.additional_info["version"] = state.round
```

### 2. Staleness-Aware Federated Averaging

The protocol implements **staleness-aware aggregation** that automatically down-weights contributions from stale model updates. This prevents outdated models from negatively impacting the global model convergence.

**Staleness Calculation:**
```python
staleness = current_round - model_version
decay_factor = 1.0 / (1.0 + staleness)
effective_weight = original_weight * decay_factor
```

**Benefits:**
- Maintains model quality even when nodes have varying update frequencies
- Reduces the impact of network delays and node failures
- Improves convergence stability in heterogeneous networks

### 3. Neighbor Model Cache

Each node maintains a **cache of the latest model** received from every neighbor in its `NodeState`. This enables:

- **Similarity-based neighbor selection** (future enhancement)
- Efficient model comparison and selection
- Reduced redundant model transfers
- Better understanding of network topology and model distribution

**Storage:**
```python
state.neighbor_models[neighbor_address] = cached_model
```

### 4. Degree-Based Preferential Gossip

The gossip protocol implements **preferential attachment** behavior by tracking interaction degrees with neighbors. Nodes prioritize communication with neighbors they have interacted with more frequently, creating a self-organizing network structure.

**How It Works:**
1. Each node tracks the number of interactions (degree) with each neighbor
2. When selecting neighbors for gossip, nodes with higher degrees are prioritized
3. This creates a preferential attachment pattern similar to scale-free networks

**Implementation:**
```python
def get_candidates_fn() -> list[str]:
    candidates = base_candidates()
    # Sort by interaction degree (higher first)
    candidates.sort(key=lambda addr: get_neighbor_degree(addr), reverse=True)
    return candidates
```

**Benefits:**
- Faster model propagation through well-connected nodes
- Natural formation of efficient communication paths
- Improved resilience to node failures
- Better utilization of high-bandwidth connections

## Architecture

The adaptive communication protocol is implemented **entirely within the communication layer**, without modifying the learning API or aggregator interfaces. This ensures:

- **Backward compatibility** with existing P2PFL code
- **Modularity** - communication enhancements are isolated
- **Flexibility** - easy to extend or modify

### Component Structure

```
Communication Layer
├── Commands
│   ├── FullModelCommand (staleness-aware aggregation)
│   └── PartialModelCommand (staleness-aware aggregation)
├── Protocols
│   └── ProtobuffCommunicationProtocol
│       ├── Neighbors (degree tracking)
│       └── Gossiper (preferential selection)
└── Stages
    └── GossipModelStage (versioned updates, preferential gossip)
```

## How It Works

### Model Synchronization Flow

1. **Model Training**: Each node trains its local model on its dataset
2. **Version Assignment**: Before gossiping, the model is tagged with the current round number
3. **Preferential Selection**: The gossip protocol selects neighbors based on interaction degree
4. **Model Propagation**: Models are propagated through the network using the gossip protocol
5. **Staleness Detection**: Upon receipt, each node calculates the staleness of the received model
6. **Weighted Aggregation**: Stale models are down-weighted during FedAvg aggregation
7. **Cache Update**: The received model is cached for future reference

### Example: Staleness-Aware Aggregation

Consider a scenario where:
- Current round: 10
- Node A sends a model from round 8 (staleness = 2)
- Node B sends a model from round 10 (staleness = 0)

The aggregation weights would be:
- Node A: `weight_A * (1.0 / (1.0 + 2)) = weight_A * 0.33`
- Node B: `weight_B * (1.0 / (1.0 + 0)) = weight_B * 1.0`

This ensures fresher models have more influence on the aggregated result.

## Usage

The adaptive communication protocol is **automatically enabled** when using the standard P2PFL node initialization. No additional configuration is required.

### Basic Usage

```python
from p2pfl.node import Node
from p2pfl.learning.frameworks.frameworks import PyTorchLearner
from p2pfl.learning.dataset.p2pfl_dataset import MnistFederatedDM
from p2pfl.examples.mnist.model.mlp import MLP

# Create a node with adaptive communication
node = Node(
    model=MLP(),
    data=MnistFederatedDM(),
    learner=PyTorchLearner(),
)

# Start the node
node.start()

# Connect to other nodes
node.connect("127.0.0.1:6667")

# Start learning (adaptive protocol handles communication automatically)
node.set_start_learning(rounds=10, epochs=1)
```

### Monitoring Adaptive Behavior

The protocol logs key events that help monitor adaptive behavior:

- `🗣️ Gossiping aggregated model` - Model propagation with versioning
- `🧩 Model added (X/Y) from [node]` - Model reception with staleness handling
- `🧠 Aggregating models` - Staleness-aware aggregation in progress

## Performance Benefits

### Communication Efficiency

- **Reduced Redundancy**: Neighbor model cache prevents unnecessary transfers
- **Faster Convergence**: Preferential gossip prioritizes efficient paths
- **Better Bandwidth Usage**: Focused communication on high-degree neighbors

### Model Quality

- **Improved Accuracy**: Staleness-aware aggregation maintains model freshness
- **Stable Convergence**: Down-weighting stale updates prevents divergence
- **Resilience**: System adapts to network delays and node failures

### Network Resilience

- **Fault Tolerance**: Preferential attachment creates redundant paths
- **Self-Organization**: Network structure adapts to communication patterns
- **Scalability**: Efficient communication scales with network size

## Technical Details

### Version Tracking

Model versions are stored in the `additional_info` dictionary:
```python
model.additional_info = {
    "version": current_round,
    # ... other metadata
}
```

### Degree Tracking

Interaction degrees are maintained in the `Neighbors` class:
```python
nei_stats[neighbor_address] = {
    "degree": interaction_count
}
```

Each successful communication increments the degree counter.

### Staleness Decay Function

The staleness decay follows a simple inverse relationship:
```
decay = 1 / (1 + staleness)
```

This ensures:
- Fresh models (staleness=0): full weight (1.0)
- One round stale (staleness=1): 50% weight (0.5)
- Two rounds stale (staleness=2): 33% weight (0.33)
- And so on...

## Future Enhancements

The current implementation provides a foundation for future adaptive features:

1. **Similarity-Based Neighbor Selection**: Use cached models to select neighbors with similar model states
2. **Dynamic Staleness Thresholds**: Adapt decay rates based on network conditions
3. **Bandwidth-Aware Selection**: Consider connection quality in neighbor selection
4. **Adaptive Gossip Frequency**: Adjust gossip rate based on model convergence

## References

- **Base Framework**: [P2PFL - Federated Learning over P2P networks](https://github.com/p2pfl/p2pfl)
- **Communication Protocol**: Built on gRPC with Protocol Buffers
- **Gossip Protocol**: Implements epidemic-style model propagation

## License

This project extends P2PFL and is licensed under the [GNU General Public License, Version 3.0](LICENSE.md).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

For more information about the base P2PFL framework, visit the [official documentation](https://p2pfl.github.io/p2pfl/).

