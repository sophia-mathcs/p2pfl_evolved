# Fault Tolerance and Recovery Mechanisms in P2PFL

This document provides a comprehensive overview of the fault tolerance and recovery mechanisms implemented in P2PFL (Peer-to-Peer Federated Learning). These features ensure that federated learning experiments can continue even when nodes fail, and that failed nodes can rejoin the network seamlessly.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Checkpoint System](#checkpoint-system)
- [Node Failure Handling](#node-failure-handling)
- [Recovery Mechanisms](#recovery-mechanisms)
- [Aggregation with Failed Nodes](#aggregation-with-failed-nodes)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)

## Overview

P2PFL implements a robust fault tolerance system that allows federated learning experiments to:

- **Continue training** when nodes fail during execution
- **Detect node failures** automatically and adjust aggregation accordingly
- **Recover failed nodes** by restoring from checkpoints and rejoining the network
- **Maintain consistency** across the network even with dynamic node membership

The system is designed to handle both temporary and permanent node failures gracefully, ensuring that training progress is not lost and that the experiment can complete successfully.

## Key Features

### 1. **Automatic Checkpointing**
- Local checkpoints saved after each training round
- Remote checkpoints distributed to neighbor nodes periodically
- Multiple checkpoint types (local, aggregated, round_finished, remote)
- Delta compression for efficient remote checkpoint storage

### 2. **Failure Detection**
- Automatic detection of node failures during aggregation
- Train set synchronization to exclude failed nodes
- Timeout-based failure detection with configurable timeouts

### 3. **Graceful Degradation**
- Partial aggregation support when nodes fail
- Continue training with available nodes
- Automatic train set updates to reflect current network state

### 4. **Node Recovery**
- Automatic checkpoint restoration from local or remote sources
- Seamless reconnection to the network
- State synchronization with current training round
- Support for multiple recovery strategies

## Checkpoint System

### Checkpoint Types

P2PFL maintains several types of checkpoints:

1. **Local Checkpoints** (`local`)
   - Saved after each training round
   - Contains full model parameters and metadata
   - Stored on the node's local filesystem

2. **Round Finished Checkpoints** (`round_finished`)
   - Saved at the end of each round
   - Used as backup for recovery
   - Includes complete experiment state

3. **Remote Checkpoints** (`remote`)
   - Distributed to neighbor nodes periodically
   - Compressed using delta encoding for efficiency
   - Enables recovery even if local checkpoints are lost

4. **Aggregated Checkpoints** (`aggregated`)
   - Saved after model aggregation
   - Represents the aggregated model state

### Checkpoint Storage

Checkpoints are organized in a hierarchical directory structure:
```
checkpoints/
  └── {experiment_name}/
      └── {node_address}/
          ├── round_{N}_local.pkl
          ├── round_{N}_round_finished.pkl
          ├── round_{N}_aggregated.pkl
          └── round_{N}_remote.pkl
```

### Remote Checkpoint Distribution

- Remote checkpoints are saved every `REMOTE_CHECKPOINT_INTERVAL` rounds (configurable)
- Distributed to distant nodes in the network topology for redundancy
- Uses delta compression: stores only the difference from the last backup checkpoint
- Enables recovery even when local storage is lost

## Node Failure Handling

### Failure Detection

The system detects node failures through multiple mechanisms:

1. **Aggregation Timeout**
   - If a node doesn't send its model within `AGGREGATION_TIMEOUT`, it's considered failed
   - Configurable timeout (default: 60 seconds)

2. **Train Set Synchronization**
   - Active nodes maintain a `train_set` containing all participating nodes
   - When a node fails, it's removed from the train set
   - The aggregator automatically syncs with the updated train set

3. **Neighbor Detection**
   - Nodes monitor their neighbors' connectivity
   - Missing neighbors are detected and reported

### Handling Failed Nodes During Aggregation

When nodes fail during aggregation:

1. **Immediate Sync**: The aggregator checks if missing models are from failed nodes
   ```python
   # If all missing models are from failed nodes, sync immediately
   if failed_nodes == set(missing_models):
       self.__train_set = state.train_set.copy()
       # Trigger aggregation completion if all required models are present
   ```

2. **Timeout Handling**: If aggregation times out:
   - System checks if missing models are from failed nodes
   - Updates train set to exclude failed nodes
   - Proceeds with partial aggregation if supported

3. **Partial Aggregation**: Some aggregators (e.g., FedAvg) support partial aggregation:
   - Aggregate models from available nodes only
   - Continue training without waiting for failed nodes

### Train Set Management

The `train_set` is a critical component for fault tolerance:

- **Initialization**: Contains all nodes at the start of training
- **Updates**: Automatically updated when nodes fail or recover
- **Synchronization**: Propagated through gossip protocol
- **Aggregation**: Used to determine which models are required for aggregation

## Recovery Mechanisms

### Recovery Process

The recovery process follows a priority-based approach:

#### Priority 1: Local Checkpoint Recovery
1. Attempts to load local checkpoint from the last completed round
2. Tries different checkpoint types: `local`, `aggregated`, `round_finished`
3. Restores model parameters and experiment state

#### Priority 2: Remote Checkpoint Recovery
1. Queries neighbor nodes for remote checkpoints
2. Searches for checkpoints from recent rounds
3. Uses delta restoration if a last backup checkpoint exists
4. Falls back to full checkpoint restoration if no backup is available

#### Priority 3: Fresh Initialization
1. If no checkpoints are found, initializes a fresh model
2. Node must catch up by receiving models from neighbors

### Rejoin Function

The `rejoin()` function orchestrates the recovery process:

```python
def rejoin(
    state: NodeState,
    learner: Learner,
    communication_protocol,
    experiment_name: Optional[str] = None,
    round: Optional[int] = None,
) -> bool:
    """
    Rejoin the training process by restoring from checkpoint.
    
    Priority:
    1. Try local checkpoint recovery
    2. Try remote checkpoint recovery from neighbors
    3. Initialize fresh (worst case)
    """
```

### Recovery Steps

When a node recovers:

1. **Checkpoint Restoration**
   - Loads checkpoint from the last completed round before failure
   - Restores model parameters to the checkpoint state

2. **Network Reconnection**
   - Reconnects to active neighbor nodes
   - Re-establishes communication channels
   - Updates neighbor status information

3. **State Synchronization**
   - Sets current round to the recovery round
   - Updates `nei_status` to indicate the node is behind
   - Requests models from neighbors to catch up

4. **Training Resumption**
   - Starts learning workflow for the current round
   - Participates in aggregation and training

### Example: Node Recovery Flow

```python
# 1. Node fails after round 2
# 2. System detects failure, removes from train_set
# 3. Training continues with remaining nodes
# 4. At recovery round (e.g., round 4):
#    - Load checkpoint from round 2 (last completed before failure)
#    - Restore model parameters
#    - Reconnect to network
#    - Set round to 4
#    - Request models from neighbors
#    - Resume training
```

## Aggregation with Failed Nodes

### Train Set Synchronization

The aggregator automatically synchronizes with the node state's train set:

```python
def wait_and_get_aggregation(self, timeout, state=None):
    # Before waiting, check if state.train_set differs
    if set(state.train_set) != set(self.__train_set):
        missing_models = self.get_missing_models()
        failed_nodes = set(missing_models) - set(state.train_set)
        
        if failed_nodes == set(missing_models):
            # All missing models are from failed nodes
            self.__train_set = state.train_set.copy()
            # Proceed with aggregation
```

### Partial Aggregation Support

Aggregators can support partial aggregation:

- **FedAvg**: Supports partial aggregation by default
- **Scaffold**: May support partial aggregation depending on implementation
- **Custom Aggregators**: Can implement `SUPPORTS_PARTIAL_AGGREGATION = True`

When partial aggregation is enabled:
- Aggregation proceeds with available models only
- Failed nodes are excluded from the aggregation
- Training continues without waiting for failed nodes

### Timeout Configuration

Configurable timeouts for failure detection:

- `VOTE_TIMEOUT`: Time to wait for voting (default: 10 seconds)
- `AGGREGATION_TIMEOUT`: Time to wait for model aggregation (default: 60 seconds)

These can be adjusted based on network conditions and experiment requirements.

## Usage Examples

### Basic Failure Simulation

Run an experiment with node failures:

```bash
python -m p2pfl.examples.mnist.mnist_with_failure \
  --nodes 4 \
  --rounds 6 \
  --epochs 1 \
  --protocol memory \
  --failure_round 2 \
  --recovery_round 4 \
  --failure_nodes node-0
```

This will:
- Start 4 nodes
- Run 6 training rounds
- Fail `node-0` after round 2 completes
- Recover `node-0` before round 4 starts

### Programmatic Usage

```python
from p2pfl.node import Node
from p2pfl.checkpoints import rejoin

# Create and start nodes
nodes = [Node(model, data) for _ in range(4)]
for node in nodes:
    node.start()

# ... connect nodes and start training ...

# If a node fails, recover it:
failed_node = nodes[0]
if rejoin(
    state=failed_node.state,
    learner=failed_node.learner,
    communication_protocol=failed_node.communication_protocol,
    round=current_round
):
    print("Node recovered successfully")
else:
    print("Recovery failed, initializing fresh")
```

### Monitoring Node Failures

The system provides logging for failure detection and recovery:

```
⚠️ Node shortage detected at node-1: 2 neighbors, expected 3. Waiting 10 seconds...
💥💥💥 All nodes completed round 1. Failing nodes ['node-0'] before round 2 starts 💥💥💥
🔄🔄🔄 All nodes completed round 3. Recovering nodes ['node-0'] before round 4 starts 🔄🔄🔄
✅ Node node-0 restarted.
🚀 Started learning workflow for node-0 to participate in round 4 training.
```

## Configuration

### Settings for Fault Tolerance

Key settings in `p2pfl/settings.py`:

```python
class Settings:
    class general:
        REMOTE_CHECKPOINT_INTERVAL = 5  # Save remote checkpoint every N rounds
        
    class training:
        VOTE_TIMEOUT = 10  # Seconds to wait for voting
        AGGREGATION_TIMEOUT = 60  # Seconds to wait for aggregation
```

### Adjusting Timeouts

For faster failure detection in simulations:

```python
from p2pfl.settings import Settings

Settings.training.VOTE_TIMEOUT = 10
Settings.training.AGGREGATION_TIMEOUT = 60
```

For slower networks or large models, increase timeouts:

```python
Settings.training.AGGREGATION_TIMEOUT = 120  # 2 minutes
```

### Checkpoint Configuration

Control checkpoint frequency and storage:

```python
# Save remote checkpoints more frequently
Settings.general.REMOTE_CHECKPOINT_INTERVAL = 3

# Include evaluation metrics in checkpoints (slower but more informative)
save_checkpoint(
    state=state,
    learner=learner,
    round=round,
    include_evaluation=True
)
```

## Best Practices

1. **Checkpoint Frequency**: Balance between recovery granularity and storage overhead
   - More frequent checkpoints = better recovery but more storage
   - Remote checkpoints every 3-5 rounds is typically sufficient

2. **Timeout Configuration**: Adjust based on network conditions
   - Fast local networks: shorter timeouts (10-30 seconds)
   - Slow or unreliable networks: longer timeouts (60-120 seconds)

3. **Partial Aggregation**: Enable for aggregators that support it
   - Allows training to continue with fewer nodes
   - Reduces impact of node failures

4. **Monitoring**: Monitor logs for failure detection and recovery
   - Watch for timeout warnings
   - Verify successful recovery messages
   - Check train set updates

5. **Network Topology**: Consider topology impact on recovery
   - Dense topologies: faster recovery (more neighbors to fetch from)
   - Sparse topologies: may need longer timeouts

## Troubleshooting

### Common Issues

1. **Recovery Fails with "No checkpoint found"**
   - Ensure checkpoints are being saved (check logs)
   - Verify checkpoint directory permissions
   - Check that `REMOTE_CHECKPOINT_INTERVAL` is set appropriately

2. **Aggregation Times Out Frequently**
   - Increase `AGGREGATION_TIMEOUT`
   - Check network connectivity
   - Verify nodes are not overloaded

3. **Recovered Node Doesn't Catch Up**
   - Check that `nei_status` is updated correctly
   - Verify neighbors are sending models
   - Ensure round number is set correctly

4. **Train Set Not Syncing**
   - Verify gossip protocol is working
   - Check that failed nodes are removed from train_set
   - Monitor aggregator train_set synchronization

## Implementation Details

### Key Components

- **`p2pfl/checkpoints/checkpoint_loader.py`**: Checkpoint loading and recovery logic
- **`p2pfl/checkpoints/checkpoint_saver.py`**: Checkpoint saving functionality
- **`p2pfl/learning/aggregators/aggregator.py`**: Aggregation with failure handling
- **`p2pfl/examples/mnist/mnist_with_failure.py`**: Complete failure/recovery example

### Recovery Algorithm

The recovery algorithm follows these steps:

1. **Detection**: System detects node failure through timeouts or explicit signals
2. **Isolation**: Failed node is removed from train_set across all active nodes
3. **Continuation**: Training continues with remaining nodes
4. **Recovery Trigger**: At predetermined round, recovery is triggered
5. **Checkpoint Load**: System attempts to load checkpoint (local → remote → fresh)
6. **Reconnection**: Node reconnects to network and synchronizes state
7. **Resumption**: Node resumes training from recovery round

## Conclusion

P2PFL's fault tolerance and recovery mechanisms provide robust support for federated learning in dynamic, unreliable network environments. The system automatically handles node failures, continues training with available nodes, and enables seamless recovery of failed nodes through a comprehensive checkpoint system.

For more information, see:
- [Main README](README.md)
- [Documentation](https://p2pfl.github.io/p2pfl/)
- [Example: MNIST with Failure](p2pfl/examples/mnist/mnist_with_failure.py)

