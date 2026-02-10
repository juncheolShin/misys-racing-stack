# AI Coding Agent Instructions for ForzaETH Race Stack

## Project Overview

ForzaETH Race Stack is a **ROS 2-based autonomous racing system** combining perception, localization, planning, and control for F1TENTH vehicles. The focus is on **Model Predictive Path Integral (MPPI) control** running on **AGX Orin GPU** with aggressive latency optimization.

### Architecture
- **Modular ROS2 Nodes**: Each subsystem (perception, state_estimation, control, planner) is a separate ROS package
- **GPU-Accelerated Control**: MPPI solver uses JAX (BFloat16) for 10-15ms control loop on AGX Orin
- **High-Performance Patterns**: Emphasis on memory efficiency, non-blocking I/O, and asynchronous GPU execution

### Key Technology Stack
- **ROS 2 Humble**: Inter-process communication
- **JAX + BFloat16**: GPU acceleration (Tensor Core utilization on Ampere)
- **Python 3.10+**: All control/planning code
- **NumPy/Numba**: Perception preprocessing (CPU-bound)
- **CUDA**: GPU kernel execution via JAX

---

## Critical Workflow: Building & Running MPPI Controller

### Build from Source
```bash
# One-time setup
cd ~/ws/src/race_stack
colcon build --symlink-install --packages-select mppi

# Fast rebuild after code changes
colcon build --symlink-install --packages-select mppi --cmake-args "-DCMAKE_BUILD_TYPE=Release"
```

### Run on Hardware
```bash
# Terminal 1: Core system (perception + state estimation)
ros2 launch stack_master mppi_sim.launch.py

# Terminal 2: MPPI control loop (runs mppi_node)
# Automatic via launch file, but monitorable via:
ros2 node list   # See active nodes
ros2 topic echo /drive  # Monitor control outputs
```

### Critical Monitoring Commands
```bash
# GPU memory (watch for fragmentation causing >20ms spikes)
watch -n 0.1 "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,nounits,noheader"

# Real-time control loop latency (aim for 10-15ms, flag if >20ms)
ros2 topic echo /mppi_debug --max-count 100 | grep "TOTAL PLAN:"

# JAX compilation status (first run triggers JIT, expect 5-10s delay)
export JAX_TRACEBACK_FILTERING=off
```

---

## MPPI Solver Architecture (Core of Project)

### File Hierarchy
```
controller/mppi/mppi/
├── solver.py          # JAX JIT-compiled optimization core [CRITICAL]
├── cost.py            # Reward/cost computation with map integration
├── planner.py         # State machine + reference trajectory interpolation
├── mppi_node.py       # ROS2 node + parameter management
├── discretizers.py    # RK4 dynamics integration (JAX-compatible)
├── delay_compensator.py # Kinematic delay prediction
└── track_sdf.py       # Signed Distance Field for collision avoidance
```

### Data Flow
1. **`mppi_node.py:timer_cb()`** (50Hz)
   - Receives: odom state, waypoints, map
   - Calls: `planner.plan()` → returns `u_next` (2 floats)
   - Outputs: ROS `/drive` message

2. **`planner.py:plan()`** (low-level planning)
   - Interpolates reference trajectory from raw waypoints
   - Calls: `solver.solve()` 
   - Returns: next steering & velocity (DeviceArray)

3. **`solver.py:solve()`** (GPU optimization loop)
   - Runs N_ITERATIONS of `iteration_step()`
   - Iterates: sampling → rollout → weighting → update
   - Returns: `u_next` (only 2 floats, **NOT full trajectory**)

### Critical JAX Patterns in solver.py

#### Pattern 1: Static Arguments for JIT Stability
```python
@partial(jax.jit, static_argnums=(0, 10))
def _rollout(self, u_or_noise, x0, xref, p, Q, R, P, map_data, map_metadata, use_feedback_policy=False):
    # Index 0: self (always static)
    # Index 10: use_feedback_policy (Python control flow must be static)
```
**Why**: Traced values cannot control Python control flow. Make decision-making args static.

#### Pattern 2: BFloat16 for GPU Tensor Core Performance
```python
COMPUTE_DTYPE = jnp.bfloat16  # 2x speedup vs float32 on Ampere
# Convert ALL inputs at function entry:
state = jnp.asarray(state, dtype=COMPUTE_DTYPE)
```
**Why**: AGX Orin's Tensor Cores (Ampere) have 2x throughput for BF16. MPPI's probabilistic nature tolerates lower precision.

#### Pattern 3: Avoiding GPU-CPU Blocking (Asynchronous Execution)
```python
# GOOD: Return DeviceArray (stays on GPU)
u_next = a_opt[0]  # Still DeviceArray
return u_next  # mppi_node converts only when publishing

# BAD: Early float() conversion blocks GPU
u_next_cpu = float(u_next)  # ❌ GPU waits for completion (13ms blocking!)
```
**Why**: JAX is async. Only convert to NumPy right before ROS publish.

#### Pattern 4: JAX-Native Conditionals (Not Python If)
```python
# GOOD: jnp.where for JAX-traced conditions
u = jnp.where(use_feedback_policy, u_feedback, u_exploration)

# BAD: Python if statement in JIT function
if use_feedback_policy:  # ❌ TracerBoolConversionError
    u = readout_flow_control(...)
```
**Why**: Python `if` is evaluated at trace time, not runtime.

---

## Performance Tuning (AGX Orin Specific)

### Latency Budget (Target: 10-15ms)
| Component | Time | Notes |
|-----------|------|-------|
| MPPI compute (GPU) | 6-8ms | 4096 samples × 40 horizon × 5 iterations |
| GPU→CPU transfer | 13ms | Only 2 floats, but GPU sync blocks |
| Map interpolation | 2-3ms | Bilinear sampling × 81,920 queries |
| Reference trajectory | 0.5ms | Numba-JIT'd |
| ROS overhead | 0.5ms | Message serialization |

### Known Performance Killers
1. **Memory Fragmentation** (causes 25ms spikes)
   - 3GB GPU memory allocation per iteration
   - Solution: Pre-allocate noise_pool (500 × 4096 × 40 × 2)

2. **Discretizer Overhead** (original code)
   - `self.discretizer()` cannot be JIT'd (not JAX-friendly)
   - Solution: Inline RK4 directly in `_step()` using `self.model.f_jax()`

3. **Python-Level Conditionals in JIT** 
   - `if map_data is not None:` inside JIT breaks JAX
   - Solution: Make `map_data` required, not optional

4. **Cost Function Bilinear Interpolation** (50% of total)
   - 81,920 grid accesses per iteration
   - Solution: Vectorized interpolation + map cropping to ROI

---

## Configuration Management

### Config Files (Loaded in Specific Order)
1. **`config/mppi_params.yaml`** → `MPPIConfig` object
   - `n_samples`: 4096 (fixed, don't reduce below 1024)
   - `n_iterations`: 5 (tunable for speed/quality tradeoff)
   - `N`: 40 timesteps (1-second horizon at 40Hz)
   - `dt`: 0.025s
   - `guided_ratio`: 0.6 (60% guided + 40% exploration sampling)

2. **`config/models_param.yaml`** → `DynamicBicycleModel` params
   - Vehicle mass, wheelbase, tire coefficients
   - Steering/velocity rate limits

3. **Parameter Overrides** in `mppi_node.py`
   - Q, R costs can be tuned via ROS parameter service
   - Changes take effect immediately (no rebuild needed)

### Adding New Parameters
```python
# In mppi_node.py __init__:
self.declare_parameter('new_param', value=default, descriptor=ParameterDescriptor(...))

# In parameters_callback():
if param.name == 'new_param':
    self.mppi_config.new_param = param.value
    # Force re-JIT by clearing JAX cache (optional)
    jax.clear_backends()
```

---

## Common JAX/NumPy Pitfalls in This Codebase

### ❌ DON'T: Modify Arrays In-Place
```python
a_opt[0] = u_next  # JAX arrays are immutable!
```
→ Use `a_opt.at[0].set(u_next)` instead

### ❌ DON'T: Use Python Scalars in JIT
```python
radius = int(2 * sigma + 0.5)  # ❌ if sigma is traced
```
→ Make `sigma` static argument: `static_argnums=(2,)`

### ❌ DON'T: Print Inside JIT Functions
```python
@jax.jit
def loss(x):
    print(x)  # ❌ Won't print at runtime
    return x**2
```
→ Move print outside JIT or use `jax.debug.print()`

### ✅ DO: Use jnp.asarray for Type Coercion
```python
# Safe cross-dtype arithmetic:
state = jnp.asarray(state, dtype=COMPUTE_DTYPE)  # Converts CPU→GPU, float32→bfloat16
```

---

## Testing & Debugging

### Unit Tests
```bash
# Run MPPI solver tests
cd controller/mppi
python -m pytest test/test_solver.py -v

# Check JIT compilation
JAX_DISABLE_JIT=1 python test/test_solver.py  # Slow but easier debugging
```

### Debugging JAX Errors
```bash
# Enable full traceback (default hides internal JAX frames)
export JAX_TRACEBACK_FILTERING=off

# Profile JIT compilation time
import jax
jax.profiler.start_trace("trace.txt")
# ... run solver.solve() ...
jax.profiler.stop_trace()
```

### Profiling GPU Execution
```bash
# Monitor GPU during solve
watch -n 0.01 nvidia-smi --query-gpu=name,memory.used,utilization.gpu,utilization.memory

# Capture performance metrics
nvidia-smi dmon > gpu_metrics.log  # Real-time monitoring
```

---

## Integration Points & Dependencies

### ROS2 Message Flow
- **Input Topics**:
  - `/odom` (sensor_msgs/Odometry) → vehicle state
  - `/waypoints` (f110_msgs/WpntArray) → reference path
  - `/map` (nav_msgs/OccupancyGrid) → collision checking

- **Output Topics**:
  - `/drive` (ackermann_msgs/AckermannDriveStamped) → vehicle commands
  - `/mppi_debug` (custom) → latency metrics

### External Models
- **`config/model.py:DynamicBicycleModel`**: Must implement `.f_jax(x, u, p)` returning next state
- **`track_sdf.py`**: Computes signed distance field for safety constraints

---

## Editing Checklist for AI Agents

When modifying MPPI solver:

- [ ] **dtype consistency**: All NumPy→JAX conversions use `jnp.asarray(..., dtype=COMPUTE_DTYPE)`
- [ ] **no Python control flow in JIT**: Use `jnp.where()` instead of `if`
- [ ] **static vs traced args**: Check `static_argnums` matches decision-making parameters
- [ ] **no optional arguments in JIT**: Make map_data/metadata required
- [ ] **RK4 inlined**: Use `self.model.f_jax()` directly, not `self.discretizer()`
- [ ] **test performance**: Verify control loop stays under 15ms with `nvidia-smi watch`

---

## Quick Reference: File Purposes

| File | Purpose | Critical? |
|------|---------|-----------|
| `solver.py` | JAX JIT optimization loop | ⭐⭐⭐ |
| `cost.py` | Reward function + map lookup | ⭐⭐ |
| `mppi_node.py` | ROS2 node + state management | ⭐⭐ |
| `planner.py` | Reference trajectory generation | ⭐ |
| `discretizers.py` | RK4 dynamics integration | ⭐ |
| `delay_compensator.py` | Kinematic prediction | ⭐ |
| `track_sdf.py` | Collision constraint computation | ⭐ |

