# Data Loading Approaches for PyTorch Microphysics Emulator

This document describes the three different data loading approaches available in the PyTorch microphysics emulator and provides guidance on when to use each one.

## 🎯 **Quick Start: Which Data Loader Should I Use?**

| Use Case | Recommended Loader | Config Setting |
|----------|-------------------|----------------|
| **Small-scale testing** (< 10 files) | Optimized Streaming | `loader_type: "optimized"` |
| **Production training** (large datasets) | Dask-based | `loader_type: "dask"` |
| **Debugging/compatibility** | Current Streaming | `loader_type: "streaming"` |

---

## 🏗️ **Available Data Loaders**

### 1. **Current Streaming Data Loader** 
**File:** `models/streaming_data_loader.py`
**Status:** ⚠️ **Extremely slow for large datasets**

#### Features:
- Sample-by-sample iteration
- Memory-efficient for very small datasets
- Compatible with existing configs

#### Performance:
- ❌ **Very slow**:
- ❌ **Poor scaling**: Gets slower with larger datasets
- ✅ **Low memory**: Minimal memory usage

#### When to Use:
- 🔧 **For Debugging**
- 🧪 **Small compatibility tests**
- ⚠️ **Not recommended for production**

#### Configuration:
```yaml
data:
  loader_type: "streaming"  # or "current"
  # ... other data config
```

---

### 2. **Optimized Streaming Data Loader** 
**File:** `models/streaming_data_loader_v2.py`
**Status:** ✅ **Recommended for most use cases**

#### Features:
- Yields **pre-batched data** instead of individual samples
- Vectorized preprocessing of entire chunks
- No sample-by-sample iteration bottleneck
- Still memory-efficient

#### When to Use:
- 🚀 **Default choice for most training**
- ✅ **Small to medium datasets** (< 50 files)
- ✅ **Quick prototyping and testing**

#### Configuration:
```yaml
data:
  loader_type: "optimized"  # or "optimized_streaming"
  chunk_size: 50000  # Samples per file chunk
  # ... other data config
```

---

### 3. **Dask-Based Data Loader** 
**File:** `models/dask_data_loader.py`
**Status:** ✅ **Best for large-scale production**

#### Features:
- **Distributed processing** using Dask
- **Parallel preprocessing** across multiple workers
- Loads entire dataset into memory after preprocessing
- Most similar to original Keras approach


#### When to Use:
- 🏭 **Large-scale production training** (50+ files)
- 🚀 **Maximum performance needed**
- 💾 **When you have sufficient RAM**

#### Configuration:
```yaml
data:
  loader_type: "dask"
  chunk_size: 100000  # Larger chunks for efficiency
  n_dask_workers: 4   # Number of Dask workers
  # ... other data config
```

---

## 🧪 **Performance Testing**

### Running Performance Tests

Test a specific loader:
```bash
python scripts/test_data_loaders.py --config configs/dask_test.yml --loader dask
```

Compare all loaders:
```bash
python scripts/test_data_loaders.py --config configs/dask_test.yml --loader all
```

Submit SLURM test job:
```bash
sbatch test_data_loaders.sh
```



## ⚙️ **Configuration Guide**

### Switching Data Loaders

Simply change the `loader_type` in your config file:

```yaml
data:
  loader_type: "optimized"  # Change this line
  data_path: "/path/to/parquet/files/"
  batch_size: 1024
  # ... rest of config unchanged
```

### Loader-Specific Parameters

#### Optimized Streaming:
```yaml
data:
  loader_type: "optimized"
  chunk_size: 50000        # Samples per file chunk
  batch_size: 1024         # Final batch size
```

#### Dask-based:
```yaml
data:
  loader_type: "dask"
  chunk_size: 100000       # Larger chunks for Dask
  n_dask_workers: 4        # Number of parallel workers
  batch_size: 1024         # Final batch size
```

#### Current Streaming:
```yaml
data:
  loader_type: "streaming"
  chunk_size: 50000        # File reading chunk size
  shuffle_buffer_size: 100000  # Shuffling buffer
  batch_size: 1024         # DataLoader batching
```
