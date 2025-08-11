# 🚀 Migration Guide: NERSC Perlmutter → PNNL Deception

## 📋 **Migration Checklist**

### **Phase 1: Code Repository & Documentation** ✅
- [x] Export current conda environment (`environment.yml`)
- [x] Export package list (`requirements_conda.txt`)
- [x] Sync code to GitHub repository
- [x] Create comprehensive README for new cluster setup (see `pytorch_emulator/README_DECEPTION.md`)
- [ ] Document cluster-specific configurations

### **Phase 2: Environment Setup**
- [ ] Create conda environment on Deception
- [ ] Install PyTorch with CUDA support
- [ ] Verify all dependencies are compatible
- [ ] Test data loading and preprocessing
- [ ] Validate model training pipeline

### **Phase 3: Data Migration**
- [ ] Transfer processed parquet files
- [ ] Update data paths in configurations
- [ ] Verify data accessibility and permissions
- [ ] Test data loading performance

### **Phase 4: Cluster-Specific Configuration**
- [ ] Update SLURM job scripts
- [ ] Configure module loading
- [ ] Set up output directories
- [ ] Test job submission and execution

### **Phase 5: Validation & Testing**
- [ ] Run small-scale tests
- [ ] Validate model outputs
- [ ] Performance benchmarking
- [ ] Full training pipeline test

---

## 🛠️ **Detailed Migration Steps**

### **1. Code Repository Setup**

#### **GitHub Repository Structure**
```
mlmicrophysics-pytorch/
├── README.md                    # Main documentation
├── environment.yml              # Conda environment
├── requirements_conda.txt       # Package list
├── setup.py                     # Package installation
├── configs/                     # Configuration files
├── models/                      # Model implementations
├── training/                    # Training infrastructure
├── scripts/                     # Execution scripts
├── data_loaders/                # Data loading modules
├── utils/                       # Utility functions
└── docs/                        # Documentation
```

#### **Files to Include in Repository**
- ✅ All Python source code
- ✅ Configuration files (YAML)
- ✅ SLURM job scripts
- ✅ Documentation (README, migration guide)
- ✅ Environment files
- ❌ Large data files (use .gitignore)
- ❌ Output directories (use .gitignore)
- ❌ Checkpoint files (use .gitignore)

### **2. Environment Setup on Deception**

#### **Step 1: Clone Repository**
```bash
git clone <your-github-repo>
cd mlmicrophysics-pytorch
```

#### **Step 2: Create Conda Environment**
```bash
# Load conda module (if needed)
module load conda

# Create environment from exported file
conda env create -f environment.yml

# Activate environment
conda activate mlmicrophysics-env
```

#### **Step 3: Verify Installation**
```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch.cuda; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import dask; print(f'Dask: {dask.__version__}')"
```

### **3. Data Migration Strategy**

#### **Option A: Direct Transfer (Recommended)**
```bash
# From Perlmutter to Deception
rsync -avz /pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/ \
    <deception-username>@deception.pnl.gov:/path/to/data/
```

#### **Option B: Archive and Transfer**
```bash
# Create compressed archive
tar -czf processed_data.tar.gz /pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/

# Transfer archive
scp processed_data.tar.gz <deception-username>@deception.pnl.gov:/path/to/data/

# Extract on Deception
tar -xzf processed_data.tar.gz
```

### **4. Configuration Updates**

#### **Update Data Paths**
```yaml
# configs/multi_gpu_production.yml
data:
  data_path: "/path/to/processed_data/"  # Update for Deception
  max_files: 100
  subsample: 1.0
```

#### **Update Output Paths**
```yaml
# configs/multi_gpu_production.yml
logging:
  output_dir: "/path/to/outputs/"  # Update for Deception
  checkpoint_dir: "/path/to/checkpoints/"
```

### **5. SLURM Script Updates**

#### **Deception-Specific SLURM Script**
```bash
#!/bin/bash
#SBATCH --job-name=pytorch_microphysics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=<your-account>
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load modules (adjust for Deception)
module purge
module load cuda/11.8
module load conda

# Activate environment
source activate mlmicrophysics-env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

# Run training
python scripts/train_streaming_parallel.py configs/multi_gpu_production.yml
```

### **6. Performance Optimization**

#### **Deception-Specific Optimizations**
- **Storage**: Use fast storage for data and checkpoints
- **Network**: Optimize for inter-node communication
- **Memory**: Adjust batch sizes based on available memory
- **GPU**: Configure for Deception's GPU architecture

---

## 🔧 **Troubleshooting Common Issues**

### **1. CUDA/GPU Issues**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### **2. Module Loading Issues**
```bash
# Check available modules
module avail

# Load required modules
module load cuda
module load conda
```

### **3. Data Path Issues**
```bash
# Verify data accessibility
ls -la /path/to/processed_data/
python -c "import pandas as pd; print(pd.read_parquet('/path/to/processed_data/file.parquet').shape)"
```

### **4. Memory Issues**
```bash
# Monitor memory usage
htop
nvidia-smi

# Adjust batch sizes in config
batch_size: 512  # Reduce if OOM
```

---

## 📊 **Validation Checklist**

### **Pre-Migration Validation**
- [ ] All tests pass on Perlmutter
- [ ] Environment file is complete
- [ ] Documentation is up-to-date
- [ ] Data paths are documented

### **Post-Migration Validation**
- [ ] Environment creation successful
- [ ] All imports work
- [ ] Data loading successful
- [ ] Small training run completes
- [ ] Performance is acceptable
- [ ] Full pipeline works end-to-end

---

## 🎯 **Success Metrics**

### **Performance Targets**
- **Data Loading**: < 30 seconds for first batch
- **Training Speed**: > 1000 samples/second
- **Memory Usage**: < 80% of available GPU memory
- **Checkpointing**: < 60 seconds per checkpoint

### **Functionality Targets**
- **Reproducibility**: Same results as Perlmutter
- **Scalability**: Multi-GPU training works
- **Reliability**: No crashes or hangs
- **Monitoring**: Proper logging and metrics

---

## 📞 **Support Resources**

### **Deception-Specific**
- **Documentation**: [Deception User Guide](https://deception.pnl.gov/docs/)
- **Support**: [Deception Support](mailto:support@deception.pnl.gov)
- **Slack**: #deception-users

### **General PyTorch**
- **PyTorch Docs**: [pytorch.org/docs](https://pytorch.org/docs/)
- **Dask Docs**: [dask.org](https://dask.org/)
- **SLURM Docs**: [slurm.schedmd.com](https://slurm.schedmd.com/)

---

## 🚨 **Critical Notes**

1. **Backup Everything**: Keep copies of working configurations
2. **Test Incrementally**: Start with small tests before full runs
3. **Monitor Resources**: Watch for memory/GPU issues
4. **Document Changes**: Keep track of cluster-specific modifications
5. **Version Control**: Commit all configuration changes

---

*Last Updated: $(date)*
*Migration Status: In Progress* 
