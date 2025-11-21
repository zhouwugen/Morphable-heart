# Crafting A Morphable Heart Model for End-to-end Regression

# Install Dependencies
```bash
conda create -n morphable-heart python==3.8
conda activate morphable-heart
pip install -r requirements.txt
```

# Cardiac mesh fitting
```bash
cd Mesh fitting
python FourChambers.py
```

# Run PCA
```bash
cd PCA
python run_pca_model.py
```

# Pre-training on synthsis data
```bash
python train.py
```

# Finetune on real CTA data
```bash
./finetune.sh
```

# Inference to predict cardiac mesh
```bash
python inference.py
```


