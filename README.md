CS5491

# Step 0: Dependency
conda create -n CS5491 python=3.9
conda activate CS5491
pip install -r requirement.txt

# Step 1: Generate data
python data_extractor.py

# Step 2: Baselines
python baseline.py

# Step 3: CNN
python main.py --aug 0

# Step 4: CNN + Mixup
python main.py --aug 1

# Step 5: Confusion Matrix
python test.py
