"""
config.py — Dataset configurations for csp-fairness-tool.

Each config entry defines paths, sensitive features, and target column
for a supported dataset. The tool is dataset-agnostic; all dataset-specific
information lives here.
"""

DATASET_CONFIGS = {
    "kdd": {
        "sensitive": ["sex", "race", "age"],  # sex is binary (0/1); race is 0-4
        "target": "income",
        "model_path": "DNN/model_processed_kdd_cleaned.h5",
        "data_path": "dataset/processed_kdd.csv",
    },
    "adult": {
        "sensitive": ["gender", "race", "age"],  # gender is binary (0/1); race is 0-4
        "target": "Class-label",
        "model_path": "DNN/model_processed_adult.h5",
        "data_path": "dataset/processed_adult.csv",
    },
    "compas": {
        "sensitive": ["Sex", "Age", "Race"],  # Sex is binary (0/1); Race is 0-5; note capitals
        "target": "Recidivism",
        "model_path": "DNN/model_processed_compas.h5",
        "data_path": "dataset/processed_compas.csv",
    },
    "german": {
        "sensitive": ["PersonStatusSex", "AgeInYears"],  # PersonStatusSex is 0-3 (multi-value)
        "target": "CREDITRATING",
        "model_path": "DNN/model_processed_greman_cleaned.h5",  # note: typo in source filename
        "data_path": "dataset/processed_german.csv",
    },
    "dutch": {
        "sensitive": ["sex", "age"],  # sex is binary (0/1)
        "target": "occupation",
        "model_path": "DNN/model_processed_dutch.h5",
        "data_path": "dataset/processed_dutch.csv",
    },
}

# Budget used by baseline — keep consistent across all comparisons
DEFAULT_BUDGET = 1000

# Fraction of budget spent on random seed sampling in Phase 1
DEFAULT_SEED_RATIO = 0.15

# Decision tree max depth for Phase 1
DT_MAX_DEPTH = 4

# Diversity constraint: new inputs must differ from known discriminatory
# inputs by at least this many features
DIVERSITY_K = 1
