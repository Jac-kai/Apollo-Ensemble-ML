# -------------------- Common training options --------------------
COMMON_PARAM_CONFIG = {
    # ---------- Test size ----------
    "test_size": {
        "label": "✒️ Test Size",
        "options": {
            1: 0.1,
            2: 0.2,
            3: 0.25,
            4: 0.3,
        },
        "default": 2,
    },
    # ---------- Random state ----------
    "split_random_state": {
        "label": "🎲 Split Random State",
        "options": {
            1: 42,
            2: 0,
            3: 7,
            4: 123,
        },
        "default": 1,
    },
    # ---------- Use CV ----------
    "use_cv": {
        "label": "🧪 Use Cross Validation",
        "options": {
            1: True,
            2: False,
        },
        "default": 1,
    },
    # ---------- CV folds ----------
    "cv_folds": {
        "label": "🎰 CV Folds",
        "options": {
            1: 3,
            2: 5,
            3: 10,
        },
        "default": 2,
    },
    # ---------- Encoding options ----------
    "cat_encoder": {
        "label": "🧩 Categorical Encoder",
        "options": {
            1: "ohe",
            2: "ordinal",
        },
        "default": 1,
    },
    # ---------- Model internal random state ----------
    "model_random_state": {
        "label": "🎯 Model Random State",
        "options": {
            1: 42,
            2: 0,
            3: 7,
            4: 123,
        },
        "default": 1,
    },
    # ---------- Using preprocess ----------
    "use_preprocess": {
        "label": "💼 Use Outer Preprocess",
        "options": {
            1: True,
            2: False,
        },
        "default": 1,
    },
}


# -------------------- Scoring options by task type --------------------
SCORING_CONFIG = {
    "classifier": {
        "label": "🎯 Scoring",
        "options": {
            1: "accuracy",
            2: "f1",
            3: "f1_weighted",
            4: "precision_weighted",
            5: "recall_weighted",
        },
        "default": 3,
    },
    "regressor": {
        "label": "🎯 Scoring",
        "options": {
            1: "r2",
            2: "neg_mean_squared_error",
            3: "neg_mean_absolute_error",
        },
        "default": 1,
    },
}

# -------------------- PCA parameter options --------------------
PCA_PARAM_CONFIG = {
    "use_pca": {
        "label": "🪄 Use PCA",
        "options": {
            1: True,
            2: False,
        },
        "default": 2,
    },
    "pca_n_components": {
        "label": "🍒 PCA Components",
        "options": {
            1: 2,
            2: 3,
            3: 5,
            4: 10,
            5: 20,
            6: None,
        },
        "default": 6,
    },
}

# -------------------- Permutation importance options --------------------
PERMUTATION_IMPORTANCE_CONFIG = {
    "n_repeats": {
        "label": "🔁 Permutation Repeats",
        "options": {
            1: 5,
            2: 10,
            3: 20,
            4: 30,
        },
        "default": 2,
    },
    "max_display": {
        "label": "📊 Max Display Features",
        "options": {
            1: 10,
            2: 20,
            3: 30,
            4: 50,
        },
        "default": 2,
    },
}

# -------------------- Ensemble parameter options --------------------
ENSEMBLE_PARAM_CONFIG = {
    # ---------- Voting classification parameter options ----------
    "VotingClassifier": [
        {
            "name": "voting",
            "label": "🗳️ Voting Type",
            "options": {
                1: "hard",
                2: "soft",
            },
            "default": 1,
        },
        {
            "name": "n_jobs",
            "label": "⚙️ N Jobs",
            "options": {
                1: None,
                2: -1,
            },
            "default": 2,
        },
        {
            "name": "flatten_transform",
            "label": "📦 Flatten Transform",
            "options": {
                1: True,
                2: False,
            },
            "default": 1,
        },
        {
            "name": "use_default_weights",
            "label": "⚖️ Use Default Equal Weights",
            "options": {
                1: True,
                2: False,
            },
            "default": 1,
        },
    ],
    # ---------- Voting regression parameter options ----------
    "VotingRegressor": [
        {
            "name": "n_jobs",
            "label": "⚙️ N Jobs",
            "options": {
                1: None,
                2: -1,
            },
            "default": 2,
        },
        {
            "name": "use_default_weights",
            "label": "⚖️ Use Default Equal Weights",
            "options": {
                1: True,
                2: False,
            },
            "default": 1,
        },
    ],
    # ---------- Bagging classification parameter options ----------
    "BaggingClassifier": [
        {
            "name": "n_estimators",
            "label": "🌲 N Estimators",
            "options": {
                1: 10,
                2: 30,
                3: 50,
                4: 100,
            },
            "default": 1,
        },
        {
            "name": "max_samples",
            "label": "🧪 Max Samples",
            "options": {
                1: 0.7,
                2: 1.0,
            },
            "default": 2,
        },
        {
            "name": "max_features",
            "label": "🧬 Max Features",
            "options": {
                1: 0.7,
                2: 1.0,
            },
            "default": 2,
        },
        {
            "name": "bootstrap",
            "label": "🎲 Bootstrap",
            "options": {
                1: True,
                2: False,
            },
            "default": 1,
        },
        {
            "name": "bootstrap_features",
            "label": "🧩 Bootstrap Features",
            "options": {
                1: False,
                2: True,
            },
            "default": 1,
        },
        {
            "name": "oob_score",
            "label": "🪟 OOB Score",
            "options": {
                1: False,
                2: True,
            },
            "default": 1,
        },
        {
            "name": "n_jobs",
            "label": "⚙️ N Jobs",
            "options": {
                1: None,
                2: -1,
            },
            "default": 2,
        },
    ],
    # ---------- Bagging regression parameter options ----------
    "BaggingRegressor": [
        {
            "name": "n_estimators",
            "label": "🌲 N Estimators",
            "options": {
                1: 10,
                2: 30,
                3: 50,
                4: 100,
            },
            "default": 1,
        },
        {
            "name": "max_samples",
            "label": "🧪 Max Samples",
            "options": {
                1: 0.7,
                2: 1.0,
            },
            "default": 2,
        },
        {
            "name": "max_features",
            "label": "🧬 Max Features",
            "options": {
                1: 0.7,
                2: 1.0,
            },
            "default": 2,
        },
        {
            "name": "bootstrap",
            "label": "🎲 Bootstrap",
            "options": {
                1: True,
                2: False,
            },
            "default": 1,
        },
        {
            "name": "bootstrap_features",
            "label": "🧩 Bootstrap Features",
            "options": {
                1: False,
                2: True,
            },
            "default": 1,
        },
        {
            "name": "oob_score",
            "label": "🪟 OOB Score",
            "options": {
                1: False,
                2: True,
            },
            "default": 1,
        },
        {
            "name": "n_jobs",
            "label": "⚙️ N Jobs",
            "options": {
                1: None,
                2: -1,
            },
            "default": 2,
        },
    ],
    # ---------- AdaBoost classification parameter options ----------
    "AdaBoostClassifier": [
        {
            "name": "n_estimators",
            "label": "🌲 N Estimators",
            "options": {
                1: 30,
                2: 50,
                3: 100,
            },
            "default": 2,
        },
        {
            "name": "learning_rate",
            "label": "📈 Learning Rate",
            "options": {
                1: 0.01,
                2: 0.1,
                3: 1.0,
            },
            "default": 3,
        },
    ],
    # ---------- AdaBoost regression parameter options ----------
    "AdaBoostRegressor": [
        {
            "name": "n_estimators",
            "label": "🌲 N Estimators",
            "options": {
                1: 30,
                2: 50,
                3: 100,
            },
            "default": 2,
        },
        {
            "name": "learning_rate",
            "label": "📈 Learning Rate",
            "options": {
                1: 0.01,
                2: 0.1,
                3: 1.0,
            },
            "default": 3,
        },
    ],
    # ---------- Gradient Boosting classification parameter options ----------
    "GradientBoostingClassifier": [
        {
            "name": "n_estimators",
            "label": "🌲 N Estimators",
            "options": {
                1: 50,
                2: 100,
            },
            "default": 2,
        },
        {
            "name": "learning_rate",
            "label": "📈 Learning Rate",
            "options": {
                1: 0.01,
                2: 0.1,
                3: 0.2,
            },
            "default": 2,
        },
        {
            "name": "subsample",
            "label": "🧪 Subsample",
            "options": {
                1: 0.8,
                2: 1.0,
            },
            "default": 2,
        },
        {
            "name": "criterion",
            "label": "✒️ Criterion",
            "options": {
                1: "friedman_mse",
                2: "squared_error",
            },
            "default": 1,
        },
        {
            "name": "min_samples_split",
            "label": "🪓 Min Samples Split",
            "options": {
                1: 2,
                2: 5,
            },
            "default": 1,
        },
        {
            "name": "min_samples_leaf",
            "label": "🍃 Min Samples Leaf",
            "options": {
                1: 1,
                2: 2,
            },
            "default": 1,
        },
        {
            "name": "max_depth",
            "label": "🪵 Max Depth",
            "options": {
                1: 2,
                2: 3,
                3: 5,
            },
            "default": 2,
        },
    ],
    # ---------- Gradient Boosting regression parameter options ----------
    "GradientBoostingRegressor": [
        {
            "name": "n_estimators",
            "label": "🌲 N Estimators",
            "options": {
                1: 50,
                2: 100,
            },
            "default": 2,
        },
        {
            "name": "learning_rate",
            "label": "📈 Learning Rate",
            "options": {
                1: 0.01,
                2: 0.1,
                3: 0.2,
            },
            "default": 2,
        },
        {
            "name": "subsample",
            "label": "🧪 Subsample",
            "options": {
                1: 0.8,
                2: 1.0,
            },
            "default": 2,
        },
        {
            "name": "criterion",
            "label": "✒️ Criterion",
            "options": {
                1: "squared_error",
                2: "friedman_mse",
                3: "absolute_error",
            },
            "default": 1,
        },
        {
            "name": "min_samples_split",
            "label": "🪓 Min Samples Split",
            "options": {
                1: 2,
                2: 5,
            },
            "default": 1,
        },
        {
            "name": "min_samples_leaf",
            "label": "🍃 Min Samples Leaf",
            "options": {
                1: 1,
                2: 2,
            },
            "default": 1,
        },
        {
            "name": "max_depth",
            "label": "🪵 Max Depth",
            "options": {
                1: 2,
                2: 3,
                3: 5,
            },
            "default": 2,
        },
    ],
    # ---------- Stacking classification parameter options ----------
    "StackingClassifier": [
        {
            "name": "stacking_cv",
            "label": "🎰 Stacking CV",
            "options": {
                1: 3,
                2: 5,
                3: 10,
            },
            "default": 2,
        },
        {
            "name": "n_jobs",
            "label": "⚙️ N Jobs",
            "options": {
                1: None,
                2: -1,
            },
            "default": 2,
        },
        {
            "name": "passthrough",
            "label": "🧷 Passthrough",
            "options": {
                1: False,
                2: True,
            },
            "default": 1,
        },
        {
            "name": "stack_method",
            "label": "🪜 Stack Method",
            "options": {
                1: "auto",
            },
            "default": 1,
        },
    ],
    # ---------- Stacking regression parameter options ----------
    "StackingRegressor": [
        {
            "name": "stacking_cv",
            "label": "🎰 Stacking CV",
            "options": {
                1: 3,
                2: 5,
                3: 10,
            },
            "default": 2,
        },
        {
            "name": "n_jobs",
            "label": "⚙️ N Jobs",
            "options": {
                1: None,
                2: -1,
            },
            "default": 2,
        },
        {
            "name": "passthrough",
            "label": "🧷 Passthrough",
            "options": {
                1: False,
                2: True,
            },
            "default": 1,
        },
    ],
}


# =================================================
