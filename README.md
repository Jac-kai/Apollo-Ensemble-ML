# Apollo-Ensemble-ML

A modular Python-based ensemble machine learning system for classification, regression, training management, evaluation, visualization, and model persistence.

Apollo-Ensemble-ML is an interactive terminal-based ensemble learning project designed to support structured machine learning workflows with multiple ensemble strategies.  
It provides an end-to-end pipeline for loading datasets, selecting features and targets, training ensemble models, evaluating results, visualizing model behavior, and saving or reloading trained models.

This project was built not only to practice ensemble learning, but also to demonstrate modular system design, reusable architecture, and engineering-oriented machine learning workflow development.

---

## Project Overview

Apollo is a menu-driven ensemble learning framework that separates the machine learning workflow into organized modules.

The system integrates:

- dataset loading workflow
- feature and target selection
- classification ensemble training
- regression ensemble training
- model summary, saving, and loading
- prediction and probability preview
- evaluation result display
- feature importance and permutation importance
- tree visualization and confusion matrix plotting
- modular engine / mission / model / menu architecture

Apollo is designed to keep workflow state across operations so users can reuse loaded data, configured features, trained models, and evaluation results throughout the same session.

---

## Project Purpose

The purpose of Apollo is to provide a structured ensemble learning system that helps users:

- practice ensemble machine learning methods
- understand the workflow difference between classification and regression ensembles
- manage ensemble model training in a modular architecture
- inspect and visualize model results more conveniently
- build engineering-oriented machine learning projects with reusable components

This project is suitable for:

- ensemble learning practice
- machine learning portfolio projects
- structured ML workflow design
- model evaluation and visualization practice
- reusable Python project architecture development

---

## Supported Ensemble Models

Apollo currently supports both classification and regression workflows for the following ensemble families:

### Classification
- VotingClassifier
- BaggingClassifier
- AdaBoostClassifier
- GradientBoostingClassifier
- StackingClassifier

### Regression
- VotingRegressor
- BaggingRegressor
- AdaBoostRegressor
- GradientBoostingRegressor
- StackingRegressor

The internal model registry also records each model’s task type, family, and default scoring method so Apollo can dispatch the correct training and evaluation workflow automatically.

---

## Core Features

### 1. Dataset Loading
Apollo can search folders, list available files, and load a selected dataset into the workflow through its engine-based data-loading system.

### 2. Feature and Target Selection
Apollo uses a dedicated feature-selection workflow so users can choose one or multiple target columns and configure feature columns before model training.

### 3. Classification and Regression Training
Apollo separates classifier and regressor ensemble training menus.  
Users can choose a model family, configure common training settings, collect ensemble-specific parameters, and launch training from a guided menu system.

### 4. Configurable Training Workflow
Apollo supports training options such as:

- test size selection
- split random state
- model random state
- optional cross validation
- configurable CV folds
- optional outer preprocessing
- categorical encoder selection
- optional PCA integration
- default parameter-grid support for GridSearchCV

### 5. Preprocessing Support
Apollo supports tabular preprocessing with:

- missing-value imputation
- one-hot encoding
- ordinal encoding
- optional outer preprocessing pipeline
- feature-name extraction after transformation

It is designed to support both:
- outer unified preprocessing
- estimator-owned preprocessing pipelines

### 6. Evaluation and Visualization
Apollo provides multiple evaluation-related workflows, including:

- evaluation result display
- prediction preview
- predict-probability preview
- permutation importance
- feature importance
- tree plot visualization
- confusion matrix generation

### 7. Permutation Importance Workflow
Apollo includes a permutation importance engine that can:

- compute feature importance by shuffled-feature performance drop
- let the user configure permutation repeats
- let the user choose scoring metrics
- limit displayed features
- optionally save output plots

### 8. Model Management
Apollo supports:

- current model summary display
- trained model saving
- trained model loading
- prediction with the current active model

### 9. Modular Architecture
Apollo is structured around reusable components, including:

- engine layer
- backbone / base configuration layer
- mission layer
- model layer
- menu layer
- estimator and param-grid helpers

### 10. Cross-Validation Result Tracking
Apollo records cross-validation settings and best-search results during GridSearchCV workflows.  
It can store compact CV summaries, keep raw CV search results in memory, and export top-ranked CV results as CSV reports for later inspection.

This architecture makes the project easier to extend, maintain, and reuse across future ML systems.

---

## Project Architecture

A simplified conceptual structure is:

```text
Apollo/
│
├── Apollo_Main.py
├── Apollo_ML_Engine.py
├── Apollo_Load_Data_Menu.py
├── Apollo_Model_Menu.py
├── Apollo_Evaluation_Menu.py
├── Apollo_Model_Menu_Helper.py
├── Menu_Config.py
├── Menu_Helper_Decorator.py
│
├── Backbone/
│   ├── Ensemble_BaseConfig.py
│   └── FeatureCore.py
│
├── Ensemble_Missioner/
│   ├── Voting_Missioner.py
│   ├── Bagging_Missioner.py
│   ├── AdaBoosting_Missioner.py
│   ├── GradientBoosting_Missioner.py
│   └── Stacking_Missioner.py
│
├── Ensemble_Model/
│   ├── Voting_Model.py
│   ├── Bagging_Model.py
│   ├── AdaBoosting_Model.py
│   ├── GradientBoosting_Model.py
│   └── Stacking_Model.py
│
└── Estimators_ParamsGrid/
    ├── Estimators.py
    └── Params_Grid.py