"""
Module for training machine learning models for FPL player recommendations.
Includes functions for training, evaluating, and saving models.
"""

import logging
import os
import pickle
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_DIR = "models"
DEFAULT_RESULTS_DIR = "results"

def prepare_data(
    data: pd.DataFrame,
    target_col: str = 'total_points',
    id_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepare data for model training by splitting into train and test sets.
    
    Args:
        data: Input DataFrame containing features and target
        target_col: Name of the target column
        id_cols: List of identifier columns to exclude from features
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    if data.empty:
        logger.error("Empty dataframe provided for model training")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), []
    
    # Ensure target column exists
    if target_col not in data.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), []
    
    # Define columns to exclude from features
    if id_cols is None:
        id_cols = ['id', 'web_name', 'full_name', 'first_name', 'second_name']
    
    # Exclude non-numeric and identifier columns
    exclude_cols = id_cols + [target_col]
    feature_cols = [col for col in data.columns if col not in exclude_cols 
                  and pd.api.types.is_numeric_dtype(data[col])]
    
    if not feature_cols:
        logger.error("No valid numeric feature columns found in dataframe")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), []
    
    logger.info(f"Selected {len(feature_cols)} features for model training")
    
    # Split data
    X = data[feature_cols]
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test, feature_cols

def create_preprocessing_pipeline(categorical_features: Optional[List[str]] = None, 
                                 numeric_features: Optional[List[str]] = None) -> ColumnTransformer:
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        categorical_features: List of categorical feature column names
        numeric_features: List of numeric feature column names
        
    Returns:
        ColumnTransformer preprocessing pipeline
    """
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    return preprocessor

def get_model(model_type: str = 'random_forest', **kwargs) -> Optional[BaseEstimator]:
    """
    Get a model instance based on the specified type.
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional parameters to pass to the model constructor
        
    Returns:
        Model instance
    """
    models = {
        'random_forest': RandomForestRegressor(random_state=42, **kwargs),
        'gradient_boosting': GradientBoostingRegressor(random_state=42, **kwargs),
        'linear_regression': LinearRegression(**kwargs),
        'ridge': Ridge(random_state=42, **kwargs),
        'lasso': Lasso(random_state=42, **kwargs),
        'elastic_net': ElasticNet(random_state=42, **kwargs),
        'svr': SVR(**kwargs),
        'adaboost': AdaBoostRegressor(random_state=42, **kwargs)
    }
    
    if model_type not in models:
        logger.error(f"Unknown model type: {model_type}")
        logger.info(f"Available models: {list(models.keys())}")
        return None
    
    return models[model_type]

def get_hyperparameter_grid(model_type: str) -> Dict[str, List[Union[str, int, float, None]]]:
    """
    Get default hyperparameter grid for different model types.
    
    Args:
        model_type: Type of model for which to get hyperparameter grid
        
    Returns:
        Dictionary mapping parameter names to lists of values to try
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'elastic_net': {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        'svr': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['linear', 'rbf']
        },
        'adaboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    }
    
    if model_type not in param_grids:
        logger.warning(f"No default hyperparameter grid for model type: {model_type}")
        return cast(Dict[str, List[Union[str, int, float, None]]], {})
    
    return cast(Dict[str, List[Union[str, int, float, None]]], param_grids[model_type])

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
              model_type: str = 'random_forest',
              params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Train a machine learning model with the provided data.
    
    Args:
        X_train: Training features
        y_train: Training target values
        model_type: Type of model to train
        params: Parameters to use for the model
        
    Returns:
        Trained model
    """
    if X_train.empty or y_train.empty:
        logger.error("Empty training data provided")
        return None
    
    # Get model with specified parameters
    if params is None:
        params = {}
    
    model = get_model(model_type, **params)
    
    if model is None:
        return None
    
    # Train model
    logger.info(f"Training {model_type} model with {X_train.shape[0]} samples")
    model.fit(X_train, y_train)
    
    return model

def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                      model_type: str = 'random_forest',
                      param_grid: Optional[Dict[str, List[Union[str, int, float, None]]]] = None,
                      cv: int = 5,
                      n_jobs: int = -1,
                      method: str = 'grid',
                      n_iter: int = 20,
                      verbose: int = 1) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning for the specified model.
    
    Args:
        X_train: Training features
        y_train: Training target values
        model_type: Type of model to tune
        param_grid: Parameter grid to search (if None, uses default)
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 to use all cores)
        method: Search method - 'grid' or 'random'
        n_iter: Number of iterations for randomized search
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_estimator, best_params)
    """
    if X_train.empty or y_train.empty:
        logger.error("Empty training data provided")
        return None, {}
    
    # Get base model
    model = get_model(model_type)
    
    if model is None:
        return None, {}
    
    # Use default param grid if none provided
    if param_grid is None:
        param_grid = get_hyperparameter_grid(model_type)
        if not param_grid:
            logger.warning(f"No hyperparameter grid available for {model_type}, skipping tuning")
            return model.fit(X_train, y_train), {}
    
    # Choose search method
    if method == 'grid':
        logger.info(f"Performing GridSearchCV for {model_type} with {cv} folds")
        search = GridSearchCV(
            model, param_grid, cv=cv, n_jobs=n_jobs,
            scoring='neg_mean_squared_error', verbose=verbose
        )
    elif method == 'random':
        logger.info(f"Performing RandomizedSearchCV for {model_type} with {n_iter} iterations")
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
            scoring='neg_mean_squared_error', verbose=verbose
        )
    else:
        logger.error(f"Unknown search method: {method}")
        return None, {}
    
    # Perform search
    search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary of evaluation metrics
    """
    if model is None:
        logger.error("No model provided for evaluation")
        return {}
    
    if X_test.empty or y_test.empty:
        logger.error("Empty test data provided")
        return {}
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_variance
    }
    
    logger.info(f"Model evaluation metrics: {metrics}")
    return metrics

def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                       cv: int = 5, scoring: Union[str, List[str]] = 'neg_mean_squared_error') -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to cross-validate
        X: Features
        y: Target values
        cv: Number of cross-validation folds
        scoring: Scoring metric(s) to use
        
    Returns:
        Dictionary of cross-validation scores
    """
    if model is None:
        logger.error("No model provided for cross-validation")
        return {}
    
    if X.empty or y.empty:
        logger.error("Empty data provided for cross-validation")
        return {}
    
    logger.info(f"Performing {cv}-fold cross-validation")
    
    try:
        from sklearn.model_selection import cross_val_score
        
        if isinstance(scoring, list):
            # Handle multiple scoring metrics
            cv_results = {}
            for metric in scoring:
                logger.info(f"Calculating {metric} scores...")
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                cv_results[metric] = {
                    'scores': scores.tolist(),
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max()
                }
                logger.info(f"Cross-validation {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            return cv_results
        else:
            # Handle single scoring metric
            logger.info(f"Calculating {scoring} scores...")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            cv_results = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            logger.info(f"Cross-validation {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            return {scoring: cv_results}
    
    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        return {'error': str(e)}


class FPLPointsPredictor:
    """
    Class for loading a trained model and making points predictions for FPL players.
    """
    
    def __init__(self):
        """
        Initialize the FPL points predictor.
        """
        self.model = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)

    def load(self, model_file: Union[str, Path, PathLike]) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            model_file: Path to the saved model file (can be a string path or Path object)
            
        Returns:
            True if the model was successfully loaded, False otherwise
        """
        try:
            if not os.path.exists(model_file):
                self.logger.error(f"Model file not found: {model_file}")
                return False
            
            with open(str(model_file), 'rb') as f:
                model_data = pickle.load(f)
                
            # Extract model and feature names
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
            else:
                # Assume the file contains just the model
                self.model = model_data
                self.feature_names = None
                
            self.logger.info(f"Model loaded successfully from {model_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_file}: {e}")
            return False
    
    def predict(self, player_data: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[float]:
        """
        Predict the expected points for a player based on the loaded model.
        
        Args:
            player_data: Player features either as a DataFrame row or dictionary
            
        Returns:
            Predicted points or None if prediction failed
        """
        if self.model is None:
            self.logger.error("No model loaded. Call load() first.")
            return None
        
        try:
            # Convert dict to DataFrame if necessary
            if isinstance(player_data, dict):
                player_data = pd.DataFrame([player_data])
            
            # Filter features if we know what features the model needs
            if self.feature_names is not None:
                missing_features = [f for f in self.feature_names if f not in player_data.columns]
                if missing_features:
                    self.logger.error(f"Missing required features: {missing_features}")
                    return None
                
                player_data = player_data[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(player_data)
            
            # Return the first prediction (assuming single player input)
            return float(prediction[0]) if len(prediction) > 0 else None
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None

