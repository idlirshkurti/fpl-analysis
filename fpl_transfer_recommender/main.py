"""
Main application entry point for FPL Transfer Recommender.

This module handles the command-line interface and orchestrates the workflow
from data fetching to generating recommendations.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Import the modules from our package
from fpl_transfer_recommender.data.api import (
    get_fixture_data,
    get_player_data,
    get_team_data,
    get_user_team_data,
)
from fpl_transfer_recommender.data.preprocessing import get_preprocessed_data
from fpl_transfer_recommender.models.feature_engineering import generate_full_feature_set
from fpl_transfer_recommender.models.recommendation import TransferRecommender
from fpl_transfer_recommender.models.training import FPLPointsPredictor
from fpl_transfer_recommender.utils.config import (
    DEFAULT_EXTRA_BUDGET,
    DEFAULT_MODEL_FILE,
    DEFAULT_NUM_TRANSFERS,
)
from fpl_transfer_recommender.utils.helpers import setup_logging

# Set up logger
logger = setup_logging("fpl_transfer_recommender")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Fantasy Premier League (FPL) Transfer Recommendation System"
    )
    
    # Required arguments
    parser.add_argument("--user_id", type=int, required=True, help="Your FPL user ID")
    
    # Optional arguments
    parser.add_argument(
        "--transfers", type=int, default=DEFAULT_NUM_TRANSFERS,
        help=f"Number of transfers to recommend (default: {DEFAULT_NUM_TRANSFERS})"
    )
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_EXTRA_BUDGET,
        help=f"Extra budget in millions (default: {DEFAULT_EXTRA_BUDGET})"
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_FILE),
        help=f"Path to pre-trained model file (default: {DEFAULT_MODEL_FILE})"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain the model even if a model file exists"
    )
    parser.add_argument(
        "--gameweek", type=int, help="Specific gameweek to analyze (default: current gameweek)"
    )
    parser.add_argument(
        "--output", type=str, choices=["console", "json", "csv"], default="console",
        help="Output format for recommendations (default: console)"
    )
    parser.add_argument(
        "--chip", type=str, choices=["none", "wildcard", "freehit", "triple_captain", "bench_boost"],
        default="none", help="Analyze chip strategy (default: none)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable data caching (always fetch fresh data)"
    )
    
    return parser.parse_args()

async def fetch_data(use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch all required data from the FPL API.
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrame objects with all required data
    """
    logger.info("Fetching data from FPL API...")
    
    try:
        # Fetch player, team, and fixture data (functions handle async internally)
        player_data = get_player_data()
        team_data = get_team_data()
        fixture_data = get_fixture_data()
        
        # Check if we got data
        if not player_data or not team_data or not fixture_data:
            logger.error("Failed to fetch required data from FPL API")
            return {}
            
        logger.info(f"Fetched data for {len(player_data)} players, {len(team_data)} teams, {len(fixture_data)} fixtures")
        
        # Preprocess the data
        processed_data = get_preprocessed_data(player_data, team_data, fixture_data)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {}

async def fetch_user_team(user_id: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch the user's current team from the FPL API.
    
    Args:
        user_id (int): The user's FPL ID
        use_cache: Whether to use cached data if available
        
    Returns:
        pd.DataFrame: DataFrame containing the user's current team
    """
    logger.info(f"Fetching team data for user ID {user_id}...")
    
    try:
        user_team_data = get_user_team_data(user_id)
        
        # Check if the dictionary is empty or doesn't contain team data
        if not user_team_data or 'team' not in user_team_data or not user_team_data['team']:
            logger.error(f"Failed to fetch team data for user ID {user_id}")
            return pd.DataFrame()
        
        # Convert the team list to a DataFrame
        user_team_df = pd.DataFrame(user_team_data['team'])
        
        logger.info(f"Successfully fetched team with {len(user_team_df)} players")
        return user_team_df
        
    except Exception as e:
        logger.error(f"Error fetching user team: {e}")
        return pd.DataFrame()

def generate_features(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate features for player prediction.
    
    Args:
        raw_data (Dict[str, pd.DataFrame]): Raw data from the FPL API
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    logger.info("Generating features for prediction...")
    
    try:
        if not raw_data or 'players' not in raw_data or 'teams' not in raw_data or 'fixtures' not in raw_data:
            logger.error("Missing required data for feature generation")
            return pd.DataFrame()
            
        # Generate full feature set
        players_with_features = generate_full_feature_set(
            raw_data['players'], raw_data['fixtures'], raw_data['teams']
        )
        
        logger.info(f"Generated features for {len(players_with_features)} players")
        return players_with_features
        
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        return pd.DataFrame()

def predict_points(players_with_features: pd.DataFrame, model_path: str, retrain: bool = False) -> pd.DataFrame:
    """
    Predict player points using a trained model or expected points calculation.
    
    Args:
        players_with_features (pd.DataFrame): Players with engineered features
        model_path (str): Path to pre-trained model
        retrain (bool): Whether to retrain the model even if it exists
        
    Returns:
        pd.DataFrame: DataFrame with predicted points
    """
    logger.info("Predicting player points...")
    
    try:
        # Check if we already have expected points calculated
        if 'xP_next_n' in players_with_features.columns:
            logger.info("Using calculated expected points (xP_next_n)")
            # Rename for consistency
            players_with_predictions = players_with_features.copy()
            players_with_predictions['predicted_points'] = players_with_features['xP_next_n']
            return players_with_predictions
        
        # Otherwise, try to use a trained model
        model_file = Path(model_path)
        predictor = FPLPointsPredictor()
        
        # Check if model exists and should be loaded
        if model_file.exists() and not retrain:
            logger.info(f"Loading model from {model_file}")
            if predictor.load(str(model_file)):
                # Predict points
                players_with_predictions = players_with_features.copy()
                players_with_predictions['predicted_points'] = predictor.predict(players_with_features)
                return players_with_predictions
            else:
                logger.warning(f"Failed to load model from {model_file}")
        
        # If no model or retrain requested, use expected points as prediction
        logger.info("No model available or retrain requested, using expected points")
        players_with_predictions = players_with_features.copy()
        if 'xP_next_n' in players_with_features.columns:
            players_with_predictions['predicted_points'] = players_with_features['xP_next_n']
        else:
            # Use total points as fallback
            logger.warning("No expected points available, using total_points as fallback")
            if 'total_points' in players_with_features.columns:
                players_with_predictions['predicted_points'] = players_with_features['total_points']
            else:
                logger.error("No points data available for prediction")
                players_with_predictions['predicted_points'] = 0.0
        
        return players_with_predictions
        
    except Exception as e:
        logger.error(f"Error predicting points: {e}")
        # Return original data with zero predicted points
        players_with_predictions = players_with_features.copy()
        players_with_predictions['predicted_points'] = 0.0
        return players_with_predictions

def make_transfer_recommendations(
    user_team: pd.DataFrame,
    all_players: pd.DataFrame,
    budget: float,
    num_transfers: int,
    chip: str = "none"
) -> Dict[str, Any]:
    """
    Generate transfer recommendations based on the user's team and predicted player points.
    
    Args:
        user_team (pd.DataFrame): The user's current team
        all_players (pd.DataFrame): All players with predicted points
        budget (float): Available extra budget in millions
        num_transfers (int): Number of transfers to recommend
        chip (str): Chip to analyze ("none", "wildcard", "freehit", etc.)
        
    Returns:
        Dict[str, Any]: Dictionary containing transfer recommendations
    """
    logger.info(f"Generating transfer recommendations for {num_transfers} transfers with budget {budget}m...")
    
    try:
        # Create recommender
        recommender = TransferRecommender()
        
        # If analyzing chip strategy
        if chip != "none":
            logger.info(f"Analyzing {chip} chip strategy")
            chip_options = [chip]
            strategy = recommender.recommend_team_strategy(
                user_team, all_players, budget, num_transfers, chip_options
            )
            return strategy
        
        # Generate standard recommendations
        recommendations = recommender.optimize_transfers(
            user_team, all_players, budget, num_transfers
        )
        
        logger.info(f"Generated {len(recommendations['transfers'])} transfer recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {"transfers": [], "total_points_gain": 0.0, "remaining_budget": budget}

def display_recommendations(recommendations: Dict[str, Any], output_format: str = "console") -> None:
    """
    Display transfer recommendations to the user.
    
    Args:
        recommendations (Dict[str, Any]): Recommendations from the recommender
        output_format (str): Format for displaying recommendations (console, json, csv)
    """
    if not recommendations or 'transfers' not in recommendations or not recommendations['transfers']:
        logger.warning("No recommendations available to display")
        print("\nNo transfer recommendations found. Try with a different budget or number of transfers.")
        return
    
    if output_format == "json":
        import json
        print(json.dumps(recommendations, indent=2, default=str))
        return
        
    if output_format == "csv":
        csv_rows = []
        headers = ["Out", "Out Team", "Out Cost", "In", "In Team", "In Cost", "Cost Diff", "Points Gain"]
        
        for transfer in recommendations['transfers']:
            csv_rows.append([
                transfer['out']['name'],
                transfer['out']['team'],
                f"£{transfer['out']['cost']}m",
                transfer['in']['name'],
                transfer['in']['team'],
                f"£{transfer['in']['cost']}m",
                f"£{transfer['cost_change']:.1f}m",
                f"+{transfer['points_gain']:.2f} pts"
            ])
        
        import csv
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        writer.writerows(csv_rows)
        return
    
    # Default console output
    print("\n===== FPL Transfer Recommendations =====\n")
    
    for i, transfer in enumerate(recommendations['transfers'], 1):
        out_player = transfer['out']
        in_player = transfer['in']
        
        print(f"Transfer {i}:")
        print(f"OUT: {out_player['name']} (£{out_player['cost']}m, {out_player['predicted_points']:.2f} pts)")
        print(f"IN:  {in_player['name']} (£{in_player['cost']}m, {in_player['predicted_points']:.2f} pts)")
        print(f"Cost: {'+ ' if transfer['cost_change'] > 0 else ''}£{transfer['cost_change']:.1f}m")
        print(f"Expected points gain: +{transfer['points_gain']:.2f}")
        print()
    
    print(f"Total expected points gain: +{recommendations['total_points_gain']:.2f}")
    print(f"Remaining budget: £{recommendations['remaining_budget']:.1f}m")
    print("\n=========================================\n")

async def main():
    """Main function that orchestrates the application workflow."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Fetch data
        raw_data = await fetch_data(not args.no_cache)
        if not raw_data:
            logger.error("Failed to fetch and process required data")
            print("Error: Could not fetch required data from the FPL API.")
            return 1
        
        # Fetch user team
        user_team = await fetch_user_team(args.user_id, not args.no_cache)
        if user_team.empty:
            logger.error(f"Failed to fetch team for user ID {args.user_id}")
            print(f"Error: Could not fetch team data for user ID {args.user_id}.")
            return 1
        
        # Generate features
        players_with_features = generate_features(raw_data)
        if players_with_features.empty:
            logger.error("Failed to generate features for players")
            print("Error: Could not generate features for player prediction.")
            return 1
        
        # Predict points
        players_with_predictions = predict_points(
            players_with_features, args.model, args.retrain
        )
        
        # Generate transfer recommendations
        recommendations = make_transfer_recommendations(
            user_team, players_with_predictions, args.budget, args.transfers, args.chip
        )
        
        # Display recommendations
        display_recommendations(recommendations, args.output)
        
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        print(f"An unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

