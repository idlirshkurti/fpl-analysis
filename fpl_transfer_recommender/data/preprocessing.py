"""
Module for preprocessing Fantasy Premier League data.
Provides functions for cleaning and preparing data for analysis.
"""

import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_player_data(players: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clean and transform raw player data into a structured DataFrame.
    
    Args:
        players (List[Dict[str, Any]]): Raw player data from the FPL API.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned player data.
    """
    if not players:
        logger.warning("No player data to clean")
        return pd.DataFrame()

    try:
        # Convert to DataFrame
        df = pd.DataFrame(players)
        
        # Select relevant columns
        relevant_columns = [
            'id', 'first_name', 'second_name', 'web_name', 'team', 'element_type',
            'selected_by_percent', 'now_cost', 'form', 'points_per_game', 'total_points',
            'goals_scored', 'assists', 'clean_sheets', 'saves', 'bonus',
            'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'minutes', 'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards', 'expected_goals',
            'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded'
        ]
        
        # Filter columns that exist in the data
        existing_columns = [col for col in relevant_columns if col in df.columns]
        df = df[existing_columns]
        
        # Convert cost from FPL format (multiplied by 10) to actual value in millions
        if 'now_cost' in df.columns:
            df['price'] = df['now_cost'] / 10.0
            
        # Convert percentage strings to float values
        if 'selected_by_percent' in df.columns:
            df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')
            
        # Create full_name column
        if 'first_name' in df.columns and 'second_name' in df.columns:
            df['full_name'] = df['first_name'] + ' ' + df['second_name']
            
        # Calculate points per million (value metric)
        if 'total_points' in df.columns and 'price' in df.columns:
            df['points_per_million'] = df['total_points'] / df['price']
            
        # Fill missing values
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning player data: {e}")
        return pd.DataFrame()


def clean_team_data(teams: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clean and transform raw team data into a structured DataFrame.
    
    Args:
        teams (List[Dict[str, Any]]): Raw team data from the FPL API.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned team data.
    """
    if not teams:
        logger.warning("No team data to clean")
        return pd.DataFrame()

    try:
        # Convert to DataFrame
        df = pd.DataFrame(teams)
        
        # Select relevant columns if they exist
        relevant_columns = [
            'id', 'name', 'short_name', 'strength', 'strength_overall_home', 
            'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
            'strength_defence_home', 'strength_defence_away'
        ]
        
        # Filter columns that exist in the data
        existing_columns = [col for col in relevant_columns if col in df.columns]
        df = df[existing_columns]
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning team data: {e}")
        return pd.DataFrame()


def clean_fixture_data(fixtures: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clean and transform raw fixture data into a structured DataFrame.
    
    Args:
        fixtures (List[Dict[str, Any]]): Raw fixture data from the FPL API.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned fixture data.
    """
    if not fixtures:
        logger.warning("No fixture data to clean")
        return pd.DataFrame()

    try:
        # Convert to DataFrame
        df = pd.DataFrame(fixtures)
        
        # Select relevant columns if they exist
        relevant_columns = [
            'id', 'event', 'finished', 'kickoff_time', 'team_h', 'team_a',
            'team_h_score', 'team_a_score', 'team_h_difficulty', 'team_a_difficulty'
        ]
        
        # Filter columns that exist in the data
        existing_columns = [col for col in relevant_columns if col in df.columns]
        df = df[existing_columns]
        
        # Convert date strings to datetime objects
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        
        # Fill missing scores for unfinished matches
        if 'finished' in df.columns and 'team_h_score' in df.columns and 'team_a_score' in df.columns:
            df.loc[~df['finished'], ['team_h_score', 'team_a_score']] = df.loc[~df['finished'], ['team_h_score', 'team_a_score']].fillna(0)
            
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning fixture data: {e}")
        return pd.DataFrame()


def generate_fixture_difficulty_matrix(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a fixture difficulty matrix for predicting player performance.
    
    Args:
        fixtures_df (pd.DataFrame): Cleaned fixture data.
        teams_df (pd.DataFrame): Cleaned team data.
        
    Returns:
        pd.DataFrame: Fixture difficulty matrix with teams as index and gameweeks as columns.
    """
    if fixtures_df.empty or teams_df.empty:
        logger.warning("Empty fixtures or teams data provided")
        return pd.DataFrame()
    
    try:
        # Create a matrix of team IDs and gameweeks
        difficulty_matrix = pd.DataFrame(index=teams_df['id'].unique())
        
        # Process only future fixtures (not finished)
        if 'finished' in fixtures_df.columns:
            future_fixtures = fixtures_df[~fixtures_df['finished']]
        else:
            future_fixtures = fixtures_df.copy()
        
        # For each gameweek, populate the difficulty rating
        for gameweek in future_fixtures['event'].unique():
            gw_fixtures = future_fixtures[future_fixtures['event'] == gameweek]
            
            # Initialize gameweek column with high difficulty (no match = 5)
            difficulty_matrix[f'GW{gameweek}'] = 5
            
            # Update difficulties based on home matches
            for _, fixture in gw_fixtures.iterrows():
                home_team = fixture['team_h']
                away_team = fixture['team_a']
                
                if 'team_h_difficulty' in fixture and 'team_a_difficulty' in fixture:
                    difficulty_matrix.loc[home_team, f'GW{gameweek}'] = fixture['team_h_difficulty']
                    difficulty_matrix.loc[away_team, f'GW{gameweek}'] = fixture['team_a_difficulty']
        
        return difficulty_matrix
        
    except Exception as e:
        logger.error(f"Error generating fixture difficulty matrix: {e}")
        return pd.DataFrame()


def calculate_player_form(player_df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Calculate a rolling form metric for players based on recent performances.
    
    Args:
        player_df (pd.DataFrame): Cleaned player data.
        window_size (int): Number of recent matches to consider for form calculation.
        
    Returns:
        pd.DataFrame: Player data with added form metrics.
    """
    if player_df.empty:
        logger.warning("Empty player data provided")
        return pd.DataFrame()
    
    try:
        # Make a copy to avoid modifying the original
        df = player_df.copy()
        
        # Check if necessary columns exist
        required_columns = ['minutes', 'total_points', 'id']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Required columns missing for form calculation")
            return player_df
        
        # Calculate minutes-weighted form (for players who played consistently)
        if 'form' not in df.columns:
            df['form'] = 0.0
        else:
            # Convert form to numeric if it's not already
            df['form'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
        
        # Calculate recent points per minute (if available in data)
        if 'event_points' in df.columns:
            df['points_per_minute'] = df.apply(
                lambda x: x['event_points'] / x['minutes'] if x['minutes'] > 0 else 0, 
                axis=1
            )
            
            # Weighted form based on minutes played
            df['weighted_form'] = df['form'] * np.sqrt(df['minutes'] / 90)
            
        return df
        
    except Exception as e:
        logger.error(f"Error calculating player form: {e}")
        return player_df


def merge_player_team_data(players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge player and team data to enrich player information with team context.
    
    Args:
        players_df (pd.DataFrame): Cleaned player data.
        teams_df (pd.DataFrame): Cleaned team data.
        
    Returns:
        pd.DataFrame: Enriched player data with team information.
    """
    if players_df.empty or teams_df.empty:
        logger.warning("Empty player or team data provided")
        return players_df
    
    try:
        # Make sure team column exists in player data
        if 'team' not in players_df.columns or 'id' not in teams_df.columns:
            logger.warning("Required columns for merging not found")
            return players_df
        
        # Rename team ID column in teams dataframe to avoid conflict
        teams_copy = teams_df.copy()
        teams_copy = teams_copy.rename(columns={'id': 'team_id'})
        
        # Merge player data with team data
        merged_df = pd.merge(
            players_df,
            teams_copy,
            left_on='team',
            right_on='team_id',
            how='left'
        )
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging player and team data: {e}")
        return players_df


def get_preprocessed_data(player_data: List[Dict], team_data: List[Dict], 
                        fixture_data: List[Dict]) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Preprocess all FPL data and return a dictionary of processed dataframes.
    
    Args:
        player_data (List[Dict]): Raw player data from the FPL API.
        team_data (List[Dict]): Raw team data from the FPL API.
        fixture_data (List[Dict]): Raw fixture data from the FPL API.
        
    Returns:
        Dict[str, Union[pd.DataFrame, Dict]]: Dictionary containing all processed data.
    """
    # Clean individual data components
    players_df = clean_player_data(player_data)
    teams_df = clean_team_data(team_data)
    fixtures_df = clean_fixture_data(fixture_data)
    
    # Generate derived data
    fixture_difficulty = generate_fixture_difficulty_matrix(fixtures_df, teams_df)
    
    # Calculate form for players
    players_with_form = calculate_player_form(players_df)
    
    # Merge player and team data
    enriched_players = merge_player_team_data(players_with_form, teams_df)
    
    # Return all processed data
    return {
        'players': enriched_players,
        'teams': teams_df,
        'fixtures': fixtures_df,
        'fixture_difficulty': fixture_difficulty
    }

