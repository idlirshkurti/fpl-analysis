"""
Module for feature engineering on Fantasy Premier League data.
Transforms preprocessed FPL data into features suitable for machine learning models.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def create_form_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on player form and recent performance.
    
    Args:
        player_df (pd.DataFrame): Preprocessed player data.
        
    Returns:
        pd.DataFrame: DataFrame with form-based features added.
    """
    df = player_df.copy()
    
    try:
        # Convert form to numeric if it's not already
        if 'form' in df.columns:
            df['form_numeric'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
            
        # Calculate form relative to price
        if 'form_numeric' in df.columns and 'now_cost' in df.columns:
            df['form_per_cost'] = df['form_numeric'] / (df['now_cost'] / 10)
            
        # Recent points trend (if available)
        if 'event_points' in df.columns:
            df['recent_points_trend'] = df['event_points'] - df['form_numeric']
            
        # Points consistency (if history data available)
        if 'history' in df.columns and isinstance(df['history'].iloc[0], list):
            df['points_std'] = df['history'].apply(
                lambda h: np.std([match.get('total_points', 0) for match in h[-5:]]) 
                if len(h) >= 5 else np.nan
            )
            
            # Points consistency ratio (higher is more consistent)
            df['consistency_ratio'] = df.apply(
                lambda row: row['form_numeric'] / (row['points_std'] + 0.1) 
                if pd.notnull(row.get('points_std', np.nan)) else 0,
                axis=1
            )
            
        return df
        
    except Exception as e:
        logger.error(f"Error creating form features: {e}")
        return player_df


def create_fixture_difficulty_features(player_df: pd.DataFrame, 
                                      fixture_df: pd.DataFrame,
                                      team_df: pd.DataFrame,
                                      num_future_gws: int = 5) -> pd.DataFrame:
    """
    Create features based on upcoming fixture difficulty.
    
    Args:
        player_df (pd.DataFrame): Preprocessed player data.
        fixture_df (pd.DataFrame): Fixture information.
        team_df (pd.DataFrame): Team information.
        num_future_gws (int): Number of future gameweeks to consider.
        
    Returns:
        pd.DataFrame: DataFrame with fixture difficulty features added.
    """
    df = player_df.copy()
    
    try:
        # Check if required columns are present
        if 'team' not in df.columns or fixture_df.empty or team_df.empty:
            logger.warning("Required data for fixture difficulty features is missing")
            return df
            
        # Create lookup dictionary for team difficulties
        team_id_to_strength = team_df.set_index('id')['strength'].to_dict() if 'strength' in team_df.columns else {}
        
        # Filter for future fixtures
        future_fixtures = fixture_df[fixture_df['finished'] == False].copy() if 'finished' in fixture_df.columns else pd.DataFrame()
        
        if not future_fixtures.empty and 'event' in future_fixtures.columns:
            # Sort by gameweek
            future_fixtures = future_fixtures.sort_values('event')
            
            # Group fixtures by team
            team_fixtures = {}
            for _, fixture in future_fixtures.iterrows():
                gw = fixture['event']
                
                # Skip if beyond the number of future gameweeks we want to consider
                if gw > num_future_gws:
                    continue
                    
                # Add home fixture
                team_h = fixture['team_h']
                diff_h = fixture.get('team_h_difficulty', 
                                     team_id_to_strength.get(fixture['team_a'], 3))
                                     
                if team_h not in team_fixtures:
                    team_fixtures[team_h] = []
                team_fixtures[team_h].append((gw, diff_h, 'H'))
                
                # Add away fixture
                team_a = fixture['team_a']
                diff_a = fixture.get('team_a_difficulty', 
                                     team_id_to_strength.get(fixture['team_h'], 3))
                                     
                if team_a not in team_fixtures:
                    team_fixtures[team_a] = []
                team_fixtures[team_a].append((gw, diff_a, 'A'))
            
            # Calculate next fixture difficulty
            df['next_fixture_difficulty'] = df['team'].map(
                lambda t: team_fixtures.get(t, [[0, 3, '']])[0][1] if t in team_fixtures and team_fixtures[t] else 3
            )
            
            # Calculate average difficulty over next N gameweeks
            df['avg_difficulty_next_n'] = df['team'].map(
                lambda t: np.mean([f[1] for f in team_fixtures.get(t, [])[:num_future_gws]]) 
                if t in team_fixtures and team_fixtures[t] else 3
            )
            
            # Calculate fixture ease score (inverse of difficulty)
            df['fixture_ease_score'] = 6 - df['avg_difficulty_next_n']
            
            # Count double gameweeks (if a team plays more than once in a GW)
            df['double_gameweeks'] = df['team'].map(
                lambda t: len(set([f[0] for f in team_fixtures.get(t, [])])) - len(team_fixtures.get(t, []))
                if t in team_fixtures else 0
            )
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating fixture difficulty features: {e}")
        return player_df


def create_team_strength_features(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on the strength of a player's team.
    
    Args:
        player_df (pd.DataFrame): Preprocessed player data.
        team_df (pd.DataFrame): Team information.
        
    Returns:
        pd.DataFrame: DataFrame with team strength features added.
    """
    df = player_df.copy()
    
    try:
        # Check if required columns are present
        if 'team' not in df.columns or team_df.empty:
            logger.warning("Required data for team strength features is missing")
            return df
        
        # Prepare team attributes dictionary
        team_attributes = {}
        
        # Overall strength
        if 'strength' in team_df.columns:
            team_attributes['team_strength'] = team_df.set_index('id')['strength'].to_dict()
            
        # Home and away strength
        for attr in ['strength_overall_home', 'strength_overall_away', 
                     'strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away']:
            if attr in team_df.columns:
                team_attributes[attr] = team_df.set_index('id')[attr].to_dict()
        
        # Add team attributes to player dataframe
        for attr_name, attr_dict in team_attributes.items():
            df[attr_name] = df['team'].map(lambda t: attr_dict.get(t, 0))
        
        # Create composite features
        if all(col in df.columns for col in ['strength_attack_home', 'strength_attack_away']):
            df['team_attack_strength'] = (df['strength_attack_home'] + df['strength_attack_away']) / 2
            
        if all(col in df.columns for col in ['strength_defence_home', 'strength_defence_away']):
            df['team_defence_strength'] = (df['strength_defence_home'] + df['strength_defence_away']) / 2
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating team strength features: {e}")
        return player_df


def create_positional_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specific to player positions.
    
    Args:
        player_df (pd.DataFrame): Preprocessed player data.
        
    Returns:
        pd.DataFrame: DataFrame with positional features added.
    """
    df = player_df.copy()
    
    try:
        # Check if required columns are present
        if 'element_type' not in df.columns:
            logger.warning("Element type column missing for positional features")
            return df
        
        # Map element_type to position name
        position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position'] = df['element_type'].map(position_map)
        
        # Create position dummy variables
        position_dummies = pd.get_dummies(df['element_type'], prefix='pos')
        df = pd.concat([df, position_dummies], axis=1)
        
        # Position-specific value calculations
        if 'total_points' in df.columns and 'now_cost' in df.columns:
            # Calculate position average points
            pos_avg_points = df.groupby('element_type')['total_points'].transform('mean')
            df['points_vs_position_avg'] = df['total_points'] - pos_avg_points
            
            # Calculate position average value (points per million)
            df['value'] = df['total_points'] / (df['now_cost'] / 10)
            pos_avg_value = df.groupby('element_type')['value'].transform('mean')
            df['value_vs_position_avg'] = df['value'] - pos_avg_value
        
        # Position-specific performance metrics
        if 'pos_1' in df.columns:  # Goalkeepers
            if 'saves' in df.columns:
                df['save_points_potential'] = df['saves'] * df['pos_1']
                
            if 'clean_sheets' in df.columns:
                df['gkp_cs_value'] = (df['clean_sheets'] * 4) * df['pos_1'] / (df['now_cost'] / 10)
        
        if 'pos_2' in df.columns:  # Defenders
            if 'clean_sheets' in df.columns:
                df['def_cs_value'] = (df['clean_sheets'] * 4) * df['pos_2'] / (df['now_cost'] / 10)
                
            if 'goals_scored' in df.columns:
                df['def_attacking_potential'] = df['goals_scored'] * 6 * df['pos_2']
        
        if 'pos_3' in df.columns:  # Midfielders
            if all(col in df.columns for col in ['goals_scored', 'assists']):
                df['mid_attacking_value'] = (df['goals_scored'] * 5 + df['assists'] * 3) * df['pos_3'] / (df['now_cost'] / 10)
        
        if 'pos_4' in df.columns:  # Forwards
            if all(col in df.columns for col in ['goals_scored', 'assists']):
                df['fwd_attacking_value'] = (df['goals_scored'] * 4 + df['assists'] * 3) * df['pos_4'] / (df['now_cost'] / 10)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating positional features: {e}")
        return player_df


def create_value_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to player value and return on investment.
    
    Args:
        player_df (pd.DataFrame): Preprocessed player data.
        
    Returns:
        pd.DataFrame: DataFrame with value features added.
    """
    df = player_df.copy()
    
    try:
        # Check if required columns are present
        if 'now_cost' not in df.columns:
            logger.warning("Cost column missing for value features")
            return df
        
        # Convert cost to actual value in millions
        df['price'] = df['now_cost'] / 10.0
        
        # Calculate points per million if total_points exists
        if 'total_points' in df.columns:
            df['points_per_million'] = df['total_points'] / df['price']
        
        # Calculate form per million if form exists
        if 'form_numeric' in df.columns:
            df['form_per_million'] = df['form_numeric'] / df['price']
        
        # Calculate expected points per million if xP exists
        if 'xP_next_n' in df.columns:
            df['xP_per_million'] = df['xP_next_n'] / df['price']
        
        # Popularity weighted value
        if 'selected_by_percent' in df.columns:
            df['selected_by_percent_numeric'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0)
            df['popularity_value_ratio'] = df['points_per_million'] / (df['selected_by_percent_numeric'] + 1)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating value features: {e}")
        return player_df


def calculate_expected_points(player_features: pd.DataFrame, 
                            fixture_df: pd.DataFrame,
                            team_df: pd.DataFrame,
                            num_future_gws: int = 5) -> pd.DataFrame:
    """
    Calculate expected points for players over the next N gameweeks.
    
    Args:
        player_features (pd.DataFrame): Player data with engineered features.
        fixture_df (pd.DataFrame): Fixture information.
        team_df (pd.DataFrame): Team information.
        num_future_gws (int): Number of future gameweeks to consider.
        
    Returns:
        pd.DataFrame: DataFrame with expected points predictions.
    """
    df = player_features.copy()
    
    try:
        # Check required columns
        required_columns = ['form_numeric', 'team', 'element_type']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Required columns missing for expected points calculation")
            return df
        
        # Filter future fixtures
        future_fixtures = fixture_df[fixture_df['finished'] == False].copy() if 'finished' in fixture_df.columns else pd.DataFrame()
        
        if future_fixtures.empty:
            logger.warning("No future fixtures available for expected points calculation")
            return df
                
        # Get the next N gameweeks
        gameweeks = sorted(future_fixtures['event'].unique())[:num_future_gws]
        
        if not gameweeks:
            logger.warning("No future gameweeks found")
            return df
                
        # Initialize expected points column
        df['xP_next_n'] = 0.0
        
        # Create fixture lookup for each team and gameweek
        fixture_lookup = {}
        for _, fixture in future_fixtures.iterrows():
            gw = fixture['event']
            if gw not in gameweeks:
                continue
                
            # Add home team fixture
            team_h = fixture['team_h']
            if team_h not in fixture_lookup:
                fixture_lookup[team_h] = {}
            if gw not in fixture_lookup[team_h]:
                fixture_lookup[team_h][gw] = []
            fixture_lookup[team_h][gw].append({
                'opponent': fixture['team_a'],
                'is_home': True,
                'difficulty': fixture.get('team_h_difficulty', 3)
            })
            
            # Add away team fixture
            team_a = fixture['team_a']
            if team_a not in fixture_lookup:
                fixture_lookup[team_a] = {}
            if gw not in fixture_lookup[team_a]:
                fixture_lookup[team_a][gw] = []
            fixture_lookup[team_a][gw].append({
                'opponent': fixture['team_h'],
                'is_home': False,
                'difficulty': fixture.get('team_a_difficulty', 3)
            })
        
        # Calculate expected points for each player
        for idx, player in df.iterrows():
            team_id = player['team']
            position = player['element_type']
            form = player['form_numeric']
            
            # Base expected points
            base_xp = form * 2 if form > 0 else 1.0
            
            # Position-based multiplier
            position_multiplier = {
                1: 1.0,  # GKP
                2: 1.1,  # DEF
                3: 1.2,  # MID
                4: 1.3   # FWD
            }.get(position, 1.0)
            
            # Calculate team fixtures expected points
            total_xp = 0.0
            fixture_count = 0
            
            if team_id in fixture_lookup:
                for gw in gameweeks:
                    if gw in fixture_lookup[team_id]:
                        for fixture in fixture_lookup[team_id][gw]:
                            # Difficulty-based adjustment (easier fixtures = more points)
                            difficulty = fixture['difficulty']
                            difficulty_factor = (6 - difficulty) / 2.5  # Scale from 0.4 to 2.0
                            
                            # Home advantage
                            home_factor = 1.1 if fixture['is_home'] else 0.9
                            
                            # Calculate expected points for this fixture
                            fixture_xp = base_xp * difficulty_factor * home_factor * position_multiplier
                            
                            # Add to total
                            total_xp += fixture_xp
                            fixture_count += 1
            
            # Calculate average expected points per gameweek, then multiply by number of gameweeks
            if fixture_count > 0:
                df.at[idx, 'xP_next_n'] = (total_xp / fixture_count) * min(fixture_count, num_future_gws)
            else:
                # If no fixtures found, use form as a fallback
                df.at[idx, 'xP_next_n'] = base_xp * num_future_gws * 0.5  # Penalize for lack of fixture data
        
        # Create per-gameweek expected points
        df['xP_per_gw'] = df['xP_next_n'] / num_future_gws
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating expected points: {e}")
        return player_features


def generate_full_feature_set(player_df: pd.DataFrame, fixture_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a complete set of features by applying all feature engineering steps.
    
    Args:
        player_df (pd.DataFrame): Preprocessed player data.
        fixture_df (pd.DataFrame): Fixture information.
        team_df (pd.DataFrame): Team information.
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    # Apply feature engineering steps in sequence
    df = create_form_features(player_df)
    df = create_fixture_difficulty_features(df, fixture_df, team_df)
    df = create_team_strength_features(df, team_df)
    df = create_positional_features(df)
    df = create_value_features(df)
    df = calculate_expected_points(df, fixture_df, team_df)
    
    return df
