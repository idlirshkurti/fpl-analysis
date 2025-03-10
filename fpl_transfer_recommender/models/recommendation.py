"""
Module for providing FPL transfer recommendations based on expected player points,
budget constraints, and optimal team structure.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np
import pandas as pd

T = TypeVar('T')

logger = logging.getLogger(__name__)
class TransferRecommender:
    """
    Class for generating FPL transfer recommendations based on model predictions
    and team constraints.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the transfer recommender with an optional prediction model.
        
        Args:
            model_path (Optional[str]): Path to a saved prediction model.
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a saved prediction model.
        
        Args:
            model_path (str): Path to the saved model file.
            
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('feature_columns')
            
            logger.info(f"Loaded prediction model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_points(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected points for players using the loaded model.
        
        Args:
            player_data (pd.DataFrame): DataFrame containing player features.
            
        Returns:
            pd.DataFrame: DataFrame with added expected points predictions.
        """
        if self.model is None:
            logger.error("No prediction model loaded")
            return player_data
        
        try:
            # Make a copy to avoid modifying the original
            df = player_data.copy()
            
            # Check if all required feature columns are present
            if self.feature_columns:
                missing_cols = [col for col in self.feature_columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing feature columns: {missing_cols}")
                    # Add missing columns with default values
                    for col in missing_cols:
                        df[col] = 0
            
                # Extract features for prediction
                X = df[self.feature_columns].values
                
                # Apply scaling if available
                if self.scaler:
                    X = self.scaler.transform(X)
                
                # Make predictions
                df['predicted_points'] = self.model.predict(X)
                
            else:
                logger.warning("No feature columns specified, using all numeric columns")
                # Use all numeric columns as features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                X = df[numeric_cols].values
                df['predicted_points'] = self.model.predict(X)
            
            return df
            
        except Exception as e:
            logger.error(f"Error predicting points: {e}")
            return player_data
    
    def find_transfer_targets(self, current_team: pd.DataFrame, 
                           all_players: pd.DataFrame,
                           budget: float,
                           num_transfers: int = 1,
                           exclude_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Find optimal transfer targets based on predicted points and constraints.
        
        Args:
            current_team (pd.DataFrame): DataFrame containing current team players.
            all_players (pd.DataFrame): DataFrame with all available players and predictions.
            budget (float): Available transfer budget in millions.
            num_transfers (int): Number of transfers to recommend.
            exclude_ids (List[int]): List of player IDs to exclude from recommendations.
            
        Returns:
            pd.DataFrame: DataFrame with recommended transfers.
        """
        try:
            # Ensure we have predicted points
            if 'predicted_points' not in all_players.columns and 'xP_next_n' in all_players.columns:
                # Use xP_next_n if it exists and predicted_points doesn't
                all_players = all_players.rename(columns={'xP_next_n': 'predicted_points'})
            elif 'predicted_points' not in all_players.columns and self.model is not None:
                all_players = self.predict_points(all_players)
            elif 'predicted_points' not in all_players.columns:
                logger.warning("No predicted points available and no model loaded")
                return pd.DataFrame()
            
            # Exclude current team players and explicitly excluded players
            current_team_ids = current_team['id'].tolist() if 'id' in current_team.columns else []
            
            if exclude_ids is None:
                exclude_ids = []
                
            exclude_all = current_team_ids + exclude_ids
            available_players = all_players[~all_players['id'].isin(exclude_all)].copy()
            
            # Get position counts in current team
            if 'element_type' in current_team.columns:
                position_counts = current_team['element_type'].value_counts().to_dict()
            else:
                # Default position distribution if unknown
                position_counts = {1: 2, 2: 5, 3: 5, 4: 3}
            
            # Calculate value metric (predicted points per cost)
            if 'now_cost' in available_players.columns:
                available_players['value'] = available_players['predicted_points'] / (available_players['now_cost'] / 10)
            
            # Sort by predicted points
            available_players = available_players.sort_values('predicted_points', ascending=False)
            
            # Find players to transfer out (lowest predicted points in current team)
            if 'predicted_points' not in current_team.columns:
                if 'xP_next_n' in current_team.columns:
                    current_team = current_team.rename(columns={'xP_next_n': 'predicted_points'})
                elif self.model is not None:
                    current_team = self.predict_points(current_team)
                else:
                    # Just use total_points if no predictions are available
                    if 'total_points' in current_team.columns:
                        current_team['predicted_points'] = current_team['total_points']
                    else:
                        logger.warning("No points metric available for current team")
                        return pd.DataFrame()
            
            current_team_sorted = current_team.sort_values('predicted_points')
            
            # Select candidates to transfer out (one from each position to provide options)
            transfer_out_candidates = []
            
            for pos in sorted(position_counts.keys()):
                pos_players = current_team_sorted[current_team_sorted['element_type'] == pos]
                if not pos_players.empty:
                    # Add lowest scoring player in this position as a candidate
                    transfer_out_candidates.append(pos_players.iloc[0])
            
            # Prepare recommendations
            recommendations = []
            
            for out_player in transfer_out_candidates:
                pos = out_player['element_type']
                out_cost = out_player['now_cost'] / 10 if 'now_cost' in out_player else 0
                
                # Find replacement candidates in same position with budget constraint
                pos_candidates = available_players[
                    (available_players['element_type'] == pos) & 
                    (available_players['now_cost'] / 10 <= out_cost + budget)
                ]
                
                # Get top candidates by predicted points
                top_candidates = pos_candidates.head(3)
                
                for _, candidate in top_candidates.iterrows():
                    in_cost = candidate['now_cost'] / 10 if 'now_cost' in candidate else 0
                    cost_diff = in_cost - out_cost
                    points_diff = candidate['predicted_points'] - out_player['predicted_points']
                    
                    recommendations.append({
                        'out_id': out_player['id'],
                        'out_name': out_player.get('web_name', out_player.get('full_name', f"Player {out_player['id']}")),
                        'out_team': out_player.get('team', 0),
                        'out_points': out_player['predicted_points'],
                        'out_cost': out_cost,
                        'in_id': candidate['id'],
                        'in_name': candidate.get('web_name', candidate.get('full_name', f"Player {candidate['id']}")),
                        'in_team': candidate.get('team', 0),
                        'in_points': candidate['predicted_points'],
                        'in_cost': in_cost,
                        'cost_diff': cost_diff,
                        'points_diff': points_diff,
                        'position': pos,
                        'value': points_diff / max(0.1, cost_diff) if cost_diff > 0 else points_diff * 10
                    })
            
            # Convert to DataFrame and sort by points difference
            recommendations_df = pd.DataFrame(recommendations)
            if not recommendations_df.empty:
                recommendations_df = recommendations_df.sort_values('points_diff', ascending=False)
            
            return recommendations_df
            
        except Exception as e:
            logger.error(f"Error finding transfer targets: {e}")
            return pd.DataFrame()
    
    def optimize_transfers(self, current_team: pd.DataFrame,
                         all_players: pd.DataFrame,
                         budget: float,
                         num_transfers: int = 1,
                         strategy: str = "greedy") -> Dict[str, Any]:
        """
        Find the optimal set of transfers to maximize predicted points.
        
        Args:
            current_team (pd.DataFrame): DataFrame containing current team players.
            all_players (pd.DataFrame): DataFrame with all available players and predictions.
            budget (float): Available transfer budget in millions.
            num_transfers (int): Number of transfers to make.
            strategy (str): Optimization strategy ("greedy" or "batch").
            
        Returns:
            Dict[str, Any]: Dictionary with optimal transfers and predicted improvement.
        """
        try:
            # Get initial recommendations
            recommendations = self.find_transfer_targets(
                current_team, all_players, budget, num_transfers
            )
            
            if recommendations.empty:
                logger.warning("No transfer recommendations found")
                return {"transfers": [], "total_points_gain": 0.0, "remaining_budget": budget}
            
            if strategy == "greedy":
                # Simple greedy approach: take the top N recommendations by points difference
                optimal_transfers: List[Dict[str, Any]] = []
                total_points_gain = 0.0
                remaining_budget = budget
                excluded_ids: List[int] = []
                
                for i in range(num_transfers):
                    # Get recommendations excluding players already in our transfer plan
                    if i > 0:
                        recommendations = self.find_transfer_targets(
                            current_team, all_players, remaining_budget, 1, excluded_ids
                        )
                        
                        if recommendations.empty:
                            break
                    
                    # Get the top recommendation
                    best_transfer = recommendations.iloc[0]
                    
                    # Check if we have enough budget
                    if best_transfer['cost_diff'] <= remaining_budget:
                        optimal_transfers.append({
                            'out': {
                                'id': best_transfer['out_id'],
                                'name': best_transfer['out_name'],
                                'team': best_transfer['out_team'],
                                'predicted_points': best_transfer['out_points'],
                                'cost': best_transfer['out_cost']
                            },
                            'in': {
                                'id': best_transfer['in_id'],
                                'name': best_transfer['in_name'],
                                'team': best_transfer['in_team'],
                                'predicted_points': best_transfer['in_points'],
                                'cost': best_transfer['in_cost']
                            },
                            'points_gain': best_transfer['points_diff'],
                            'cost_change': best_transfer['cost_diff']
                        })
                        
                        total_points_gain += best_transfer['points_diff']
                        remaining_budget -= best_transfer['cost_diff']
                        
                        # Add to excluded IDs for next iteration
                        excluded_ids.append(best_transfer['out_id'])
                        excluded_ids.append(best_transfer['in_id'])
                    
                return {
                    "transfers": optimal_transfers,
                    "total_points_gain": total_points_gain,
                    "remaining_budget": remaining_budget
                }
                
            elif strategy == "batch":
                # Batch optimization: consider multiple transfers together
                # This is more sophisticated than the greedy approach
                
                # Try different combinations of transfers up to num_transfers
                best_batch_transfers = []
                best_batch_points_gain = 0.0
                max_batch_size = min(num_transfers, 3)  # Limit batch size for computational reasons
                
                # Try different starting points
                for start_pos in range(min(5, len(recommendations))):
                    batch_transfers = []
                    batch_points_gain = 0.0
                    remaining_budget = budget
                    excluded_ids = []
                    
                    # Add first transfer from this starting point
                    first_transfer = recommendations.iloc[start_pos]
                    
                    if first_transfer['cost_diff'] <= remaining_budget:
                        batch_transfers.append({
                            'out': {
                                'id': first_transfer['out_id'],
                                'name': first_transfer['out_name'],
                                'team': first_transfer['out_team'],
                                'predicted_points': first_transfer['out_points'],
                                'cost': first_transfer['out_cost']
                            },
                            'in': {
                                'id': first_transfer['in_id'],
                                'name': first_transfer['in_name'],
                                'team': first_transfer['in_team'],
                                'predicted_points': first_transfer['in_points'],
                                'cost': first_transfer['in_cost']
                            },
                            'points_gain': first_transfer['points_diff'],
                            'cost_change': first_transfer['cost_diff']
                        })
                        
                        batch_points_gain += first_transfer['points_diff']
                        remaining_budget -= first_transfer['cost_diff']
                        
                        # Add to excluded IDs
                        excluded_ids.append(first_transfer['out_id'])
                        excluded_ids.append(first_transfer['in_id'])
                    
                # Update the excluded IDs list with players already included in transfers
                excluded_ids = excluded_ids + [t['out']['id'] for t in optimal_transfers] + [t['in']['id'] for t in optimal_transfers]
                
                # Find next batch of recommendations
                remaining_transfers = max_batch_size - len(batch_transfers)
                if remaining_transfers > 0:
                    more_recommendations = self.find_transfer_targets(
                        current_team, all_players, remaining_budget, 
                        remaining_transfers, excluded_ids
                    )
                    
                    # Add any additional transfers possible
                    if not more_recommendations.empty:
                        for _, transfer in more_recommendations.head(remaining_transfers).iterrows():
                            # Check if we can afford this transfer
                            if transfer['cost_diff'] <= remaining_budget:
                                batch_transfers.append({
                                    'out': {
                                        'id': transfer['out_id'],
                                        'name': transfer['out_name'],
                                        'team': transfer['out_team'],
                                        'predicted_points': transfer['out_points'],
                                        'cost': transfer['out_cost']
                                    },
                                    'in': {
                                        'id': transfer['in_id'],
                                        'name': transfer['in_name'],
                                        'team': transfer['in_team'],
                                        'predicted_points': transfer['in_points'],
                                        'cost': transfer['in_cost']
                                    },
                                    'points_gain': transfer['points_diff'],
                                    'cost_change': transfer['cost_diff']
                                })
                                
                                batch_points_gain += transfer['points_diff']
                                remaining_budget -= transfer['cost_diff']
            
                # Add this batch of transfers to our result if it's better than previous batches
                if batch_points_gain > best_batch_points_gain:
                    best_batch_transfers = batch_transfers
                    best_batch_points_gain = batch_points_gain
        
                # Return the best batch we found
                return {
                    "transfers": best_batch_transfers,
                    "total_points_gain": best_batch_points_gain,
                    "remaining_budget": budget - sum(t['cost_change'] for t in best_batch_transfers)
                }
            
            else:
                logger.warning(f"Unknown optimization strategy: {strategy}")
                return {"transfers": [], "total_points_gain": 0.0, "remaining_budget": budget}
            
        except Exception as e:
            logger.error(f"Error optimizing transfers: {e}")
            return {"transfers": [], "total_points_gain": 0.0, "remaining_budget": budget}

    def recommend_team_strategy(self, current_team: pd.DataFrame,
                              all_players: pd.DataFrame,
                              budget: float = 0.0,
                              num_transfers: int = 1,
                              chip_options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Provide team strategy recommendations including transfers and chip usage.
        
        Args:
            current_team (pd.DataFrame): DataFrame containing current team players.
            all_players (pd.DataFrame): DataFrame with all available players and predictions.
            budget (float): Available transfer budget in millions.
            num_transfers (int): Number of free transfers available.
            chip_options (List[str]): List of available chips ('wildcard', 'freehit', 'triple_captain', 'bench_boost').
            
        Returns:
            Dict[str, Any]: Dictionary with team strategy recommendations.
        """
        if chip_options is None:
            chip_options = []
            
        # Normal transfer recommendations
        normal_transfers = self.optimize_transfers(current_team, all_players, budget, num_transfers)
        
        strategy = {
            "recommended_transfers": normal_transfers["transfers"],
            "expected_points_gain": normal_transfers["total_points_gain"],
            "remaining_budget": normal_transfers["remaining_budget"],
            "chip_recommendation": None,
            "chip_strategy": {}
        }
        
        # Evaluate wildcard scenario if available
        if 'wildcard' in chip_options:
            # With wildcard, we can make unlimited transfers
            # For simplicity, we'll recommend top 15 players within budget constraints
            wildcard_strategy = self._evaluate_wildcard(current_team, all_players)
            strategy["chip_strategy"]["wildcard"] = wildcard_strategy
            
            # If wildcard gain is substantially better, recommend using it
            if wildcard_strategy["points_gain"] > normal_transfers["total_points_gain"] * 2:
                strategy["chip_recommendation"] = "wildcard"
        
        # Evaluate other chips as needed
        if 'freehit' in chip_options:
            freehit_strategy = self._evaluate_freehit(current_team, all_players)
            strategy["chip_strategy"]["freehit"] = freehit_strategy
            
            # If freehit gain is substantially better, recommend using it
            if freehit_strategy["points_gain"] > normal_transfers["total_points_gain"] * 1.5:
                strategy["chip_recommendation"] = "freehit"
        
        if 'triple_captain' in chip_options:
            tc_strategy = self._evaluate_triple_captain(current_team, all_players)
            strategy["chip_strategy"]["triple_captain"] = tc_strategy
            
            # If the TC gain is significant and better than other chips
            if tc_strategy["points_gain"] > 8 and (
                strategy["chip_recommendation"] is None or 
                tc_strategy["points_gain"] > strategy["chip_strategy"].get(
                    strategy["chip_recommendation"], {}).get("points_gain", 0) * 0.8
            ):
                strategy["chip_recommendation"] = "triple_captain"
                
        if 'bench_boost' in chip_options:
            bb_strategy = self._evaluate_bench_boost(current_team, all_players)
            strategy["chip_strategy"]["bench_boost"] = bb_strategy
            
            # If the BB gain is significant and better than other chips
            if bb_strategy["points_gain"] > 10 and (
                strategy["chip_recommendation"] is None or 
                bb_strategy["points_gain"] > strategy["chip_strategy"].get(
                    strategy["chip_recommendation"], {}).get("points_gain", 0) * 0.9
            ):
                strategy["chip_recommendation"] = "bench_boost"
        
        return strategy
    
    def _evaluate_wildcard(self, current_team: pd.DataFrame, all_players: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the potential benefit of using a wildcard chip.
        
        Args:
            current_team (pd.DataFrame): Current team.
            all_players (pd.DataFrame): All available players.
            
        Returns:
            Dict[str, Any]: Wildcard evaluation results.
        """
        try:
            # Ensure we have predicted points
            if 'predicted_points' not in all_players.columns and 'xP_next_n' in all_players.columns:
                all_players = all_players.rename(columns={'xP_next_n': 'predicted_points'})
            elif 'predicted_points' not in all_players.columns and self.model is not None:
                all_players = self.predict_points(all_players)
            
            # Calculate current team total predicted points
            if 'predicted_points' not in current_team.columns:
                if 'xP_next_n' in current_team.columns:
                    current_team = current_team.rename(columns={'xP_next_n': 'predicted_points'})
                elif self.model is not None:
                    current_team = self.predict_points(current_team)
                    
            current_total_points = current_team['predicted_points'].sum()
            
            # Create an optimal team (simple version - just pick top players by position)
            # In a real implementation, this would need to respect team constraints (max 3 per team)
            # and budget constraints
            
                
            # Simple team selection - just to demonstrate
            optimal_team = []
            
            # Select 2 goalkeepers
            gkps = all_players[all_players['element_type'] == 1].sort_values('predicted_points', ascending=False).head(2)
            optimal_team.append(gkps)
            
            # Select 5 defenders
            defs = all_players[all_players['element_type'] == 2].sort_values('predicted_points', ascending=False).head(5)
            optimal_team.append(defs)
            
            # Select 5 midfielders
            mids = all_players[all_players['element_type'] == 3].sort_values('predicted_points', ascending=False).head(5)
            optimal_team.append(mids)
            
            # Select 3 forwards
            fwds = all_players[all_players['element_type'] == 4].sort_values('predicted_points', ascending=False).head(3)
            optimal_team.append(fwds)
            
            # Combine all players
            optimal_team_df = pd.concat(optimal_team)
            optimal_total_points = optimal_team_df['predicted_points'].sum()
            
            # Calculate potential gain
            points_gain = optimal_total_points - current_total_points
            
            return {
                "optimal_team": optimal_team_df,
                "current_points": current_total_points,
                "optimal_points": optimal_total_points,
                "points_gain": points_gain
            }
            
        except Exception as e:
            logger.error(f"Error evaluating wildcard: {e}")
            return {
                "optimal_team": pd.DataFrame(),
                "current_points": 0,
                "optimal_points": 0,
                "points_gain": 0
            }
            
    def _evaluate_freehit(self, current_team: pd.DataFrame, all_players: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the potential benefit of using a free hit chip.
        
        Args:
            current_team (pd.DataFrame): Current team.
            all_players (pd.DataFrame): All available players.
            
        Returns:
            Dict[str, Any]: Free Hit evaluation results.
        """
        try:
            # Free Hit is similar to Wildcard but just for one gameweek
            # We'll use the same logic as wildcard but with more focus on immediate returns
            
            # Ensure we have predicted points
            if 'predicted_points' not in all_players.columns and 'xP_next_n' in all_players.columns:
                all_players = all_players.rename(columns={'xP_next_n': 'predicted_points'})
            elif 'predicted_points' not in all_players.columns and self.model is not None:
                all_players = self.predict_points(all_players)
            
            # For Free Hit, we should prioritize the next GW points rather than season-long value
            # Let's focus on the next fixture difficulty
            if 'next_gw_difficulty' in all_players.columns:
                # Adjust predicted points based on next fixture difficulty
                all_players['adjusted_points'] = all_players['predicted_points'] * (6 - all_players['next_gw_difficulty']) / 3
            else:
                all_players['adjusted_points'] = all_players['predicted_points']
                
            # Calculate current team total predicted points
            if 'predicted_points' not in current_team.columns:
                if 'xP_next_n' in current_team.columns:
                    current_team = current_team.rename(columns={'xP_next_n': 'predicted_points'})
                elif self.model is not None:
                    current_team = self.predict_points(current_team)
                    
            current_total_points = current_team['predicted_points'].sum() / 5  # Divide by 5 to get single GW estimate
            
                
            # Select optimal team for free hit
            freehit_team = []
            
            # Select 2 goalkeepers (though only the first matters for free hit)
            gkps = all_players[all_players['element_type'] == 1].sort_values('adjusted_points', ascending=False).head(2)
            freehit_team.append(gkps)
            
            # Select 5 defenders
            defs = all_players[all_players['element_type'] == 2].sort_values('adjusted_points', ascending=False).head(5)
            freehit_team.append(defs)
            
            # Select 5 midfielders
            mids = all_players[all_players['element_type'] == 3].sort_values('adjusted_points', ascending=False).head(5)
            freehit_team.append(mids)
            
            # Select 3 forwards
            fwds = all_players[all_players['element_type'] == 4].sort_values('adjusted_points', ascending=False).head(3)
            freehit_team.append(fwds)
            
            # Combine all players
            freehit_team_df = pd.concat(freehit_team)
            freehit_total_points = freehit_team_df['adjusted_points'].sum()
            
            # Calculate potential gain for a single gameweek
            points_gain = freehit_total_points - current_total_points
            
            return {
                "freehit_team": freehit_team_df,
                "current_points": current_total_points,
                "freehit_points": freehit_total_points,
                "points_gain": points_gain
            }
            
        except Exception as e:
            logger.error(f"Error evaluating free hit: {e}")
            return {
                "freehit_team": pd.DataFrame(),
                "current_points": 0,
                "freehit_points": 0,
                "points_gain": 0
            }
    
    def _evaluate_triple_captain(self, current_team: pd.DataFrame, all_players: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the potential benefit of using a triple captain chip.
        
        Args:
            current_team (pd.DataFrame): Current team.
            all_players (pd.DataFrame): All available players.
            
        Returns:
            Dict[str, Any]: Triple Captain evaluation results.
        """
        try:
            # Ensure we have predicted points
            if 'predicted_points' not in current_team.columns:
                if 'xP_next_n' in current_team.columns:
                    current_team = current_team.rename(columns={'xP_next_n': 'predicted_points'})
                elif self.model is not None:
                    current_team = self.predict_points(current_team)
            
            # Triple captain should be used on a player with a good fixture
            # Ideally with info about next fixture difficulty
            
            # Calculate base points for each player in one gameweek
            current_team['gw_points'] = current_team['predicted_points'] / 5  # Rough estimate
            
            # Adjust based on fixture difficulty if available
            if 'next_gw_difficulty' in current_team.columns:
                current_team['tc_potential'] = current_team['gw_points'] * (6 - current_team['next_gw_difficulty']) / 3
            else:
                current_team['tc_potential'] = current_team['gw_points']
            
            # Find best captain choice
            tc_candidate = current_team.sort_values('tc_potential', ascending=False).iloc[0]
            
            # Calculate extra points from TC (2x the captain points)
            tc_gain = tc_candidate['tc_potential'] * 2  # Triple instead of regular captain
            
            return {
                "player": tc_candidate.get('web_name', f"Player {tc_candidate.get('id', 0)}"),
                "player_id": tc_candidate.get('id', 0),
                "base_points": tc_candidate['tc_potential'],
                "tc_points": tc_candidate['tc_potential'] * 3,  # Total TC points
                "points_gain": tc_gain
            }
            
        except Exception as e:
            logger.error(f"Error evaluating triple captain: {e}")
            return {
                "player": "Unknown",
                "player_id": 0,
                "base_points": 0,
                "tc_points": 0,
                "points_gain": 0
            }
    
    def _evaluate_bench_boost(self, current_team: pd.DataFrame, all_players: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the potential benefit of using a bench boost chip.
        
        Args:
            current_team (pd.DataFrame): Current team.
            all_players (pd.DataFrame): All available players.
            
        Returns:
            Dict[str, Any]: Bench Boost evaluation results.
        """
        try:
            # Ensure we have predicted points
            if 'predicted_points' not in current_team.columns:
                if 'xP_next_n' in current_team.columns:
                    current_team = current_team.rename(columns={'xP_next_n': 'predicted_points'})
                elif self.model is not None:
                    current_team = self.predict_points(current_team)
            
            # For bench boost, we need to know which players would be on the bench
            # We'll assume the lowest scoring 4 players would be benched
            
            # Calculate single gameweek points
            current_team['gw_points'] = current_team['predicted_points'] / 5  # Rough estimate
            
            # Sort by predicted points
            sorted_team = current_team.sort_values('gw_points', ascending=False)
            
            # First 11 are starters, last 4 are bench
            starters = sorted_team.iloc[:11]
            bench = sorted_team.iloc[11:]
            
            # Calculate bench points
            bench_points = bench['gw_points'].sum()
            
            return {
                "bench_players": bench,
                "starter_points": starters['gw_points'].sum(),
                "bench_points": bench_points,
                "points_gain": bench_points
            }
            
        except Exception as e:
            logger.error(f"Error evaluating bench boost: {e}")
            return {
                "bench_players": pd.DataFrame(),
                "starter_points": 0,
                "bench_points": 0,
                "points_gain": 0
            }
