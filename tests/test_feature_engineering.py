"""
Unit tests for feature_engineering.py
"""

import pytest
import pandas as pd
import numpy as np
from feature_engineering import (
    create_form_features,
    create_fixture_difficulty_features,
    create_team_strength_features,
    create_positional_features,
    create_value_features,
)


def test_create_form_features():
    player_df = pd.DataFrame({
        "form": ["5.0", "3.2", "0.0"],
        "now_cost": [100, 85, 75],
        "event_points": [8, 2, 0],
        "history": [
            [{"total_points": 4}, {"total_points": 6}, {"total_points": 5}, {"total_points": 7}, {"total_points": 8}],
            [{"total_points": 2}, {"total_points": 1}, {"total_points": 3}, {"total_points": 2}, {"total_points": 1}],
            []
        ]
    })

    result = create_form_features(player_df)

    assert "form_numeric" in result.columns
    assert "form_per_cost" in result.columns
    assert "recent_points_trend" in result.columns
    assert "points_std" in result.columns
    assert "consistency_ratio" in result.columns

    assert result["form_numeric"].iloc[0] == 5.0
    assert result["form_per_cost"].iloc[0] == 5.0 / 10  # now_cost is 100 (10 in millions)
    assert np.isnan(result["points_std"].iloc[2])  # Empty history should result in NaN


def test_create_fixture_difficulty_features():
    player_df = pd.DataFrame({"team": [1, 2, 3]})
    fixture_df = pd.DataFrame({
        "team_h": [1, 2],
        "team_a": [3, 1],
        "team_h_difficulty": [3, 4],
        "team_a_difficulty": [2, 5],
        "event": [1, 2],
        "finished": [False, False]
    })
    team_df = pd.DataFrame({"id": [1, 2, 3], "strength": [5, 4, 3]})

    result = create_fixture_difficulty_features(player_df, fixture_df, team_df)

    assert "next_fixture_difficulty" in result.columns
    assert "avg_difficulty_next_n" in result.columns
    assert "fixture_ease_score" in result.columns
    assert "double_gameweeks" in result.columns

    assert result["next_fixture_difficulty"].iloc[0] == 3  # Team 1's next fixture difficulty
    assert result["next_fixture_difficulty"].iloc[2] == 2  # Team 3's next fixture difficulty


def test_create_team_strength_features():
    player_df = pd.DataFrame({"team": [1, 2, 3]})
    team_df = pd.DataFrame({
        "id": [1, 2, 3],
        "strength": [5, 4, 3],
        "strength_attack_home": [6, 5, 4],
        "strength_attack_away": [5, 4, 3],
        "strength_defence_home": [7, 6, 5],
        "strength_defence_away": [6, 5, 4],
    })

    result = create_team_strength_features(player_df, team_df)

    assert "team_strength" in result.columns
    assert "team_attack_strength" in result.columns
    assert "team_defence_strength" in result.columns

    assert result["team_strength"].iloc[0] == 5
    assert result["team_attack_strength"].iloc[0] == (6 + 5) / 2
    assert result["team_defence_strength"].iloc[0] == (7 + 6) / 2


def test_create_positional_features():
    player_df = pd.DataFrame({
        "element_type": [1, 2, 3, 4],
        "total_points": [100, 120, 150, 180],
        "now_cost": [50, 60, 80, 90],
        "saves": [30, 0, 0, 0],
        "clean_sheets": [10, 15, 5, 2],
        "goals_scored": [0, 2, 10, 15],
        "assists": [0, 3, 8, 5],
    })

    result = create_positional_features(player_df)

    assert "position" in result.columns
    assert "points_vs_position_avg" in result.columns
    assert "value_vs_position_avg" in result.columns

    assert "save_points_potential" in result.columns
    assert "def_attacking_potential" in result.columns
    assert "mid_attacking_value" in result.columns
    assert "fwd_attacking_value" in result.columns

    assert result["position"].iloc[0] == "GKP"
    assert result["position"].iloc[1] == "DEF"
    assert result["position"].iloc[2] == "MID"
    assert result["position"].iloc[3] == "FWD"


def test_create_value_features():
    player_df = pd.DataFrame({
        "now_cost": [100, 85, 75],
        "total_points": [200, 180, 160],
        "form_numeric": [10, 8, 7],
        "xP_next_n": [50, 40, 30],
    })

    result = create_value_features(player_df)

    assert "price" in result.columns
    assert "points_per_million" in result.columns
    assert "form_per_million" in result.columns
    assert "xP_per_million" in result.columns

    assert result["points_per_million"].iloc[0] == 200 / 10  # 100 cost -> 10 million
    assert result["form_per_million"].iloc[1] == 8 / 8.5  # 85 cost -> 8.5 million
    assert result["xP_per_million"].iloc[2] == 30 / 7.5  # 75 cost -> 7.5 million
