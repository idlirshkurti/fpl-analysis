"""
Module for interacting with the Fantasy Premier League API.
Uses the amosbastian/fpl package to fetch player data, teams, and fixtures.
"""

import asyncio
import logging
from typing import Any, Dict, List

import aiohttp
from fpl import FPL

logger = logging.getLogger(__name__)


class FPLAPIClient:
    """Client for interacting with the Fantasy Premier League API."""

    def __init__(self):
        """Initialize the FPL API client."""
        self.session = None
        self.fpl = None

    async def _initialize(self):
        """Initialize the aiohttp session and FPL client."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self.fpl = FPL(self.session)

    async def close(self):
        """Close the aiohttp session."""
        if self.session is not None:
            await self.session.close()
            self.session = None
            self.fpl = None

    async def get_players(self) -> List[Dict[str, Any]]:
        """
        Fetch all players from the FPL API.
        
        Returns:
            List[Dict[str, Any]]: List of player data dictionaries.
        """
        await self._initialize()
        try:
            players = await self.fpl.get_players()
            return [player.__dict__ for player in players]
        except Exception as e:
            logger.error(f"Error fetching players: {e}")
            return []

    async def get_teams(self) -> List[Dict[str, Any]]:
        """
        Fetch all teams from the FPL API.
        
        Returns:
            List[Dict[str, Any]]: List of team data dictionaries.
        """
        await self._initialize()
        try:
            teams = await self.fpl.get_teams()
            return [team.__dict__ for team in teams]
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return []

    async def get_fixtures(self) -> List[Dict[str, Any]]:
        """
        Fetch all fixtures from the FPL API.
        
        Returns:
            List[Dict[str, Any]]: List of fixture data dictionaries.
        """
        await self._initialize()
        try:
            fixtures = await self.fpl.get_fixtures()
            return [fixture.__dict__ for fixture in fixtures]
        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            return []
            
    async def get_user_team(self, user_id: int) -> Dict[str, Any]:
        """
        Fetch a user's team from the FPL API.
        
        Args:
            user_id (int): The FPL user ID.
            
        Returns:
            Dict[str, Any]: User team data dictionary.
        """
        await self._initialize()
        try:
            user = await self.fpl.get_user(user_id)
            team = await user.get_team()
            return {"user": user.__dict__, "team": [player.__dict__ for player in team]}
        except Exception as e:
            logger.error(f"Error fetching user team: {e}")
            return {}
            
    async def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Fetch detailed summary for a specific player.
        
        Args:
            player_id (int): The FPL player ID.
            
        Returns:
            Dict[str, Any]: Player summary data.
        """
        await self._initialize()
        try:
            player = await self.fpl.get_player(player_id)
            summary = await player.get_summary()
            return summary
        except Exception as e:
            logger.error(f"Error fetching player summary: {e}")
            return {}


# Synchronous wrapper functions

def get_player_data() -> List[Dict[str, Any]]:
    """
    Synchronous wrapper to fetch player data.
    
    Returns:
        List[Dict[str, Any]]: List of player data dictionaries.
    """
    client = FPLAPIClient()
    try:
        return asyncio.run(client.get_players())
    finally:
        asyncio.run(client.close())

def get_team_data() -> List[Dict[str, Any]]:
    """
    Synchronous wrapper to fetch team data.
    
    Returns:
        List[Dict[str, Any]]: List of team data dictionaries.
    """
    client = FPLAPIClient()
    try:
        return asyncio.run(client.get_teams())
    finally:
        asyncio.run(client.close())
        
def get_fixture_data() -> List[Dict[str, Any]]:
    """
    Synchronous wrapper to fetch fixture data.
    
    Returns:
        List[Dict[str, Any]]: List of fixture data dictionaries.
    """
    client = FPLAPIClient()
    try:
        return asyncio.run(client.get_fixtures())
    finally:
        asyncio.run(client.close())
        
def get_user_team_data(user_id: int) -> Dict[str, Any]:
    """
    Synchronous wrapper to fetch a user's team data.
    
    Args:
        user_id (int): The FPL user ID.
        
    Returns:
        Dict[str, Any]: User team data dictionary.
    """
    client = FPLAPIClient()
    try:
        return asyncio.run(client.get_user_team(user_id))
    finally:
        asyncio.run(client.close())

