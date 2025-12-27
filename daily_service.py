"""
Daily.co Room Management Service

This service handles creating and managing Daily.co rooms for browser streaming.
"""

import asyncio
import logging
import os
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)


class DailyService:
    """Service for creating and managing Daily.co rooms."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Daily service.
        
        Args:
            api_key: Daily.co API key (or use DAILY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("DAILY_API_KEY")
        if not self.api_key:
            raise ValueError("DAILY_API_KEY environment variable or api_key parameter is required")
        
        self.base_url = "https://api.daily.co/v1"
        
    async def create_room(
        self, 
        name: Optional[str] = None,
        privacy: str = "public",
        properties: Optional[dict] = None
    ) -> dict:
        """
        Create a new Daily.co room.
        
        Args:
            name: Room name (auto-generated if not provided)
            privacy: Room privacy setting ('public' or 'private')
            properties: Additional room properties
            
        Returns:
            dict: Room information including URL and name
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "privacy": privacy,
        }
        
        if name:
            payload["name"] = name
            
        if properties:
            payload["properties"] = properties
        else:
            # Default properties for browser streaming
            payload["properties"] = {
                "enable_screenshare": True,
                "enable_chat": False,
                "enable_knocking": False,
                "start_video_off": False,
                "start_audio_off": True,
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/rooms",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Created Daily room: {data.get('name')}")
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create room: {response.status} - {error_text}")
                        raise Exception(f"Failed to create Daily room: {response.status}")
        except Exception as e:
            logger.error(f"Error creating Daily room: {e}")
            raise
    
    async def get_room(self, room_name: str) -> dict:
        """
        Get information about a Daily.co room.
        
        Args:
            room_name: Name of the room
            
        Returns:
            dict: Room information
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/rooms/{room_name}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Failed to get room: {response.status}")
        except Exception as e:
            logger.error(f"Error getting room: {e}")
            raise
    
    async def delete_room(self, room_name: str) -> bool:
        """
        Delete a Daily.co room.
        
        Args:
            room_name: Name of the room to delete
            
        Returns:
            bool: True if successful
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/rooms/{room_name}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Deleted Daily room: {room_name}")
                        return True
                    else:
                        logger.error(f"Failed to delete room: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error deleting room: {e}")
            return False
    
    async def create_meeting_token(
        self,
        room_name: str,
        properties: Optional[dict] = None
    ) -> str:
        """
        Create a meeting token for a room.
        
        Args:
            room_name: Name of the room
            properties: Token properties (permissions, expiration, etc.)
            
        Returns:
            str: Meeting token
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "properties": properties or {
                "room_name": room_name,
                "is_owner": True,
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/meeting-tokens",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("token", "")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create token: {response.status} - {error_text}")
                        raise Exception(f"Failed to create meeting token: {response.status}")
        except Exception as e:
            logger.error(f"Error creating meeting token: {e}")
            raise


# Singleton instance
_daily_service_instance: Optional[DailyService] = None


def get_daily_service(api_key: Optional[str] = None) -> Optional[DailyService]:
    """Get or create the Daily service singleton instance."""
    global _daily_service_instance
    
    if _daily_service_instance is None:
        try:
            _daily_service_instance = DailyService(api_key)
        except ValueError as e:
            logger.warning(f"Could not initialize Daily service: {e}")
            return None
    
    return _daily_service_instance

