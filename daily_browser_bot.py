"""
Daily.co Browser Streaming Bot

This bot joins a Daily.co room and streams browser screenshots as video frames.
EXACTLY matches webbot/videotest.py implementation, adapted for browser-use (Playwright).
"""

import asyncio
import base64
import io
import logging
import threading
import time
import os
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from daily import CallClient, Daily
from PIL import Image

logger = logging.getLogger(__name__)

# Daily API configuration
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_BASE_URL = "https://api.daily.co/v1"


def extract_room_name_from_url(room_url: str) -> Optional[str]:
    """
    Extract room name from Daily.co room URL.
    
    Examples:
    - https://jobi.daily.co/browser-session-42ecb5cf -> browser-session-42ecb5cf
    """
    try:
        parsed = urlparse(room_url)
        path = parsed.path.strip('/')
        if path:
            return path
        if '/' in room_url:
            return room_url.split('/')[-1]
        return room_url
    except Exception as e:
        logger.error(f"Error extracting room name from URL: {e}")
        return None


async def get_meeting_id_for_room(room_name: str) -> Optional[str]:
    """
    Get meeting ID for a given room name.
    
    Args:
        room_name: Name of the Daily room
        
    Returns:
        Meeting ID or None if not found
    """
    if not DAILY_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DAILY_API_BASE_URL}/meetings",
                headers=headers
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                meetings = data.get("data", [])
                
                for meeting in meetings:
                    if meeting.get("room") == room_name:
                        return meeting.get("id")
                
                return None
    except Exception as e:
        logger.debug(f"Error getting meeting ID: {e}")
        return None


async def get_room_participants(meeting_id: str) -> list:
    """
    Get list of participants currently in a Daily.co meeting.
    
    Args:
        meeting_id: ID of the Daily meeting
        
    Returns:
        list: List of participant dictionaries
    """
    if not DAILY_API_KEY:
        return []
    
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DAILY_API_BASE_URL}/meetings/{meeting_id}/participants",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    participants = data.get("data", data.get("participants", []))
                    return participants
                elif response.status == 404:
                    return []
                else:
                    return []
    except Exception as e:
        logger.debug(f"Error getting room participants: {e}")
        return []


def is_bot_participant(participant: dict) -> bool:
    """
    Check if a participant is a bot (streaming bot or other automated participant).
    
    Args:
        participant: Participant dictionary from Daily API
        
    Returns:
        bool: True if participant appears to be a bot
    """
    user_name = participant.get("user_name")
    user_id = participant.get("user_id")
    participant_id = participant.get("participant_id", "")
    
    # Convert to lowercase strings for checking
    user_name_lower = (user_name or "").lower()
    user_id_lower = (user_id or "").lower()
    participant_id_lower = (participant_id or "").lower()
    
    # Check if it's a bot based on name or user_id
    bot_indicators = ["bot", "guest", "stream", "automated"]
    
    for indicator in bot_indicators:
        if indicator in user_name_lower or indicator in user_id_lower or indicator in participant_id_lower:
            return True
    
    # CRITICAL: If user_name is None/empty AND user_id exists, it's likely a bot
    # The streaming bot typically has user_name=None but has a user_id
    if not user_name and user_id:
        return True
    
    # Also check if participant_id matches user_id (common for bots)
    if user_id and participant_id and user_id == participant_id:
        return True
    
    return False


async def has_non_bot_participants(meeting_url: str, exclude_participant_id: Optional[str] = None) -> bool:
    """
    Check if there are any non-bot participants in the room.
    
    Args:
        meeting_url: Daily.co meeting URL
        exclude_participant_id: Optional participant ID to exclude from the check (e.g., someone who just left)
        
    Returns:
        bool: True if there are non-bot participants, False otherwise
    """
    room_name = extract_room_name_from_url(meeting_url)
    if not room_name:
        return False
    
    meeting_id = await get_meeting_id_for_room(room_name)
    if not meeting_id:
        return False
    
    participants = await get_room_participants(meeting_id)
    logger.info(f"📊 Streaming bot checking participants: found {len(participants)} total participants")
    
    # Filter out the excluded participant if provided
    if exclude_participant_id:
        participants = [p for p in participants if p.get('participant_id') != exclude_participant_id and p.get('id') != exclude_participant_id]
        logger.info(f"📊 After excluding participant {exclude_participant_id[:8]}..., {len(participants)} participants remain")
    
    # Log all participants for debugging
    for p in participants:
        user_name = p.get('user_name', 'None')
        user_id = p.get('user_id', 'None')
        participant_id = p.get('participant_id', p.get('id', 'None'))
        is_bot = is_bot_participant(p)
        logger.info(f"   👤 Participant: name={user_name}, user_id={user_id}, participant_id={participant_id}, is_bot={is_bot}")
    
    # Check if there's at least one non-bot participant
    user_participants = [p for p in participants if not is_bot_participant(p)]
    logger.info(f"📊 Found {len(user_participants)} non-bot participants")
    
    return len(user_participants) > 0


class DailyBrowserStreamer:
    """
    Streams browser screenshots to a Daily.co room.
    EXACTLY matches webbot/videotest.py SendBrowserApp class structure.
    """
    
    def __init__(self, session_id: str, browser, meeting_url: str, framerate: int = 30, width: int = 1280, height: int = 720):
        """
        Initialize the browser streamer.
        Matches webbot/videotest.py __init__ structure exactly.
        
        Args:
            session_id: Browser session ID
            browser: Browser instance from browser-use (Playwright)
            meeting_url: Daily.co meeting URL
            framerate: Frames per second (default: 30)
            width: Video width (default: 1280)
            height: Video height (default: 720)
        """
        self.session_id = session_id
        self.browser = browser
        self.meeting_url = meeting_url
        self.framerate = framerate
        self.width = width
        self.height = height
        self.main_loop = None  # Will store the main event loop
        
        # NOTE: Daily.init() is called in start_daily_bot() before creating this streamer
        # Do NOT call Daily.init() here - it will cause "Execution context already exists" error
        
        # Give browser time to load and render (like webbot does)
        logger.info("Waiting for browser to be ready...")
        time.sleep(3)
        logger.info("Browser ready, starting to stream frames...")
        
        # Create camera device with browser dimensions (EXACTLY like webbot line 243-245)
        try:
            self.camera = Daily.create_camera_device(
                "my-camera", width=width, height=height, color_format="RGB"
            )
            logger.info(f"✅ Camera device created: {width}x{height} RGB")
        except Exception as e:
            logger.error(f"❌ Failed to create camera device: {e}", exc_info=True)
            raise
        
        # Create Daily client (EXACTLY like webbot line 247)
        self.client = CallClient()
        
        # Configure client to not subscribe to others' audio/video (EXACTLY like webbot line 249-251)
        self.client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "unsubscribed"}}
        )
        
        self.app_quit = False
        self.app_error = None
        self.start_event = threading.Event()
        self.stream_thread = None  # Will be started after page is ready (after 200 OK)
        self._stream_started = False
        self.monitor_thread = None  # Thread to monitor participants
        self._monitoring = False  # Flag to control monitoring
        self._leaving = False  # Flag to prevent double-leaving
        
    def on_joined(self, data, error):
        """Callback when bot joins the room. EXACTLY like webbot line 260-264."""
        if error:
            logger.error(f"Unable to join meeting: {error}")
            self.app_error = error
        else:
            logger.info("✅ Bot successfully joined Daily room")
        self.start_event.set()
        logger.info("✅ Start event set, streaming thread can begin")
    
    def run(self):
        """
        Join the Daily room. EXACTLY like webbot line 266-277.
        Streaming thread will be started after page is ready (after 200 OK).
        """
        self.client.join(
            self.meeting_url,
            client_settings={
                "inputs": {
                    "camera": {"isEnabled": True, "settings": {"deviceId": "my-camera"}},
                    "microphone": False,
                }
            },
            completion=self.on_joined,
        )
        # Wait for thread to complete (if it was started)
        if self.stream_thread:
            self.stream_thread.join()
    
    def start_streaming(self, main_loop=None):
        """
        Start the streaming thread. Call this AFTER page is loaded (after 200 OK).
        
        Args:
            main_loop: The main asyncio event loop (from FastAPI). If provided, 
                      screenshots will run in that loop to avoid event loop conflicts.
        """
        if not self._stream_started:
            self.main_loop = main_loop
            logger.info("🎬 Starting streaming thread (page is ready)...")
            self.stream_thread = threading.Thread(target=self.send_frames)
            self.stream_thread.start()
            self._stream_started = True
            logger.info("✅ Streaming thread started")
            
            # Start participant monitoring
            if DAILY_API_KEY:
                self._monitoring = True
                self.monitor_thread = threading.Thread(target=self.monitor_participants, daemon=False, name=f"StreamMonitor-{self.session_id[:8]}")
                self.monitor_thread.start()
                logger.info(f"✅ Participant monitoring started for streaming bot (thread: {self.monitor_thread.name})")
            else:
                logger.warning("⚠️ DAILY_API_KEY not set - participant monitoring disabled for streaming bot")
    
    def send_frames(self):
        """
        Continuously capture browser screenshots and send as video frames.
        EXACTLY matches guide.py send_image() method pattern (line 63-75).
        Simple, synchronous approach - capture screenshot, convert to bytes, send.
        """
        # Wait for join to complete (EXACTLY like guide.py line 64)
        self.start_event.wait()
        
        if self.app_error:
            logger.error(f"Unable to send frames!")
            return
        
        sleep_time = 1.0 / self.framerate
        
        logger.info(f"🎥 Starting to stream browser frames at {self.framerate} FPS")
        
        frame_count = 0
        
        try:
            while not self.app_quit:
                try:
                    # Use main event loop if available, otherwise create new one
                    if self.main_loop and self.main_loop.is_running():
                        # Use run_coroutine_threadsafe to run in main loop
                        future = asyncio.run_coroutine_threadsafe(
                            self.browser.get_current_page(), 
                            self.main_loop
                        )
                        page = future.result(timeout=2.0)
                    else:
                        # Fallback: create new event loop (may have issues)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            page = loop.run_until_complete(
                                asyncio.wait_for(self.browser.get_current_page(), timeout=2.0)
                            )
                        finally:
                            loop.close()
                    
                    if not page:
                        if frame_count == 0:
                            logger.info("Waiting for page to be ready...")
                        time.sleep(sleep_time)
                        continue
                    
                    # Log first successful page access
                    if frame_count == 0:
                        logger.info("✅ Page ready, starting to capture frames...")
                    
                    # Capture screenshot (EXACTLY like webbot line 299 - simple, direct call)
                    logger.info(f"📸 Capturing screenshot for frame {frame_count + 1}...")
                    try:
                        # Use main event loop if available to avoid event loop conflicts
                        if self.main_loop and self.main_loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                page.screenshot(), 
                                self.main_loop
                            )
                            screenshot_data = future.result(timeout=2.0)
                        else:
                            # Fallback: use new event loop
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                screenshot_data = loop.run_until_complete(
                                    asyncio.wait_for(page.screenshot(), timeout=2.0)
                                )
                            finally:
                                loop.close()
                        
                        logger.info(f"📸 Screenshot captured successfully: type={type(screenshot_data).__name__}, length={len(screenshot_data) if screenshot_data else 0}")
                    except asyncio.TimeoutError:
                        logger.error(f"❌ Screenshot TIMEOUT after 2 seconds!")
                        time.sleep(sleep_time)
                        continue
                    except Exception as screenshot_ex:
                        logger.error(f"❌ Screenshot EXCEPTION: {type(screenshot_ex).__name__}: {screenshot_ex}")
                        import traceback
                        logger.error(traceback.format_exc())
                        time.sleep(sleep_time)
                        continue
                    
                    if not screenshot_data:
                        logger.error("❌ Empty screenshot data!")
                        time.sleep(sleep_time)
                        continue
                    
                    # Handle both base64 string and bytes
                    if isinstance(screenshot_data, str):
                        logger.info(f"📸 Decoding base64 screenshot (str length: {len(screenshot_data)})...")
                        screenshot_bytes = base64.b64decode(screenshot_data)
                        logger.info(f"📸 Decoded to {len(screenshot_bytes)} bytes")
                    else:
                        screenshot_bytes = screenshot_data
                        logger.info(f"📸 Screenshot is bytes: {len(screenshot_bytes)} bytes")
                    
                    if not screenshot_bytes or len(screenshot_bytes) == 0:
                        logger.error("❌ Empty screenshot bytes after processing!")
                        time.sleep(sleep_time)
                        continue
                    
                    # Convert to PIL Image (EXACTLY like webbot line 302)
                    logger.info(f"🖼️ Converting to PIL Image from {len(screenshot_bytes)} bytes...")
                    image = Image.open(io.BytesIO(screenshot_bytes))
                    logger.info(f"🖼️ Image opened: {image.size}, mode: {image.mode}")
                    
                    # Resize if needed to match camera dimensions (EXACTLY like webbot line 305-306)
                    if image.size != (self.width, self.height):
                        logger.info(f"🖼️ Resizing from {image.size} to {self.width}x{self.height}...")
                        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
                    
                    # Convert to RGB if needed (EXACTLY like webbot line 309-310)
                    if image.mode != "RGB":
                        logger.info(f"🖼️ Converting from {image.mode} to RGB...")
                        image = image.convert("RGB")
                    
                    # Convert to bytes and send (EXACTLY like webbot line 313-314)
                    logger.info(f"💾 Converting image to bytes...")
                    image_bytes = image.tobytes()
                    expected_size = self.width * self.height * 3
                    logger.info(f"💾 Image bytes: {len(image_bytes)} bytes (expected: {expected_size})")
                    
                    # Write frame to camera (EXACTLY like webbot line 314)
                    logger.info(f"📤 Writing frame {frame_count + 1} to camera...")
                    self.camera.write_frame(image_bytes)
                    frame_count += 1
                    logger.info(f"✅ Sent frame {frame_count}! ({len(image_bytes)} bytes, {image.size[0]}x{image.size[1]})")
                    
                    # Log every 10 frames after first
                    if frame_count > 1 and frame_count % 10 == 0:
                        logger.info(f"✅ Sent {frame_count} frames total")
                    
                except Exception as e:
                    logger.error(f"Error capturing/sending frame: {e}", exc_info=True)
                    # Continue loop even on error
                
                # Sleep between frames (EXACTLY like webbot line 319)
                time.sleep(sleep_time)
        
        finally:
            # Clean up event loop
            loop.close()
        
        logger.info(f"🛑 Stopped streaming browser frames (sent {frame_count} total frames)")
    
    def monitor_participants(self):
        """
        Monitor participants in the room. Stop streaming if no non-bot participants remain.
        This runs in a separate thread and checks periodically.
        Stops when the Gemini bot leaves or when no users remain.
        """
        check_interval = 2.0  # Check every 2 seconds (more frequent)
        logger.info(f"👀 Starting participant monitoring for streaming bot (checking every {check_interval} seconds)")
        
        check_count = 0
        while not self.app_quit and self._monitoring:
            try:
                check_count += 1
                logger.info(f"🔍 Streaming bot monitoring check #{check_count}...")
                
                # Check if there are any non-bot participants
                # Use a new event loop for this async call
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    has_participants = loop.run_until_complete(
                        has_non_bot_participants(self.meeting_url)
                    )
                    logger.info(f"🔍 Streaming bot check result: has_participants={has_participants}")
                finally:
                    loop.close()
                
                if not has_participants:
                    logger.info("👋 No non-bot participants found in room (Gemini bot left or no users). Stopping streaming...")
                    self.app_quit = True
                    self._monitoring = False
                    # Give a moment for the stream thread to finish
                    time.sleep(1.0)
                    # Leave the room (only if not already leaving)
                    if not self._leaving:
                        self._leaving = True
                        try:
                            logger.info("🚪 Leaving Daily room...")
                            self.client.leave()
                            self.client.release()
                            logger.info("✅ Left Daily room (no participants)")
                            
                            # Remove from registry
                            bot_id = f"daily-bot-{self.session_id}"
                            if bot_id in _active_streamers:
                                del _active_streamers[bot_id]
                                logger.info(f"🧹 Removed bot {bot_id} from registry")
                        except Exception as e:
                            logger.error(f"Error leaving room: {e}", exc_info=True)
                    break
                else:
                    logger.info(f"👋 Streaming bot will continue - non-bot participants still present")
                
            except Exception as e:
                logger.error(f"Error checking participants in streaming bot: {e}", exc_info=True)
                # Continue monitoring even on error
            
            # Wait before next check
            time.sleep(check_interval)
        
        logger.info("🛑 Participant monitoring stopped for streaming bot")
    
    def leave(self):
        """Leave the room and cleanup. EXACTLY like webbot line 279-285."""
        if self._leaving:
            logger.debug("Already leaving, skipping...")
            return
        
        self._leaving = True
        self.app_quit = True
        self._monitoring = False  # Stop monitoring
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Wait for stream thread to finish
        if self.stream_thread:
            self.stream_thread.join()
        
        self.client.leave()
        self.client.release()
        logger.info("✅ Bot left Daily room")


# Global registry of active streamers
_active_streamers: dict[str, DailyBrowserStreamer] = {}
_daily_initialized = False


def start_daily_bot(
    session_id: str,
    browser,
    meeting_url: str,
    framerate: int = 10,  # Lower FPS for stability (guide.py uses low FPS)
    width: int = 1280,
    height: int = 720
) -> str:
    """
    Start a Daily streaming bot for a browser session.
    
    Args:
        session_id: Browser session ID
        browser: Browser instance from browser-use
        meeting_url: Daily.co meeting URL
        framerate: Frames per second
        width: Video width
        height: Video height
    
    Returns:
        bot_id: Unique ID for this bot instance
    """
    bot_id = f"daily-bot-{session_id}"
    
    if bot_id in _active_streamers:
        logger.warning(f"Bot already running for session {session_id[:8]}")
        return bot_id
    
    # Initialize Daily (only once globally)
    global _daily_initialized
    if not _daily_initialized:
        try:
            Daily.init()
            _daily_initialized = True
            logger.info("✅ Daily SDK initialized")
        except Exception as e:
            logger.warning(f"Daily SDK initialization warning: {e}")
    
    # Create streamer
    streamer = DailyBrowserStreamer(
        session_id=session_id,
        browser=browser,
        meeting_url=meeting_url,
        framerate=framerate,
        width=width,
        height=height
    )
    
    # Store streamer
    _active_streamers[bot_id] = streamer
    
    # Start bot in background thread (run() is synchronous and blocks)
    bot_thread = threading.Thread(target=streamer.run, daemon=True)
    bot_thread.start()
    
    logger.info(f"✅ Started Daily bot {bot_id} for session {session_id[:8]}")
    return bot_id


def start_streaming_for_bot(bot_id: str, main_loop=None):
    """Start streaming for a bot (call this after page is loaded, after 200 OK).
    
    Args:
        bot_id: Bot ID to start streaming for
        main_loop: Main asyncio event loop (from FastAPI) to run screenshots in
    """
    if bot_id in _active_streamers:
        streamer = _active_streamers[bot_id]
        streamer.start_streaming(main_loop)
        logger.info(f"🎬 Started streaming for bot {bot_id}")
    else:
        logger.warning(f"Bot {bot_id} not found for streaming start")


def stop_daily_bot(bot_id: str):
    """Stop a running Daily bot."""
    if bot_id in _active_streamers:
        streamer = _active_streamers[bot_id]
        streamer.leave()
        del _active_streamers[bot_id]
        logger.info(f"🛑 Stopped Daily bot {bot_id}")


def get_active_daily_bots() -> list[str]:
    """Get list of active bot IDs."""
    return list(_active_streamers.keys())

