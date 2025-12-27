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
from typing import Optional

from daily import CallClient, Daily
from PIL import Image

logger = logging.getLogger(__name__)


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
        
        # Ensure Daily is initialized before creating camera device
        # (In webbot, Daily.init() is called in main() before creating SendBrowserApp)
        try:
            Daily.init()
            logger.info("✅ Daily SDK initialized in streamer")
        except Exception as e:
            logger.warning(f"Daily SDK already initialized or warning: {e}")
        
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
        
        # Start streaming thread BEFORE joining (EXACTLY like webbot line 257-258)
        self.stream_thread = threading.Thread(target=self.send_frames)
        self.stream_thread.start()
        logger.info("✅ Streaming thread started")
        
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
        Thread is already running from __init__.
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
        # Wait for thread to complete (EXACTLY like webbot line 277)
        self.stream_thread.join()
    
    def send_frames(self):
        """
        Continuously capture browser screenshots and send as video frames.
        EXACTLY matches webbot/videotest.py send_frames() method (line 287-319).
        """
        # Wait for join to complete (EXACTLY like webbot line 288)
        self.start_event.wait()
        
        if self.app_error:
            logger.error(f"Unable to send frames!")
            return
        
        sleep_time = 1.0 / self.framerate
        
        logger.info(f"🎥 Starting to stream browser frames at {self.framerate} FPS")
        
        # Create a single event loop for this thread (reuse it)
        loop = None
        frame_count = 0
        
        while not self.app_quit:
            try:
                # Create/reuse event loop for async operations
                if loop is None or loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Get current page
                page = loop.run_until_complete(self.browser.get_current_page())
                if not page:
                    logger.warning("No active page, skipping frame")
                    time.sleep(sleep_time)
                    continue
                
                # Capture screenshot - browser-use page.screenshot() returns base64 string
                # We need to decode it to get bytes (like webbot's get_screenshot_as_png returns bytes)
                screenshot_data = loop.run_until_complete(page.screenshot())
                
                if not screenshot_data:
                    logger.warning("Empty screenshot data, skipping frame")
                    time.sleep(sleep_time)
                    continue
                
                # Handle both base64 string and bytes
                if isinstance(screenshot_data, str):
                    # It's base64 encoded, decode it
                    try:
                        screenshot_bytes = base64.b64decode(screenshot_data)
                    except Exception as decode_error:
                        logger.error(f"Failed to decode base64 screenshot: {decode_error}")
                        time.sleep(sleep_time)
                        continue
                else:
                    # It's already bytes
                    screenshot_bytes = screenshot_data
                
                if not screenshot_bytes or len(screenshot_bytes) == 0:
                    logger.warning("Empty screenshot bytes, skipping frame")
                    time.sleep(sleep_time)
                    continue
                
                # Convert to PIL Image (EXACTLY like webbot line 302)
                try:
                    image = Image.open(io.BytesIO(screenshot_bytes))
                except Exception as image_error:
                    logger.error(f"Failed to open image: {image_error}")
                    time.sleep(sleep_time)
                    continue
                
                # Resize if needed to match camera dimensions (EXACTLY like webbot line 305-306)
                if image.size != (self.width, self.height):
                    image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed (EXACTLY like webbot line 309-310)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Convert to bytes and send (EXACTLY like webbot line 313-314)
                image_bytes = image.tobytes()
                
                # Verify image bytes size matches expected (width * height * 3 for RGB)
                expected_size = self.width * self.height * 3
                if len(image_bytes) != expected_size:
                    logger.warning(f"Image bytes size mismatch: got {len(image_bytes)}, expected {expected_size}")
                
                # Write frame to camera
                try:
                    self.camera.write_frame(image_bytes)
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames (1 second at 30fps)
                        logger.debug(f"Sent {frame_count} frames")
                except Exception as write_error:
                    logger.error(f"Failed to write frame: {write_error}")
                
            except Exception as e:
                logger.error(f"Error capturing frame: {e}", exc_info=True)
            
            time.sleep(sleep_time)
        
        # Clean up event loop
        if loop and not loop.is_closed():
            loop.close()
        
        logger.info(f"🛑 Stopped streaming browser frames (sent {frame_count} total frames)")
    
    def leave(self):
        """Leave the room and cleanup. EXACTLY like webbot line 279-285."""
        self.app_quit = True
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
    framerate: int = 30,
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

