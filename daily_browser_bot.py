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
        
        # Create a single event loop for this thread (reuse it for async browser operations)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        frame_count = 0
        
        try:
            while not self.app_quit:
                try:
                    # Get current page (async operation)
                    page = loop.run_until_complete(self.browser.get_current_page())
                    if not page:
                        logger.warning("No active page, skipping frame")
                        time.sleep(sleep_time)
                        continue
                    
                    # Capture screenshot (async operation)
                    screenshot_data = loop.run_until_complete(page.screenshot())
                    
                    if not screenshot_data:
                        logger.warning("Empty screenshot data, skipping frame")
                        time.sleep(sleep_time)
                        continue
                    
                    # Handle both base64 string and bytes
                    if isinstance(screenshot_data, str):
                        screenshot_bytes = base64.b64decode(screenshot_data)
                    else:
                        screenshot_bytes = screenshot_data
                    
                    if not screenshot_bytes or len(screenshot_bytes) == 0:
                        logger.warning("Empty screenshot bytes, skipping frame")
                        time.sleep(sleep_time)
                        continue
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(screenshot_bytes))
                    
                    # Resize if needed to match camera dimensions
                    if image.size != (self.width, self.height):
                        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
                    
                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    # Convert to bytes (EXACTLY like guide.py line 71)
                    image_bytes = image.tobytes()
                    
                    # Write frame to camera (EXACTLY like guide.py line 74)
                    # Simple, synchronous call - no try/except needed, let errors propagate
                    self.camera.write_frame(image_bytes)
                    frame_count += 1
                    
                    if frame_count % (self.framerate * 2) == 0:  # Log every 2 seconds
                        logger.info(f"✅ Sent {frame_count} frames")
                    
                except Exception as e:
                    logger.error(f"Error capturing/sending frame: {e}")
                    # Continue loop even on error
                
                # Sleep between frames (EXACTLY like guide.py line 75)
                time.sleep(sleep_time)
        
        finally:
            # Clean up event loop
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

