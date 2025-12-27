"""
Daily.co Browser Streaming Bot

This bot joins a Daily.co room and streams browser screenshots as video frames.
Based on the webbot/videotest.py implementation, adapted for browser-use (Playwright).
"""

import asyncio
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
    """
    
    def __init__(self, session_id: str, browser, meeting_url: str, framerate: int = 30, width: int = 1280, height: int = 720):
        """
        Initialize the browser streamer.
        
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
        
        # Create camera device
        self.camera = Daily.create_camera_device(
            f"browser-camera-{session_id[:8]}",
            width=width,
            height=height,
            color_format="RGB"
        )
        
        # Create Daily client
        self.client = CallClient()
        
        # Configure client to not subscribe to others' audio/video
        self.client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "unsubscribed"}}
        )
        
        self.app_quit = False
        self.app_error = None
        self.start_event = threading.Event()
        self.stream_thread = None
        
    def on_joined(self, data, error):
        """Callback when bot joins the room."""
        if error:
            logger.error(f"Unable to join meeting: {error}")
            self.app_error = error
        else:
            logger.info(f"✅ Bot joined Daily room successfully")
        self.start_event.set()
    
    def run(self):
        """
        Join the Daily room and start streaming.
        This method is synchronous and blocks until streaming stops.
        """
        # Join the room (synchronous call from Daily SDK)
        self.client.join(
            self.meeting_url,
            client_settings={
                "inputs": {
                    "camera": {"isEnabled": True, "settings": {"deviceId": f"browser-camera-{self.session_id[:8]}"}},
                    "microphone": False,
                }
            },
            completion=self.on_joined,
        )
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_frames, daemon=True)
        self.stream_thread.start()
        
        # Wait for thread to complete (or until quit)
        self.stream_thread.join()
    
    def _stream_frames(self):
        """
        Continuously capture browser screenshots and send as video frames.
        Runs in a separate thread.
        """
        # Wait for join to complete
        self.start_event.wait()
        
        if self.app_error:
            logger.error(f"Unable to stream frames: {self.app_error}")
            return
        
        sleep_time = 1.0 / self.framerate
        
        logger.info(f"🎥 Starting to stream browser frames at {self.framerate} FPS")
        
        while not self.app_quit:
            try:
                # Run async screenshot capture in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Get current page
                    page = loop.run_until_complete(self.browser.get_current_page())
                    if not page:
                        logger.warning("No active page, skipping frame")
                        time.sleep(sleep_time)
                        continue
                    
                    # Capture screenshot
                    screenshot_b64 = loop.run_until_complete(page.screenshot())
                    
                    # Decode base64 to image
                    import base64
                    screenshot_bytes = base64.b64decode(screenshot_b64)
                    image = Image.open(io.BytesIO(screenshot_bytes))
                    
                    # Resize if needed to match camera dimensions
                    if image.size != (self.width, self.height):
                        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
                    
                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    # Convert to bytes and send
                    image_bytes = image.tobytes()
                    self.camera.write_frame(image_bytes)
                    
                finally:
                    loop.close()
                
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
            
            time.sleep(sleep_time)
        
        logger.info("🛑 Stopped streaming browser frames")
    
    def leave(self):
        """Leave the room and cleanup."""
        self.app_quit = True
        if self.stream_thread:
            self.stream_thread.join(timeout=5.0)
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

