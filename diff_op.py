"""
FastAPI entry point for browser-use agent.
Accepts URL and action, maintains browser session state between requests.
Includes Daily.co streaming capabilities.
"""

import asyncio
import logging
import os
import shutil
import sys
import subprocess
import time
import uuid
import hashlib  # Kaushikh: Added for page state hashing
from typing import Optional

import aiohttp

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatOpenAI, ChatBrowserUse
from browser_use.agent.views import AgentHistoryList

# Load environment variables from .env file
load_dotenv()

# Note: Removed daily_streaming path - no longer needed for Daily.co implementation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Browser-Use Agent API",
    description="API for browser automation using browser-use with OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active agent sessions
# In production, use Redis or a database
active_sessions: dict[str, dict] = {}

# Screenshot cache: {session_id: {"screenshot": base64, "timestamp": time}}
screenshot_cache: dict[str, dict] = {}

# Track when actions complete: {session_id: timestamp}
# Used to force fresh screenshots for a short period after actions
action_completion_times: dict[str, float] = {}

# Track when actions are in progress: {session_id: bool}
# Used to prevent stale screenshots during action execution
action_in_progress: dict[str, bool] = {}

# Headless mode configuration (for Cloud Run / production)
# Automatically set to True in Cloud Run, or set HEADLESS_MODE=true in .env
# For Vast AI and server environments, force headless mode
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() == "true"  # Default to true for servers
# Also check if we're in a server environment (no DISPLAY)
# Force headless mode if DISPLAY is not set and HEADLESS_MODE wasn't explicitly set
if not os.getenv("DISPLAY") and os.getenv("HEADLESS_MODE") is None:
    HEADLESS_MODE = True
    logger.info("🖥️ No DISPLAY found, forcing headless mode")
logger.info(f"🖥️ Headless mode: {HEADLESS_MODE}")

# Server port - configurable via environment variable
SERVER_PORT = int(os.getenv("PORT", "8080"))
logger.info(f"🔌 Server port: {SERVER_PORT}")

# API URL for bot to connect (use localhost for local dev, GCP URL for production)
FASTAPI_URL = os.getenv("FASTAPI_URL", f"http://localhost:{SERVER_PORT}")
logger.info(f"🔗 FastAPI URL for bot: {FASTAPI_URL}")

# Server bot URL - where the Gemini bot server is running
# Set SERVER_BOT_URL environment variable to enable automatic bot joining
SERVER_BOT_URL = os.getenv("SERVER_BOT_URL", None)
if SERVER_BOT_URL:
    logger.info(f"🤖 Server bot URL configured: {SERVER_BOT_URL}")
else:
    logger.info("🤖 Server bot URL not configured (set SERVER_BOT_URL to enable)")

# Simplified browser initialization - matching guide.py approach
# browser-use/Playwright handles Chrome management internally, so we keep it simple


async def notify_server_bot(room_url: str, session_id: str, room_token: Optional[str] = None, agent_id: Optional[str] = None):
    """
    Notify the server bot to join a Daily.co room.
    
    Args:
        room_url: The Daily.co room URL
        session_id: The browser session ID
        room_token: Optional room token (if needed)
        agent_id: Optional agent ID for demo configuration
    """
    if not SERVER_BOT_URL:
        logger.warning("⚠️ SERVER_BOT_URL not configured, skipping server bot notification")
        return
    
    join_url = f"{SERVER_BOT_URL.rstrip('/')}/join-room"
    
    payload = {
        "room_url": room_url,
        "session_id": session_id,
    }
    
    if room_token:
        payload["room_token"] = room_token
    
    if agent_id:
        payload["agent_id"] = agent_id
        logger.info(f"📤 Including agent_id in notification: {agent_id}")
    
    logger.info(f"📤 Sending POST request to server bot: {join_url}")
    logger.info(f"📤 Payload: room_url={room_url[:50]}..., session_id={session_id[:8]}, agent_id={agent_id}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                join_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_text = await response.text()
                logger.info(f"📥 Server bot response status: {response.status}")
                logger.info(f"📥 Server bot response body: {response_text[:200]}")
                
                if response.status == 200:
                    try:
                        result = await response.json() if response_text else {}
                        logger.info(f"✅ Server bot notified successfully: {result.get('message', 'OK')}")
                    except:
                        logger.info(f"✅ Server bot responded with status 200: {response_text[:100]}")
                else:
                    logger.warning(f"⚠️ Server bot notification returned {response.status}: {response_text[:200]}")
    except asyncio.TimeoutError:
        logger.error(f"❌ Server bot notification timed out after 10 seconds")
    except aiohttp.ClientError as e:
        logger.error(f"❌ HTTP error notifying server bot: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to notify server bot: {e}", exc_info=True)

# Chrome args for Docker/Cloud Run/Vast AI environments
# Always include essential args for server environments, even in non-headless mode
# These are required for Chrome to work in containerized/server environments
CHROME_ARGS = [
    '--no-sandbox',  # Required for Docker/Vast AI
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',  # Overcome limited resource problems
    '--disable-gpu',  # Disable GPU acceleration (required for headless, helpful for servers)
    '--disable-software-rasterizer',
    '--disable-extensions',
    '--disable-background-networking',
    '--disable-default-apps',
    '--disable-sync',
    '--metrics-recording-only',
    '--no-first-run',
    '--safebrowsing-disable-auto-update',
    '--disable-blink-features=AutomationControlled',
]

# Add headless flag if in headless mode
if HEADLESS_MODE:
    CHROME_ARGS.append('--headless=new')

logger.info(f"🔧 Chrome args ({len(CHROME_ARGS)} flags): {CHROME_ARGS[:5]}...")  # Log first 5 args
logger.info(f"🔧 Full Chrome args: {CHROME_ARGS}")  # Log all args for debugging


# ============================================================================
# Models for Daily.co Streaming
# ============================================================================

class CreateRoomRequest(BaseModel):
    """Request model for creating a Daily.co room"""
    session_id: str
    room_name: Optional[str] = None


class CreateRoomResponse(BaseModel):
    """Response model for room creation"""
    room_name: str
    url: str
    session_id: str


class GetTokenRequest(BaseModel):
    """Request model for getting Daily.co token"""
    roomName: str
    participantName: str = "viewer"


class GetTokenResponse(BaseModel):
    """Response model for token generation"""
    token: str
    url: str


class ScreenshotResponse(BaseModel):
    """Response model for screenshot"""
    screenshot: str
    session_id: str


# ============================================================================
# Models for Agent Actions
# ============================================================================

class ActionRequest(BaseModel):
    """Request model for agent actions"""
    url: str
    action: str
    session_id: Optional[str] = None  # Optional: if provided, continues existing session
    max_steps: int = 20  # Maximum steps for this action
    agent_id: Optional[str] = None  # Optional: agent ID for demo configuration


class ActionResponse(BaseModel):
    """Response model for agent actions"""
    session_id: str
    urls_visited: list[str]
    daily_room_url: Optional[str] = None  # Daily room URL (if room was auto-created)
    daily_room_name: Optional[str] = None  # Daily room name (if room was auto-created)
    # Kaushikh: Added page state tracking fields
    page_state_hash: Optional[str] = None  # Kaushikh: Hash of DOM content for change detection
    page_title: Optional[str] = None  # Kaushikh: Current page title
    current_url: Optional[str] = None  # Kaushikh: Current URL after action
    dom_element_count: Optional[int] = None  # Kaushikh: Number of interactive elements
    page_changed: Optional[bool] = None  # Kaushikh: Did page content actually change?
    visible_text_preview: Optional[str] = None  # Kaushikh: First ~200 chars of visible text


# Kaushikh: Added page state tracking helper function
async def get_page_state_hash(page) -> dict:
    """
    Kaushikh: Generate a hash representing current page state.
    Used to detect if page content actually changed after an action.
    
    Kaushikh: FIXED - Added more robust error handling and logging.
    URL is captured separately to ensure URL changes are always detected.
    """
    # Kaushikh: Capture URL first - this is the most reliable indicator of navigation
    url = ""
    try:
        url = page.url
        logger.info(f"📍 Kaushikh: Current URL captured: {url}")
    except Exception as e:
        logger.warning(f"⚠️ Kaushikh: Could not get URL: {e}")
    
    # Kaushikh: Capture title
    title = ""
    try:
        title = await page.title()
        logger.info(f"📍 Kaushikh: Current title captured: {title[:50] if title else 'empty'}...")
    except Exception as e:
        logger.warning(f"⚠️ Kaushikh: Could not get title: {e}")
    
    # Kaushikh: Capture text content (may fail on some pages, but that's OK)
    text_content = ""
    try:
        text_content = await page.evaluate('''() => {
            try {
                const walker = document.createTreeWalker(
                    document.body, 
                    NodeFilter.SHOW_TEXT,
                    null, false
                );
                let text = '';
                let node;
                while (node = walker.nextNode()) {
                    const trimmed = node.textContent.trim();
                    if (trimmed.length > 2) text += trimmed + ' ';
                    if (text.length > 2000) break;
                }
                return text.substring(0, 2000);
            } catch(e) {
                return '';
            }
        }''')
    except Exception as e:
        logger.debug(f"⚠️ Kaushikh: Could not get text content: {e}")
    
    # Kaushikh: Count interactive elements
    element_count = 0
    try:
        element_count = await page.evaluate('''() => {
            try {
                return document.querySelectorAll('a, button, input, select, [onclick], [role="button"]').length;
            } catch(e) {
                return 0;
            }
        }''')
    except Exception as e:
        logger.debug(f"⚠️ Kaushikh: Could not count elements: {e}")
    
    # Kaushikh: Create hash - URL is the most important part
    # Even if text_content fails, URL + title should be enough
    content_to_hash = f"{url}|{title}|{text_content[:1000] if text_content else ''}|{element_count}"
    state_hash = hashlib.md5(content_to_hash.encode()).hexdigest()[:12]
    
    logger.info(f"📊 Kaushikh: State hash generated: {state_hash} (URL: {url[:50] if url else 'none'}...)")
    
    return {
        "hash": state_hash,
        "title": title,
        "url": url,  # Kaushikh: URL is now always captured, even if other parts fail
        "element_count": element_count,
        "text_preview": text_content[:200] if text_content else ""
    }


class SessionInfo(BaseModel):
    """Information about an active session"""
    session_id: str
    current_url: str
    total_steps: int
    created_at: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Browser-Use Agent API",
        "endpoints": {
            "POST /action": "Execute an action on a URL",
            "GET /sessions": "List active sessions",
            "DELETE /sessions/{session_id}": "Close a session"
        }
    }


@app.post("/action", response_model=ActionResponse)
async def execute_action(request: ActionRequest):
    """
    Execute an action on a URL.
    
    If session_id is provided, continues with existing browser session.
    Otherwise, creates a new session and navigates to the URL first.
    """
    try:
        # Validate URL
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable is not set"
            )
        
        # Get or create session
        session_id = None
        browser = None
        is_new_session = False
        
        # Check if session exists
        if request.session_id and request.session_id in active_sessions:
            # Continue existing session - reuse the SAME browser
            session_data = active_sessions[request.session_id]
            browser = session_data["browser"]
            session_id = request.session_id
            logger.info(f"✅ Continuing existing session {session_id[:8]}...")
            
            # Check if browser is still alive and connected
            try:
                current_url = await browser.get_current_page_url()
                logger.info(f"✅ Browser is alive and connected. Current URL: {current_url}")
                # Browser is good, no need to navigate - work on current page
            except Exception as e:
                logger.warning(f"⚠️ Browser session lost ({e}), will recreate...")
                # Browser died, create new one - simple approach like guide.py
                try:
                    browser = Browser(
                        headless=HEADLESS_MODE,
                        window_size={'width': 1280, 'height': 720},
                        keep_alive=True,
                        args=CHROME_ARGS,
                    )
                    # Use same timeout and error handling as new session
                    await asyncio.wait_for(browser.start(), timeout=60.0)
                    session_data["browser"] = browser
                    is_new_session = True  # Need to navigate since browser was recreated
                except asyncio.TimeoutError:
                    logger.error(f"❌ Browser recreation timed out")
                    raise HTTPException(
                        status_code=500,
                        detail="Browser session lost and failed to recreate (timeout). Check if Playwright Chromium is installed: playwright install chromium"
                    )
                except Exception as recreate_error:
                    logger.error(f"❌ Failed to recreate browser: {recreate_error}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Browser session lost and failed to recreate: {str(recreate_error)}"
                    )
        else:
            # Create new session
            if request.session_id:
                # User provided session_id but it doesn't exist - create new one with that ID
                session_id = request.session_id
                logger.info(f"⚠️ Session ID {session_id[:8]} not found, creating new session with this ID...")
            else:
                # Generate new session ID
                session_id = str(uuid.uuid4())
                logger.info(f"🆕 Creating new session {session_id[:8]}...")
            
            is_new_session = True
            
            # Initialize browser - simple approach like guide.py
            browser = Browser(
                headless=HEADLESS_MODE,  # Auto-detects from environment
                window_size={'width': 1280, 'height': 720},
                keep_alive=True,  # Keep browser alive between requests
                args=CHROME_ARGS,  # Add Docker/Cloud Run specific flags
            )
            
            # Start browser session ONCE - this creates the browser window
            logger.info(f"🚀 Starting browser for session {session_id[:8]}...")
            logger.info(f"🔍 Browser instance ID: {id(browser)}")
            logger.info(f"🔍 Headless mode: {HEADLESS_MODE}")
            logger.info(f"🔍 Chrome args: {CHROME_ARGS}")
            
            # Add diagnostics before starting
            try:
                # Check if Playwright browsers are installed
                import subprocess
                playwright_check = subprocess.run(
                    ["python", "-c", "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); print(p.chromium.executable_path); p.stop()"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if playwright_check.returncode == 0:
                    playwright_chrome = playwright_check.stdout.strip()
                    logger.info(f"✅ Playwright Chromium found at: {playwright_chrome}")
                    
                    # Test if Chrome can actually run with our args
                    logger.info("🧪 Testing Chrome executable with our args...")
                    test_args = CHROME_ARGS + ['--version']
                    test_result = subprocess.run(
                        [playwright_chrome] + test_args,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if test_result.returncode == 0:
                        logger.info(f"✅ Chrome executable test passed: {test_result.stdout.strip()}")
                    else:
                        logger.warning(f"⚠️ Chrome executable test failed: {test_result.stderr}")
                        logger.warning(f"   Test command: {playwright_chrome} {' '.join(test_args)}")
                else:
                    logger.warning(f"⚠️ Playwright Chromium check failed: {playwright_check.stderr}")
                    logger.warning("💡 Run: playwright install chromium")
            except Exception as diag_error:
                logger.warning(f"⚠️ Could not check Playwright: {diag_error}")
            
            # Check system resources
            try:
                import psutil
                memory = psutil.virtual_memory()
                logger.info(f"💾 System memory: {memory.percent}% used, {memory.available / (1024**3):.2f} GB available")
                if memory.available < 500 * 1024 * 1024:  # Less than 500MB
                    logger.warning("⚠️ Low memory available - Chrome may fail to start")
            except ImportError:
                logger.debug("psutil not available for memory check")
            except Exception as mem_error:
                logger.debug(f"Memory check failed: {mem_error}")
            
            # Try to start browser with timeout and better error handling
            try:
                await asyncio.wait_for(browser.start(), timeout=60.0)
                logger.info(f"✅ Browser started for session {session_id[:8]}")
            except asyncio.TimeoutError:
                logger.error(f"❌ Browser startup timed out after 60s for session {session_id[:8]}")
                # Check if Chrome process is running
                try:
                    import psutil
                    chrome_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                                      if 'chrome' in p.info['name'].lower() or 'chromium' in p.info['name'].lower()]
                    if chrome_processes:
                        logger.error(f"⚠️ Found {len(chrome_processes)} Chrome/Chromium processes running:")
                        for proc in chrome_processes[:5]:  # Show first 5
                            logger.error(f"   PID {proc.info['pid']}: {proc.info['name']}")
                    else:
                        logger.error("⚠️ No Chrome/Chromium processes found - browser didn't start")
                except:
                    pass
                raise HTTPException(
                    status_code=500,
                    detail="Browser startup timed out. Chrome/Chromium may not be installed or CDP connection failed. Check logs and run: playwright install chromium"
                )
            except Exception as start_error:
                logger.error(f"❌ Browser startup failed: {start_error}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Browser failed to start: {str(start_error)}. Ensure Playwright Chromium is installed: playwright install chromium"
                )
            
            # Store session immediately to prevent duplicate creation
            active_sessions[session_id] = {
                "browser": browser,
                "created_at": asyncio.get_event_loop().time(),
                "total_steps": 0,
            }
            logger.info(f"💾 Session {session_id[:8]} stored in memory")
            
            # AUTOMATICALLY create Daily room and start streaming for new sessions
            # This matches webbot behavior - room is created automatically when bot starts
            try:
                from daily_service import get_daily_service
                from daily_browser_bot import start_daily_bot
                
                daily_service = get_daily_service()
                if daily_service:
                    logger.info(f"🎥 Auto-creating Daily room for new session {session_id[:8]}...")
                    
                    # Create room
                    room_data = await daily_service.create_room(
                        name=f"browser-session-{session_id[:8]}"
                    )
                    
                    room_url = room_data.get("url")
                    room_name = room_data.get("name")
                    
                    if room_url:
                        # Store room info in session
                        active_sessions[session_id]["daily_room"] = {
                            "url": room_url,
                            "name": room_name
                        }
                        
                        # Store globally for /room-info endpoint
                        global _current_room_info
                        _current_room_info = {
                            "roomName": room_name,
                            "url": room_url
                        }
                        
                        logger.info(f"✅ Created Daily room: {room_name} ({room_url})")
                        
                        # Start bot to stream browser video
                        try:
                            logger.info(f"🤖 Starting Daily browser streaming bot...")
                            bot_id = start_daily_bot(
                                session_id=session_id,
                                browser=browser,
                                meeting_url=room_url,
                                framerate=10,  # Lower FPS for stability
                                width=1280,
                                height=720
                            )
                            active_sessions[session_id]["bot_id"] = bot_id
                            logger.info(f"✅ Bot started: {bot_id}")
                            logger.info(f"🎬 Room URL for joining: {room_url}")
                        except Exception as bot_error:
                            logger.error(f"❌ Failed to start bot: {bot_error}", exc_info=True)
                            # Don't fail the whole request if bot fails - room is still created
                            try:
                                await daily_service.delete_room(room_name)
                            except:
                                pass
                        
                        # Notify server bot to join the room (if configured)
                        if SERVER_BOT_URL:
                            logger.info(f"📤 Attempting to notify server bot at {SERVER_BOT_URL}...")
                            try:
                                await notify_server_bot(room_url, session_id, room_token=None, agent_id=request.agent_id)
                            except Exception as notify_error:
                                logger.warning(f"⚠️ Failed to notify server bot: {notify_error}", exc_info=True)
                                # Don't fail the request if notification fails
                        else:
                            logger.debug("🤖 SERVER_BOT_URL not set - skipping server bot notification")
                    else:
                        logger.warning(f"⚠️ Failed to get room URL from Daily API")
                else:
                    logger.debug("Daily service not configured - skipping auto room creation")
            except Exception as daily_error:
                # Don't fail the whole request if Daily setup fails
                logger.warning(f"⚠️ Failed to auto-create Daily room: {daily_error}")
        
        # Only navigate if this is a NEW session
        if is_new_session:
            logger.info(f"🧭 Navigating to {request.url} for new session {session_id[:8]}...")
            # Initialize LLM - use ChatBrowserUse for navigation (faster and optimized for browser tasks)
            llm = ChatBrowserUse()
            logger.info("🚀 Using ChatBrowserUse for navigation (3-5x faster than OpenAI)")
            
            # Navigate to the URL first - use the EXISTING browser instance
            # Use flash_mode and simple task - just navigate, no summaries
            initial_task = f"Your task is to navigate to {request.url}. Use the 'navigate' action to go to this URL. Once you have executed the navigate action, IMMEDIATELY call done. Do NOT wait, do NOT verify, do NOT check if it worked. Just navigate and call done right away."
            
            # Override system message completely to force immediate done after navigation
            navigation_system_message = """You are a navigation-only agent. Your job is extremely simple and focused:

**YOUR ONLY TASK:**
1. Use the 'navigate' action to go to the requested URL
2. IMMEDIATELY call done after executing the navigate action
3. That's it - nothing else

**WHAT TO DO:**
- Execute the navigate action with the URL provided
- As soon as the navigate action is executed, call done immediately
- Do NOT wait for the page to load
- Do NOT check if navigation worked
- Do NOT verify anything
- Do NOT extract content
- Do NOT provide summaries
- Do NOT evaluate page contents
- Do NOT use browser back button
- Do NOT navigate back to previous pages
- Do NOT do anything else

**CRITICAL RULES:**
- Navigation is complete the moment you execute the navigate action - no waiting needed
- Call done IMMEDIATELY after the navigate action, in the same step
- Once you're on the new page, STAY there - do NOT go back
- Do NOT verify or check anything - just navigate and call done

**REMEMBER**: Navigate → Done. That's all. No verification, no waiting, no checking."""
            
            nav_agent = Agent(
                task=initial_task,
                llm=llm,
                browser=browser,  # Use the SAME browser instance that's already started
                flash_mode=True,  # Skip evaluation and thinking - just navigate
                override_system_message=navigation_system_message,  # Completely override default behavior
                max_actions_per_step=1,  # Only allow one action per step
                use_judge=False,  # Disable judge for faster navigation
            )
            
            # Mark navigation as in progress
            action_in_progress[session_id] = True
            
            try:
                # Run initial navigation - browser is already started, so this won't create a new one
                # Use low max_steps to force quick completion
                nav_history = await nav_agent.run(max_steps=3)
                active_sessions[session_id]["total_steps"] = nav_history.number_of_steps() if nav_history else 0
                logger.info(f"✅ Session {session_id[:8]} navigated to {request.url}")
            finally:
                # Always mark navigation as complete
                action_in_progress[session_id] = False
            
            # CRITICAL: Invalidate screenshot cache after navigation
            if session_id in screenshot_cache:
                del screenshot_cache[session_id]
                logger.info(f"🗑️ Invalidated screenshot cache after navigation for session {session_id[:8]}")
            
            # Track navigation completion time
            action_completion_times[session_id] = time.time()
            logger.info(f"⏱️ Marked navigation completion time for session {session_id[:8]}")
            
            # Wait for page stability after navigation
            try:
                page = await browser.get_current_page()
                if page:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                    # Additional small delay to ensure DOM is fully rendered
                    await asyncio.sleep(0.3)
                    logger.info(f"⏳ Page stability confirmed after navigation for session {session_id[:8]}")
            except Exception as e:
                logger.debug(f"Page stability check after navigation: {e}")
                # Still add a small delay even if wait_for_load_state fails
                await asyncio.sleep(0.3)
        else:
            logger.info(f"⏭️ Skipping navigation - continuing on current page for session {session_id[:8]}")
        
        # Now execute the action using the SAME browser instance
        # Get current URL to provide context
        try:
            current_url = await browser.get_current_page_url()
            logger.info(f"📍 Current page URL: {current_url}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get current URL: {e}")
            current_url = request.url
        
        # Check if action is redundant (e.g., "go to this page" when already there after navigation)
        action_lower = request.action.lower().strip()
        is_redundant_navigation = (
            is_new_session and  # We just navigated
            any(phrase in action_lower for phrase in [
                "go to this page", "navigate to", "go to", "open this page", 
                "simply go", "just go", "visit this page"
            ]) and
            current_url == request.url  # Already on the requested URL
        )
        
        if is_redundant_navigation:
            # Skip creating a new agent - we're already where we need to be
            logger.info(f"⏭️ Skipping redundant action - already on {current_url}")
            steps_taken = 0
            urls_visited = [current_url]
        else:
            # Create action task - navigation only, no summaries
            # Check if user explicitly said "do NOTHING else" or similar - if so, stop immediately after action
            action_lower = request.action.lower()
            must_stop_immediately = any(phrase in action_lower for phrase in [
                "do nothing else", "do not do anything else", "stop after", "only", "just", 
                "nothing else", "then stop", "and stop"
            ])
            
            # Detect spatial context keywords to emphasize in task
            spatial_keywords = ["below", "under", "beneath", "after", "above", "over", "before", 
                              "in the header", "navigation bar", "top menu", "main content", 
                              "product grid", "below the", "under the"]
            has_spatial_context = any(keyword in action_lower for keyword in spatial_keywords)
            
            # Enhance task with spatial context emphasis if detected
            spatial_instruction = ""
            if has_spatial_context:
                spatial_instruction = " CRITICAL: Pay close attention to the SPATIAL POSITION of elements mentioned. If the user says 'below X' or 'under X', you MUST find the element that is VISUALLY POSITIONED BELOW element X on the page, not elements with similar text in other locations (like headers). Use the visual screenshot to determine element positions."
            
            if is_new_session:
                if must_stop_immediately:
                    action_task = f"You are currently on {current_url}. Your task: {request.action}.{spatial_instruction} CRITICAL INSTRUCTIONS: Perform the requested action immediately. As soon as you complete the action, IMMEDIATELY call done. Do NOT perform any other actions. Do NOT click anything else. Do NOT explore. Do NOT navigate back. STAY on the page after clicking. Execute the action and call done right away - no delays."
                else:
                    action_task = f"You are currently on {current_url}. Your task: {request.action}.{spatial_instruction} IMPORTANT INSTRUCTIONS: If this involves a dropdown menu, follow these EXACT steps: 1) Click to open the dropdown, 2) Use the 'wait' action to wait 3 seconds, 3) Then click the item inside the dropdown. If clicking a link/button that should navigate to a new page: wait for the URL to change and the page to load, then call done. CRITICAL: Once the URL changes to a new page, that means navigation succeeded - STAY on that new page. Do NOT navigate back. Do NOT use back button. Do NOT call done until you see the URL change (if navigation is expected). Do NOT verify content - just perform the action and call done when appropriate."
            else:
                if must_stop_immediately:
                    action_task = f"Your task: {request.action}.{spatial_instruction} CRITICAL INSTRUCTIONS: Perform the requested action immediately. As soon as you complete the action, IMMEDIATELY call done. Do NOT perform any other actions. Do NOT click anything else. Do NOT explore. Do NOT navigate back. STAY on the page after clicking. Execute the action and call done right away - no delays."
                else:
                    action_task = f"Your task: {request.action}.{spatial_instruction} IMPORTANT INSTRUCTIONS: If this involves a dropdown menu, follow these EXACT steps: 1) Click to open the dropdown, 2) Use the 'wait' action to wait 3 seconds, 3) Then click the item inside the dropdown. If clicking a link/button that should navigate to a new page: wait for the URL to change and the page to load, then call done. CRITICAL: Once the URL changes to a new page, that means navigation succeeded - STAY on that new page. Do NOT navigate back. Do NOT use back button. Do NOT call done until you see the URL change (if navigation is expected). Do NOT verify content - just perform the action and call done when appropriate."
            
            # Create a new agent instance for this action (reuses the SAME browser)
            # CRITICAL: Pass the same browser instance - Agent will reuse it, not create a new one
            # Use ChatBrowserUse for action execution (faster and optimized for browser tasks)
            llm = ChatBrowserUse()
            logger.info("🚀 Using ChatBrowserUse for action execution (3-5x faster than OpenAI)")
            
            # Verify we're using the stored browser instance
            stored_browser = active_sessions[session_id]["browser"]
            if browser is not stored_browser:
                logger.error(f"❌ Browser instance mismatch! Using stored browser instead.")
                browser = stored_browser
            
            logger.info(f"🤖 Creating Agent for action (browser instance ID: {id(browser)})...")
            
            # Override system message - handle dropdowns and explicit stop instructions
            if must_stop_immediately:
                navigation_system_message = """You are a navigation-only agent with STRICT stop instructions. Your job is extremely focused:

**YOUR TASK:**
1. Perform ONLY the requested action (click, navigate, scroll, etc.) - nothing else
2. IMMEDIATELY call done after performing the action - no delays
3. That's it - stop immediately

**WHAT NOT TO DO:**
- Do NOT perform any other actions
- Do NOT click anything else
- Do NOT explore the page
- Do NOT verify content
- Do NOT extract information
- Do NOT use browser back button
- Do NOT navigate back to previous pages
- Do NOT go back to the previous page - STAY on the new page after navigation
- Do NOT perform additional clicks, scrolling, or exploration

**CRITICAL NAVIGATION RULES:**
- After clicking a button/link that navigates to a new page, you MUST STAY on that new page
- Once the URL changes to a new page, that is your signal that navigation succeeded
- Do NOT navigate back, do NOT use back button, do NOT return to previous pages
- If navigation occurs (URL changes), wait for URL to change, then IMMEDIATELY call done and STOP
- If NO navigation occurs, IMMEDIATELY call done and STOP
- NEVER go back to a previous page - always stay on the current page after clicking

**CRITICAL RULES FOR ELEMENT SELECTION:**
- Pay careful attention to SPATIAL CONTEXT when selecting elements:
  * If the user mentions "below", "under", "beneath", or "after" a specific element, find elements that are VISUALLY POSITIONED BELOW that element on the page
  * If the user mentions "above", "over", or "before" a specific element, find elements that are VISUALLY POSITIONED ABOVE that element
  * If the user mentions "in the header", "navigation bar", or "top menu", prioritize elements in the top navigation area
  * If the user mentions "in the main content", "product grid", "below [heading]", prioritize elements in the main content area, NOT the header
  * When multiple elements have the same text (e.g., "iPhone" in header and in product grid), use the visual position and context clues to select the correct one
- Use the visual screenshot and element positions to determine which element matches the spatial description
- If the task says "do NOTHING else" or "only" or "just", you MUST stop immediately after the action

**REMEMBER**: Action → Done. Immediately. No delays, no extra steps."""
            else:
                navigation_system_message = """You are a navigation-only agent. Your job is clear and focused:

**YOUR TASK:**
1. Perform the requested navigation action (click, navigate, scroll, etc.)
2. Call done when the action is complete

**DROPDOWN MENUS - EXACT STEPS:**
If the action involves a dropdown menu, follow these EXACT steps in order:
   a) Click to open the dropdown
   b) Use the 'wait' action to wait 3 seconds (this ensures the dropdown is fully visible)
   c) Then click the item inside the dropdown

**NAVIGATION HANDLING:**
- If the action triggers navigation (clicking a link/button that navigates to a new page):
  * Wait for the URL to change - this is your signal that navigation succeeded
  * Wait for the page to load
  * Once the URL has changed and page is loaded, IMMEDIATELY call done
  * STAY on the new page - do NOT navigate back
  
- If the action does NOT trigger navigation (like scrolling, clicking buttons that don't navigate):
  * Call done immediately after the action
  * No need to wait for URL changes

**WHAT NOT TO DO:**
- Do NOT verify content or extract information
- Do NOT repeat the action unnecessarily
- Do NOT provide summaries
- Do NOT use browser back button
- Do NOT navigate back to previous pages
- Do NOT go back to the previous page - STAY on the new page after navigation

**CRITICAL NAVIGATION RULES:**
- After clicking a button/link that navigates to a new page, you MUST STAY on that new page
- Once the URL changes to a new page, that is your signal that navigation succeeded
- Do NOT navigate back, do NOT use back button, do NOT return to previous pages
- When the URL changes after clicking, wait for the page to fully load, then IMMEDIATELY call done and STOP
- NEVER go back to a previous page - always stay on the current page after clicking
- If you see the URL change, that means navigation worked - stay there and call done

**CRITICAL RULES FOR ELEMENT SELECTION:**
- Pay careful attention to SPATIAL CONTEXT when selecting elements:
  * If the user mentions "below", "under", "beneath", or "after" a specific element, find elements that are VISUALLY POSITIONED BELOW that element on the page
  * If the user mentions "above", "over", or "before" a specific element, find elements that are VISUALLY POSITIONED ABOVE that element
  * If the user mentions "in the header", "navigation bar", or "top menu", prioritize elements in the top navigation area
  * If the user mentions "in the main content", "product grid", "below [heading]", prioritize elements in the main content area, NOT the header
  * When multiple elements have the same text (e.g., "iPhone" in header and in product grid), use the visual position and context clues to select the correct one
- Use the visual screenshot and element positions to determine which element matches the spatial description
- ALWAYS use 'wait' action for 3 seconds between opening a dropdown and clicking an item inside it
- When clicking links or navigation buttons, wait for the page to load (you'll see the URL change) before calling done
- If navigation doesn't happen after clicking, try clicking again or check if the element is actually clickable

**REMEMBER**: Perform action → Wait for URL change (if navigation) → Call done → Stay on new page."""
            
            action_agent = Agent(
                task=action_task,
                llm=llm,  # ChatBrowserUse for navigation/actions
                browser=browser,  # Reuse the SAME browser instance - this is critical!
                flash_mode=True,  # Skip evaluation and thinking - just navigate
                override_system_message=navigation_system_message,  # Completely override default behavior
                max_actions_per_step=1 if must_stop_immediately else 3,  # Limit to 1 action if user said "do NOTHING else"
                use_judge=False,  # Disable judge for faster responses - verification adds significant latency
            )
            logger.info(f"✅ Agent created with browser session ID: {action_agent.browser_session.id}")
            
            # Removed judge callbacks for performance - they add 5-10 seconds per step
            # Judge evaluation is disabled (use_judge=False) to prioritize speed
            # Agent will call done when it believes the task is complete
            
            # Execute action - browser is already started, Agent.run() will reuse it
            # Use the user's requested max_steps without capping it
            # No callbacks to avoid performance overhead - return immediately when agent calls done
            logger.info(f"▶️ Executing action on session {session_id[:8]}: {request.action} (max {request.max_steps} steps)")
            
            # Kaushikh: Capture pre-action page state for change detection
            pre_action_state = {"hash": "unknown", "title": "", "url": "", "element_count": 0, "text_preview": ""}
            try:
                pre_page = await browser.get_current_page()
                if pre_page:
                    pre_action_state = await get_page_state_hash(pre_page)
                    logger.info(f"📊 Pre-action state hash: {pre_action_state['hash']}")
                    logger.info(f"📊 Kaushikh: Pre-action URL: {pre_action_state.get('url', 'unknown')}")
                    logger.info(f"📊 Kaushikh: Pre-action title: {pre_action_state.get('title', 'unknown')[:50]}...")
            except Exception as e:
                logger.warning(f"⚠️ Could not capture pre-action state: {e}")
            
            # Mark action as in progress - this prevents stale screenshots during execution
            action_in_progress[session_id] = True
            
            try:
                history: AgentHistoryList = await action_agent.run(
                    max_steps=request.max_steps
                    # Removed callbacks for performance - they add significant latency
                )
            finally:
                # Always mark action as complete, even if it fails
                action_in_progress[session_id] = False
            
            logger.info(f"✅ Action completed for session {session_id[:8]}")
            
            # CRITICAL: Invalidate screenshot cache after action completes
            # This ensures fresh screenshots are captured after navigation/clicks
            if session_id in screenshot_cache:
                del screenshot_cache[session_id]
                logger.info(f"🗑️ Invalidated screenshot cache for session {session_id[:8]}")
            
            # Track action completion time - screenshots will be forced fresh for 3 seconds
            action_completion_times[session_id] = time.time()
            logger.info(f"⏱️ Marked action completion time for session {session_id[:8]}")
            
            # Kaushikh: FIXED - Wait longer for page stability, especially for SPAs
            # Wait for page stability after action completes
            # This ensures the page has fully loaded before capturing state
            try:
                page = await browser.get_current_page()
                if page:
                    # Kaushikh: First, get the URL immediately after action
                    immediate_url = page.url
                    logger.info(f"📍 Kaushikh: URL immediately after action: {immediate_url}")
                    
                    # Kaushikh: Wait for navigation/content to settle
                    # Try networkidle first (waits for no network activity for 500ms)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5000)
                        logger.info(f"⏳ Kaushikh: Network idle achieved")
                    except Exception as e:
                        logger.debug(f"⏳ Kaushikh: networkidle timeout, trying domcontentloaded: {e}")
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=3000)
                        except:
                            pass
                    
                    # Kaushikh: Additional wait for SPA content to render
                    await asyncio.sleep(0.5)
                    
                    # Kaushikh: Check if URL changed during the wait
                    final_url = page.url
                    if final_url != immediate_url:
                        logger.info(f"📍 Kaushikh: URL changed during wait! {immediate_url} -> {final_url}")
                    
                    logger.info(f"⏳ Page stability confirmed for session {session_id[:8]}")
            except Exception as e:
                # If wait fails, still continue - page might already be stable
                logger.debug(f"Page stability check: {e}")
                # Kaushikh: Longer delay if stability check failed
                await asyncio.sleep(1.0)
            
            # Kaushikh: Capture post-action page state and compare
            post_action_state = {"hash": "unknown", "title": "", "url": "", "element_count": 0, "text_preview": ""}
            page_changed = True  # Kaushikh: Default to True if we can't determine
            try:
                post_page = await browser.get_current_page()
                if post_page:
                    post_action_state = await get_page_state_hash(post_page)
                    logger.info(f"📊 Post-action state hash: {post_action_state['hash']}")
                    logger.info(f"📊 Kaushikh: Pre-URL: {pre_action_state.get('url', 'unknown')}")
                    logger.info(f"📊 Kaushikh: Post-URL: {post_action_state.get('url', 'unknown')}")
                    
                    # Kaushikh: FIXED - More robust change detection
                    # Check URL change first (most reliable for navigation)
                    pre_url = pre_action_state.get("url", "")
                    post_url = post_action_state.get("url", "")
                    url_changed = pre_url != post_url and pre_url and post_url
                    
                    # Check hash change
                    pre_hash = pre_action_state.get("hash", "unknown")
                    post_hash = post_action_state.get("hash", "unknown")
                    hash_changed = pre_hash != post_hash and pre_hash != "unknown" and post_hash != "unknown"
                    
                    # Check title change (backup indicator)
                    pre_title = pre_action_state.get("title", "")
                    post_title = post_action_state.get("title", "")
                    title_changed = pre_title != post_title and pre_title and post_title
                    
                    # Kaushikh: Page changed if ANY of these are true
                    page_changed = url_changed or hash_changed or title_changed
                    
                    logger.info(f"🔄 Kaushikh: Change detection - URL: {url_changed}, Hash: {hash_changed}, Title: {title_changed}")
                    logger.info(f"🔄 Page changed: {page_changed} (pre-hash: {pre_hash}, post-hash: {post_hash})")
                    
                    # Kaushikh: If both hashes are "unknown", assume page changed (fail-safe)
                    if pre_hash == "unknown" and post_hash == "unknown":
                        logger.warning(f"⚠️ Kaushikh: Both hashes unknown, defaulting to page_changed=True")
                        page_changed = True
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not capture post-action state: {e}")
                # Kaushikh: If we can't capture state, assume page changed (fail-safe)
                page_changed = True
            
            # Get URLs visited from history
            urls_visited = [url for url in history.urls() if url is not None]
            steps_taken = history.number_of_steps()
        
        # Update session data (session_id is already set above)
        if session_id in active_sessions:
            active_sessions[session_id]["total_steps"] = active_sessions[session_id].get("total_steps", 0) + steps_taken
        
        # Start streaming AFTER action completes (200 OK) - page is now ready!
        if session_id in active_sessions and "bot_id" in active_sessions[session_id]:
            try:
                from daily_browser_bot import start_streaming_for_bot
                bot_id = active_sessions[session_id]["bot_id"]
                # Pass the main event loop so screenshots run in the correct loop
                main_loop = asyncio.get_event_loop()
                start_streaming_for_bot(bot_id, main_loop)
                logger.info(f"🎬 Started streaming after action completion for session {session_id[:8]}")
            except Exception as stream_error:
                logger.error(f"Failed to start streaming: {stream_error}", exc_info=True)
        
        # Include Daily room URL in response if available
        daily_room_url = None
        daily_room_name = None
        if session_id in active_sessions and "daily_room" in active_sessions[session_id]:
            daily_room = active_sessions[session_id]["daily_room"]
            daily_room_url = daily_room["url"]
            daily_room_name = daily_room["name"]
        
        # Kaushikh: Initialize defaults for edge cases where variables might not exist
        if 'post_action_state' not in dir():
            post_action_state = {"hash": None, "title": None, "url": None, "element_count": None, "text_preview": None}
        if 'page_changed' not in dir():
            page_changed = None
        
        return ActionResponse(
            session_id=session_id,
            urls_visited=urls_visited,
            daily_room_url=daily_room_url,
            daily_room_name=daily_room_name,
            # Kaushikh: Added page state tracking fields
            page_state_hash=post_action_state.get("hash"),
            page_title=post_action_state.get("title"),
            current_url=post_action_state.get("url"),
            dom_element_count=post_action_state.get("element_count"),
            page_changed=page_changed,
            visible_text_preview=post_action_state.get("text_preview"),
        )
        
    except Exception as e:
        logger.error(f"Error executing action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, data in active_sessions.items():
        try:
            current_url = await data["browser"].get_current_page_url()
        except:
            current_url = "Unknown"
        
        sessions.append(SessionInfo(
            session_id=session_id,
            current_url=current_url,
            total_steps=data.get("total_steps", 0),
            created_at=str(data.get("created_at", "Unknown"))
        ))
    
    return sessions


# ============================================================================
# Daily.co Streaming Endpoints
# ============================================================================

# Global room info storage (for /room-info endpoint)
_current_room_info: Optional[dict] = None

@app.post("/streaming/create-room", response_model=CreateRoomResponse)
async def create_daily_room(request: CreateRoomRequest):
    """
    Create a Daily.co room for streaming a browser session.
    Also starts a bot to stream the browser video.
    
    Requires DAILY_API_KEY environment variable.
    """
    global _current_room_info  # Declare global at the top of the function
    
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from daily_service import get_daily_service
        from daily_browser_bot import start_daily_bot
        
        daily_service = get_daily_service()
        if not daily_service:
            raise HTTPException(
                status_code=500,
                detail="Daily service not configured. Set DAILY_API_KEY environment variable."
            )
        
        # Create room
        room_data = await daily_service.create_room(
            name=request.room_name or f"browser-session-{request.session_id[:8]}"
        )
        
        room_url = room_data.get("url")
        room_name = room_data.get("name")
        
        if not room_url:
            raise HTTPException(status_code=500, detail="Failed to get room URL from Daily API")
        
        # Store room info in session
        active_sessions[request.session_id]["daily_room"] = {
            "url": room_url,
            "name": room_name
        }
        
        # Store globally for /room-info endpoint
        _current_room_info = {
            "roomName": room_name,
            "url": room_url
        }
        
        logger.info(f"✅ Created Daily room for session {request.session_id[:8]}: {room_name}")
        
        # Start bot to stream browser video - ONLY after room is successfully created
        try:
            logger.info(f"🤖 Starting Daily browser streaming bot...")
            
            # Get browser instance from session
            browser = active_sessions[request.session_id]["browser"]
            
            # Start Daily bot (runs in background thread)
            bot_id = start_daily_bot(
                session_id=request.session_id,
                browser=browser,
                meeting_url=room_url,
                framerate=10,  # Lower FPS for stability
                width=1280,
                height=720
            )
            
            active_sessions[request.session_id]["bot_id"] = bot_id
            
            logger.info(f"✅ Bot started: {bot_id}")
            
            # Notify server bot to join the room (if configured)
            if SERVER_BOT_URL:
                try:
                    await notify_server_bot(room_url, request.session_id, room_token=None, agent_id=None)
                except Exception as notify_error:
                    logger.warning(f"⚠️ Failed to notify server bot: {notify_error}")
                    # Don't fail the request if notification fails
        except Exception as bot_error:
            # If bot fails to start, clean up the room we just created
            logger.error(f"❌ Failed to start bot: {bot_error}", exc_info=True)
            try:
                await daily_service.delete_room(room_name)
                logger.info(f"🧹 Cleaned up room {room_name} due to bot startup failure")
            except:
                pass
            
            # Clear room info
            _current_room_info = None
            if "daily_room" in active_sessions[request.session_id]:
                del active_sessions[request.session_id]["daily_room"]
            
            raise HTTPException(
                status_code=500,
                detail=f"Room created but bot failed to start: {str(bot_error)}"
            )
        
        return CreateRoomResponse(
            room_name=room_name,
            url=room_url,
            session_id=request.session_id
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404, 500 with details)
        raise
    except Exception as e:
        logger.error(f"Error creating Daily room: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streaming/room-info")
async def get_room_info():
    """
    Get current Daily.co room information.
    Used by client to discover available rooms.
    Returns null if no room is available yet (instead of 404 to avoid log spam).
    """
    global _current_room_info
    if not _current_room_info:
        # Return null instead of 404 to avoid log spam when client polls before room creation
        return None
    return _current_room_info


async def start_room_agent(room_name: str, room_url: str, livekit_service):
    """
    Start the room agent that will join the room and listen to all participants.
    
    This can be called locally (subprocess) or remotely (HTTP request to agent service).
    """
    # Hardcoded agent service URL - deployed on Cloudflare tunnel
    agent_service_url = "https://parade-virtually-difficulty-state.trycloudflare.com"
    agent_deployment_type = os.getenv("ROOM_AGENT_DEPLOYMENT", "remote").lower()  # Default to remote
    
    # Generate agent token
    agent_token = await livekit_service.generate_token(
        room_name=room_name,
        participant_name="room-agent",
        is_agent=True
    )
    
    if agent_deployment_type == "remote" and agent_service_url:
        # Call remote agent service via HTTP/HTTPS
        try:
            import aiohttp
            # Ensure URL doesn't have trailing slash
            agent_service_url = agent_service_url.rstrip('/')
            join_room_url = f"{agent_service_url}/join-room"
            
            logger.info(f"🌐 Calling remote agent service: {join_room_url}")
            logger.info(f"📋 Room: {room_name}, URL: {room_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    join_room_url,
                    json={
                        "room_name": room_name,
                        "room_url": room_url,
                        "token": agent_token
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                    ssl=False  # Allow self-signed certificates for cloudflare tunnels
                ) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        logger.info(f"✅ Remote agent service started for room {room_name}")
                        logger.info(f"📝 Response: {response_text}")
                        return
                    else:
                        logger.error(f"❌ Remote agent service returned {response.status}: {response_text}")
                        raise Exception(f"Remote agent service returned {response.status}: {response_text}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error calling remote agent service: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"❌ Failed to start remote agent: {e}", exc_info=True)
            raise
    else:
        # Start agent locally as subprocess
        try:
            import sys
            agent_script = os.path.join(os.path.dirname(__file__), "room_agent", "start_agent.py")
            
            if not os.path.exists(agent_script):
                logger.warning(f"Agent script not found at {agent_script}, skipping agent start")
                return
            
            # Set environment variables for the agent process
            env = os.environ.copy()
            env["ROOM_NAME"] = room_name
            env["LIVEKIT_URL"] = room_url
            env["LIVEKIT_TOKEN"] = agent_token
            
            # Start agent in background
            process = subprocess.Popen(
                [sys.executable, agent_script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(__file__)
            )
            
            logger.info(f"✅ Started local room agent (PID: {process.pid}) for room {room_name}")
            
            # Store process in session for cleanup later
            # Note: This is a simple implementation - in production, use proper process management
            return process
            
        except Exception as e:
            logger.error(f"Failed to start local agent: {e}", exc_info=True)
            raise


@app.post("/streaming/get-token", response_model=GetTokenResponse)
async def get_token(request: GetTokenRequest):
    """
    Generate a Daily.co meeting token for a participant.
    
    Requires DAILY_API_KEY environment variable.
    """
    try:
        from daily_service import get_daily_service
        
        daily_service = get_daily_service()
        if not daily_service:
            raise HTTPException(
                status_code=500,
                detail="Daily service not configured. Set DAILY_API_KEY environment variable."
            )
        
        # Generate token
        token = await daily_service.create_meeting_token(
            room_name=request.roomName,
            properties={
                "room_name": request.roomName,
                "is_owner": False,
                "user_name": request.participantName,
            }
        )
        
        # Get room URL
        room_data = await daily_service.get_room(request.roomName)
        room_url = room_data.get("url", "")
        
        if not room_url:
            raise HTTPException(status_code=500, detail="Failed to get room URL")
        
        return GetTokenResponse(
            token=token,
            url=room_url
        )
        
    except Exception as e:
        logger.error(f"Error generating token: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streaming/get-screenshot/{session_id}", response_model=ScreenshotResponse)
async def get_screenshot(session_id: str, force_refresh: bool = False):
    """
    Get the latest screenshot from a browser session.
    
    Returns base64-encoded PNG image data.
    Uses caching to return last successful screenshot if new capture fails or is slow.
    
    Args:
        session_id: Browser session ID
        force_refresh: If True, bypass cache and force fresh screenshot (default: False)
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # CRITICAL: If action is in progress, wait a bit and force refresh
    # This prevents showing stale screenshots during action execution
    if session_id in action_in_progress and action_in_progress[session_id]:
        # Action is currently executing - wait a bit and force fresh screenshot
        force_refresh = True
        logger.debug(f"⏳ Action in progress, forcing fresh screenshot")
        # Small delay to let action progress
        await asyncio.sleep(0.1)
    
    # Auto-force refresh if action completed recently (within last 3 seconds)
    current_time = time.time()
    if session_id in action_completion_times:
        time_since_action = current_time - action_completion_times[session_id]
        if time_since_action < 3.0:  # Force refresh for 3 seconds after action
            force_refresh = True
            logger.debug(f"🔄 Auto-forcing refresh (action completed {time_since_action:.2f}s ago)")
        elif time_since_action > 5.0:  # Clean up old entries after 5 seconds
            del action_completion_times[session_id]
    
    # If force_refresh is True, skip cache entirely
    if not force_refresh:
        # Return cached screenshot if available and recent (< 1 second old)
        if session_id in screenshot_cache:
            cache_age = time.time() - screenshot_cache[session_id]["timestamp"]
            if cache_age < 1.0:  # Cache valid for 1 second
                return ScreenshotResponse(
                    screenshot=screenshot_cache[session_id]["screenshot"],
                    session_id=session_id
                )
    
    try:
        session_data = active_sessions[session_id]
        browser = session_data["browser"]
        
        # Get current page
        page = await browser.get_current_page()
        if not page:
            # Return cached if available, even if old
            if session_id in screenshot_cache:
                logger.warning(f"Using cached screenshot (no active page)")
                return ScreenshotResponse(
                    screenshot=screenshot_cache[session_id]["screenshot"],
                    session_id=session_id
                )
            raise HTTPException(status_code=500, detail="No active page in browser")
        
        # Capture screenshot with timeout
        try:
            screenshot_b64 = await asyncio.wait_for(page.screenshot(), timeout=2.0)
            
            # Cache the successful screenshot
            screenshot_cache[session_id] = {
                "screenshot": screenshot_b64,
                "timestamp": time.time()
            }
            
            return ScreenshotResponse(
                screenshot=screenshot_b64,
                session_id=session_id
            )
        except asyncio.TimeoutError:
            # Return cached screenshot if available
            if session_id in screenshot_cache:
                logger.warning(f"Screenshot timeout, returning cached version")
                return ScreenshotResponse(
                    screenshot=screenshot_cache[session_id]["screenshot"],
                    session_id=session_id
                )
            raise HTTPException(status_code=504, detail="Screenshot capture timed out")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}", exc_info=True)
        # Return cached if available
        if session_id in screenshot_cache:
            logger.warning(f"Error in screenshot, returning cached version")
            return ScreenshotResponse(
                screenshot=screenshot_cache[session_id]["screenshot"],
                session_id=session_id
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/streaming/end-stream/{session_id}")
async def end_stream(session_id: str):
    """
    End a streaming session and cleanup Daily.co room and bot.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_sessions[session_id]
        
        # Stop bot if it exists
        if "bot_id" in session_data:
            try:
                from daily_browser_bot import stop_daily_bot
                bot_id = session_data["bot_id"]
                stop_daily_bot(bot_id)
                logger.info(f"✅ Stopped Daily bot: {bot_id}")
            except Exception as e:
                logger.warning(f"Failed to stop bot: {e}")
        
        # Delete Daily room if it exists
        if "daily_room" in session_data:
            try:
                from daily_service import get_daily_service
                daily_service = get_daily_service()
                if daily_service:
                    room_name = session_data["daily_room"]["name"]
                    await daily_service.delete_room(room_name)
                    logger.info(f"✅ Deleted Daily room: {room_name}")
            except Exception as e:
                logger.warning(f"Failed to delete Daily room: {e}")
        
        logger.info(f"Stream ended for session {session_id[:8]}")
        return {"message": "Stream ended successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error ending stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a browser session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_sessions[session_id]
        browser = session_data["browser"]
        
        # Delete Daily room if exists
        if "daily_room" in session_data:
            try:
                from daily_service import get_daily_service
                daily_service = get_daily_service()
                if daily_service:
                    room_name = session_data["daily_room"]["name"]
                    await daily_service.delete_room(room_name)
            except Exception as e:
                logger.warning(f"Failed to delete Daily room: {e}")
        
        # Stop bot if it exists
        if "bot_id" in session_data:
            try:
                from daily_browser_bot import stop_daily_bot
                bot_id = session_data["bot_id"]
                stop_daily_bot(bot_id)
            except Exception as e:
                logger.warning(f"Failed to stop bot: {e}")
        
        # Close browser
        await browser.close()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Session {session_id[:8]} closed")
        return {"message": f"Session {session_id} closed successfully"}
        
    except Exception as e:
        logger.error(f"Error closing session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all browser sessions on shutdown"""
    logger.info("Shutting down, closing all browser sessions...")
    for session_id, data in list(active_sessions.items()):
        try:
            await data["browser"].close()
        except:
            pass
    active_sessions.clear()


if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok
    
    # Set ngrok authtoken
    ngrok.set_auth_token("2kEGVmoK5L1A7fSTRJ6k4n7YMkl_3jBZXFdHfibFjz6fh9LAN")
    
    # Create tunnel (uses SERVER_PORT from environment or default 8080)
    public_url = ngrok.connect(SERVER_PORT)
    
    print("="*60)
    print("Browser-Use FastAPI Agent")
    print("="*60)
    print(f"\nStarting server on http://localhost:{SERVER_PORT}")
    print(f"API docs available at http://localhost:{SERVER_PORT}/docs")
    print(f"\n🌐 Public URL (ngrok): {public_url}")
    print(f"🌐 Public API docs: {public_url}/docs")
    print("\nMake sure OPENAI_API_KEY and DAILY_API_KEY are set in your environment!")
    print(f"To change port, set PORT environment variable (current: {SERVER_PORT})")
    print("="*60)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except KeyboardInterrupt:
        ngrok.kill()