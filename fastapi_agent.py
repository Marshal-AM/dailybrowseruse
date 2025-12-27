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
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatOpenAI
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

# Agent configuration - easily toggle judge evaluation
USE_JUDGE = False  # Set to True to enable judge evaluation of agent tasks

# Headless mode configuration (for Cloud Run / production)
# Automatically set to True in Cloud Run, or set HEADLESS_MODE=true in .env
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "false").lower() == "true"
logger.info(f"🖥️ Headless mode: {HEADLESS_MODE}")

# Server port - configurable via environment variable
SERVER_PORT = int(os.getenv("PORT", "8080"))
logger.info(f"🔌 Server port: {SERVER_PORT}")

# API URL for bot to connect (use localhost for local dev, GCP URL for production)
FASTAPI_URL = os.getenv("FASTAPI_URL", f"http://localhost:{SERVER_PORT}")
logger.info(f"🔗 FastAPI URL for bot: {FASTAPI_URL}")

# Verify Chrome/Chromium is available
def check_chrome_available():
    """Check if Chrome/Chromium is installed and accessible."""
    chrome_paths = [
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/snap/bin/chromium",
    ]
    for path in chrome_paths:
        if path and os.path.exists(path):
            logger.info(f"✅ Found Chrome/Chromium at: {path}")
            return True
    logger.warning("⚠️ Chrome/Chromium not found in PATH. Browser startup may fail.")
    logger.warning("💡 Run setup_chrome.sh to install Chrome/Chromium")
    return False

# Check Chrome availability on startup
_chrome_available = check_chrome_available()

# Chrome args for Docker/Cloud Run/Vast AI environments
# These are essential for running Chrome in server environments
CHROME_ARGS = [
    '--no-sandbox',  # Required for Docker/Vast AI
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',  # Overcome limited resource problems
    '--disable-gpu',  # Disable GPU acceleration (required for headless)
    '--disable-software-rasterizer',
    '--disable-extensions',
    '--disable-background-networking',
    '--disable-default-apps',
    '--disable-sync',
    '--metrics-recording-only',
    '--no-first-run',
    '--safebrowsing-disable-auto-update',
    '--disable-blink-features=AutomationControlled',
    '--disable-features=VizDisplayCompositor',  # Helps with headless rendering
    '--disable-web-security',  # Sometimes needed for CDP
    '--remote-debugging-port=0',  # Let browser_use choose the port
    '--remote-allow-origins=*',  # Allow CDP connections
] if HEADLESS_MODE else [
    # Even in non-headless mode, we need some flags for server environments
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
]


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


class ActionResponse(BaseModel):
    """Response model for agent actions"""
    session_id: str
    urls_visited: list[str]


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
                # Browser died, create new one
                try:
                    browser = Browser(
                        headless=HEADLESS_MODE,
                        window_size={'width': 1280, 'height': 720},
                        keep_alive=True,
                        args=CHROME_ARGS,
                    )
                    await asyncio.wait_for(browser.start(), timeout=60.0)
                    session_data["browser"] = browser
                    is_new_session = True  # Need to navigate since browser was recreated
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
            
            # Check Chrome availability before starting
            if not _chrome_available:
                logger.error("❌ Chrome/Chromium not available. Cannot start browser.")
                raise HTTPException(
                    status_code=500,
                    detail="Chrome/Chromium is not installed or not in PATH. Please run setup_chrome.sh to install it."
                )
            
            # Initialize browser
            logger.info(f"🚀 Starting browser for session {session_id[:8]}...")
            logger.info(f"🔍 Headless mode: {HEADLESS_MODE}, Chrome args: {len(CHROME_ARGS)} flags")
            
            browser = Browser(
                headless=HEADLESS_MODE,  # Auto-detects from environment
                window_size={'width': 1280, 'height': 720},
                keep_alive=True,  # Keep browser alive between requests
                args=CHROME_ARGS,  # Add Docker/Cloud Run specific flags
            )
            
            logger.info(f"🔍 Browser instance ID: {id(browser)}")
            
            # Start browser session ONCE - this creates the browser window
            # Add timeout and better error handling
            try:
                await asyncio.wait_for(browser.start(), timeout=60.0)  # 60 second timeout
                logger.info(f"✅ Browser started for session {session_id[:8]}")
            except asyncio.TimeoutError:
                logger.error(f"❌ Browser startup timed out after 60 seconds for session {session_id[:8]}")
                raise HTTPException(
                    status_code=500,
                    detail="Browser failed to start within 60 seconds. Check if Chrome/Chromium is installed and accessible."
                )
            except Exception as e:
                logger.error(f"❌ Browser startup failed for session {session_id[:8]}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Browser failed to start: {str(e)}. Make sure Chrome/Chromium is installed (run setup_chrome.sh)."
                )
            
            # Store session immediately to prevent duplicate creation
            active_sessions[session_id] = {
                "browser": browser,
                "created_at": asyncio.get_event_loop().time(),
                "total_steps": 0,
            }
            logger.info(f"💾 Session {session_id[:8]} stored in memory")
        
        # Only navigate if this is a NEW session
        if is_new_session:
            logger.info(f"🧭 Navigating to {request.url} for new session {session_id[:8]}...")
            # Initialize LLM (OpenAI only)
            llm = ChatOpenAI(model='gpt-4o')
            
            # Navigate to the URL first - use the EXISTING browser instance
            # Use flash_mode and simple task - just navigate, no summaries
            initial_task = f"Navigate to {request.url}. After navigating, IMMEDIATELY call done. Do NOT verify, do NOT check if it worked. Just navigate and call done."
            
            # Override system message completely to force immediate done after navigation
            navigation_system_message = """You are a navigation-only agent. Your job is simple:
1. Navigate to the requested URL
2. IMMEDIATELY call done after navigating
3. Do NOT verify if navigation worked
4. Do NOT extract content or provide summaries
5. Do NOT evaluate page contents

After navigating, you MUST call done in the same step or the very next step. Navigation is complete once you've navigated - no verification needed."""
            
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
                    action_task = f"You are already on {current_url}. {request.action}.{spatial_instruction} CRITICAL: After performing the requested action, IMMEDIATELY call done. Do NOT perform any other actions. Do NOT click anything else. Do NOT explore. Just perform the action and call done."
                else:
                    action_task = f"You are already on {current_url}. {request.action}.{spatial_instruction} IMPORTANT: If this involves a dropdown, follow these steps: 1) Click to open the dropdown, 2) Use the 'wait' action to wait 3 seconds, 3) Then click the item inside the dropdown. After clicking a link/button that should navigate, wait for the URL to change and page to load before calling done. Do NOT call done until navigation completes. Do NOT verify content."
            else:
                if must_stop_immediately:
                    action_task = f"{request.action}.{spatial_instruction} CRITICAL: After performing the requested action, IMMEDIATELY call done. Do NOT perform any other actions. Do NOT click anything else. Do NOT explore. Just perform the action and call done."
                else:
                    action_task = f"{request.action}.{spatial_instruction} IMPORTANT: If this involves a dropdown, follow these steps: 1) Click to open the dropdown, 2) Use the 'wait' action to wait 3 seconds, 3) Then click the item inside the dropdown. After clicking a link/button that should navigate, wait for the URL to change and page to load before calling done. Do NOT call done until navigation completes. Do NOT verify content."
            
            # Create a new agent instance for this action (reuses the SAME browser)
            # CRITICAL: Pass the same browser instance - Agent will reuse it, not create a new one
            llm = ChatOpenAI(model='gpt-4o')
            
            # Verify we're using the stored browser instance
            stored_browser = active_sessions[session_id]["browser"]
            if browser is not stored_browser:
                logger.error(f"❌ Browser instance mismatch! Using stored browser instead.")
                browser = stored_browser
            
            logger.info(f"🤖 Creating Agent for action (browser instance ID: {id(browser)})...")
            
            # Override system message - handle dropdowns and explicit stop instructions
            if must_stop_immediately:
                navigation_system_message = """You are a navigation-only agent with STRICT stop instructions. Your job is:
1. Perform ONLY the requested action (click, navigate, scroll, etc.)
2. IMMEDIATELY call done after performing the action
3. Do NOT perform any other actions
4. Do NOT click anything else
5. Do NOT explore the page
6. Do NOT verify content
7. Do NOT extract information

CRITICAL RULES FOR ELEMENT SELECTION:
- Pay careful attention to SPATIAL CONTEXT when selecting elements:
  * If the user mentions "below", "under", "beneath", or "after" a specific element, find elements that are VISUALLY POSITIONED BELOW that element on the page
  * If the user mentions "above", "over", or "before" a specific element, find elements that are VISUALLY POSITIONED ABOVE that element
  * If the user mentions "in the header", "navigation bar", or "top menu", prioritize elements in the top navigation area
  * If the user mentions "in the main content", "product grid", "below [heading]", prioritize elements in the main content area, NOT the header
  * When multiple elements have the same text (e.g., "iPhone" in header and in product grid), use the visual position and context clues to select the correct one
- Use the visual screenshot and element positions to determine which element matches the spatial description
- If the task says "do NOTHING else" or "only" or "just", you MUST stop immediately after the action
- After clicking a button/link, if navigation occurs, wait for URL to change, then IMMEDIATELY call done
- After clicking a button/link, if NO navigation occurs, IMMEDIATELY call done
- Do NOT perform any additional clicks, scrolling, or exploration"""
            else:
                navigation_system_message = """You are a navigation-only agent. Your job is simple:
1. Perform the requested navigation action (click, navigate, scroll, etc.)
2. For dropdown menus: Follow these EXACT steps:
   a) Click to open the dropdown
   b) Use the 'wait' action to wait 3 seconds (this ensures the dropdown is fully visible)
   c) Then click the item inside the dropdown
3. If the action triggers navigation (clicking a link/button that navigates), wait for the page to load (URL changes) before calling done
4. If the action does NOT trigger navigation (like scrolling), call done immediately after the action
5. Do NOT verify content or extract information
6. Do NOT repeat the action unnecessarily
7. Do NOT provide summaries

CRITICAL RULES FOR ELEMENT SELECTION:
- Pay careful attention to SPATIAL CONTEXT when selecting elements:
  * If the user mentions "below", "under", "beneath", or "after" a specific element, find elements that are VISUALLY POSITIONED BELOW that element on the page
  * If the user mentions "above", "over", or "before" a specific element, find elements that are VISUALLY POSITIONED ABOVE that element
  * If the user mentions "in the header", "navigation bar", or "top menu", prioritize elements in the top navigation area
  * If the user mentions "in the main content", "product grid", "below [heading]", prioritize elements in the main content area, NOT the header
  * When multiple elements have the same text (e.g., "iPhone" in header and in product grid), use the visual position and context clues to select the correct one
- Use the visual screenshot and element positions to determine which element matches the spatial description
- ALWAYS use 'wait' action for 3 seconds between opening a dropdown and clicking an item inside it
- When clicking links or navigation buttons, wait for the page to load (you'll see the URL change) before calling done
- If navigation doesn't happen after clicking, try clicking again or check if the element is actually clickable"""
            
            action_agent = Agent(
                task=action_task,
                llm=llm,
                browser=browser,  # Reuse the SAME browser instance - this is critical!
                flash_mode=True,  # Skip evaluation and thinking - just navigate
                override_system_message=navigation_system_message,  # Completely override default behavior
                max_actions_per_step=1 if must_stop_immediately else 3,  # Limit to 1 action if user said "do NOTHING else"
                use_judge=False,  # Disable judge for faster responses - verification adds significant latency
                judge_llm=llm,  # Use same LLM for judge (not used when use_judge=False)
            )
            logger.info(f"✅ Agent created with browser session ID: {action_agent.browser_session.id}")
            
            # Removed judge callbacks for performance - they add 5-10 seconds per step
            # Judge evaluation is disabled (use_judge=False) to prioritize speed
            # Agent will call done when it believes the task is complete
            
            # Execute action - browser is already started, Agent.run() will reuse it
            # Use the user's requested max_steps without capping it
            # No callbacks to avoid performance overhead - return immediately when agent calls done
            logger.info(f"▶️ Executing action on session {session_id[:8]}: {request.action} (max {request.max_steps} steps)")
            
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
            
            # Wait for page stability after action completes
            # This ensures the page has fully loaded before allowing screenshots
            # Reduced wait time to make UI updates faster
            try:
                page = await browser.get_current_page()
                if page:
                    # Wait for network to be idle (no requests for 500ms) with shorter timeout
                    # Use domcontentloaded instead of networkidle for faster response
                    await page.wait_for_load_state("domcontentloaded", timeout=2000)
                    # Minimal delay - just enough for rendering
                    await asyncio.sleep(0.1)
                    logger.info(f"⏳ Page stability confirmed for session {session_id[:8]}")
            except Exception as e:
                # If wait fails, still continue - page might already be stable
                logger.debug(f"Page stability check: {e}")
                # Minimal delay even if wait_for_load_state fails
                await asyncio.sleep(0.1)
            
            # Get URLs visited from history
            urls_visited = [url for url in history.urls() if url is not None]
            steps_taken = history.number_of_steps()
        
        # Update session data (session_id is already set above)
        if session_id in active_sessions:
            active_sessions[session_id]["total_steps"] = active_sessions[session_id].get("total_steps", 0) + steps_taken
        
        return ActionResponse(
            session_id=session_id,
            urls_visited=urls_visited,
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
                framerate=30,
                width=1280,
                height=720
            )
            
            active_sessions[request.session_id]["bot_id"] = bot_id
            
            logger.info(f"✅ Bot started: {bot_id}")
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
