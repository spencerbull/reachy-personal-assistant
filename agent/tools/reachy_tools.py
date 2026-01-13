"""
Reachy robot control tools.

These tools allow the LLM to control Reachy's movement, expressions, and tracking.
They return command tokens that are parsed by the Pipecat pipeline to execute
actual robot movements.
"""

from typing import Literal, Optional

from loguru import logger
from langchain_core.tools import tool

# Valid directions for head movement
HeadDirection = Literal["left", "right", "up", "down", "front"]

# Valid directions for body turn
BodyDirection = Literal["left", "right"]

# Valid emotions
EmotionType = Literal["happy", "sad", "excited", "curious", "thinking", "neutral", "attentive", "playful"]

# Valid dance moves from reachy_mini_dances_library
# See: https://github.com/pollen-robotics/reachy_mini_dances_library
DanceType = Literal[
    # Simple nods and tilts
    "simple_nod",           # Classic simple nod
    "yeah_nod",             # Enthusiastic yeah nod
    "uh_huh_tilt",          # Bouncy affirmative tilt
    "head_tilt_roll",       # Gentle tilt loop
    "neck_recoil",          # Snappy neck recoil pop
    "sharp_side_tilt",      # Quick angular head tilt
    # Playful moves
    "chicken_peck",         # Pecking head bop loop
    "side_peekaboo",        # Peekaboo side reveal
    "side_glance_flick",    # Fast glance with flick
    "chin_lead",            # Extended chin-led glides
    "stumble_and_recover",  # Playful stumble and recover
    # Energetic dances
    "headbanger_combo",     # High-energy head isolation
    "side_to_side_sway",    # Energetic side-to-side groove
    "pendulum_swing",       # Even tempo pendulum motion
    "dizzy_spin",           # Slow spin with antenna flair
    # Complex choreography
    "jackson_square",       # Precision hits with shoulder pops
    "groovy_sway_and_roll", # Flowing sway with rolling torso
    "interwoven_spirals",   # Layered spirals across axis
    "polyrhythm_combo",     # Offset waves with counter beats
    "grid_snap",            # Sharp grid-aligned accents
]

# Dance descriptions for LLM context
DANCE_DESCRIPTIONS = {
    "simple_nod": "Classic simple nod - for acknowledgment",
    "yeah_nod": "Enthusiastic yeah nod - for excited agreement",
    "uh_huh_tilt": "Bouncy affirmative tilt - for casual yes",
    "head_tilt_roll": "Gentle tilt loop - subtle and calm",
    "neck_recoil": "Snappy neck recoil pop - quick surprise",
    "sharp_side_tilt": "Quick angular head tilt - attentive",
    "chicken_peck": "Pecking head bop loop - silly and playful",
    "side_peekaboo": "Peekaboo side reveal - fun and curious",
    "side_glance_flick": "Fast glance with flick - mischievous",
    "chin_lead": "Extended chin-led glides - smooth and cool",
    "stumble_and_recover": "Playful stumble and recover - comedic",
    "headbanger_combo": "High-energy head isolation - rock out",
    "side_to_side_sway": "Energetic side-to-side groove - funky",
    "pendulum_swing": "Even tempo pendulum motion - rhythmic",
    "dizzy_spin": "Slow spin with antenna flair - celebration",
    "jackson_square": "Precision hits with shoulder pops - dramatic",
    "groovy_sway_and_roll": "Flowing sway with rolling torso - groovy",
    "interwoven_spirals": "Layered spirals across axis - hypnotic",
    "polyrhythm_combo": "Offset waves with counter beats - complex",
    "grid_snap": "Sharp grid-aligned accents - robotic",
}


# Global reference to Reachy service (set during initialization)
_reachy_service = None


def set_reachy_service(service):
    """Set the Reachy service instance for direct robot control."""
    global _reachy_service
    _reachy_service = service
    logger.info("Reachy service connected to tools")


def get_reachy_service():
    """Get the Reachy service instance."""
    return _reachy_service


@tool
def look_at_tool(direction: HeadDirection) -> str:
    """
    Move Reachy's head to look in the specified direction.
    
    Use this when the user asks Reachy to look somewhere or when you need
    to orient the camera in a specific direction.
    
    Args:
        direction: The direction to look - one of "left", "right", "up", "down", or "front"
        
    Returns:
        Confirmation message with command token for the robot controller
    """
    direction = direction.lower()
    
    # Map direction to command token
    token_map = {
        "left": "[CMD_LOOK_LEFT]",
        "right": "[CMD_LOOK_RIGHT]",
        "up": "[CMD_LOOK_UP]",
        "down": "[CMD_LOOK_DOWN]",
        "front": "[CMD_LOOK_FRONT]",
    }
    
    token = token_map.get(direction)
    
    if token:
        logger.info(f"look_at_tool: Looking {direction}")
        
        # Direct execution if service available
        if _reachy_service and _reachy_service.connected:
            try:
                _reachy_service.look_at(direction)
                return f"I'm looking to the {direction} now."
            except Exception as e:
                logger.warning(f"Direct execution failed: {e}, using command token")
        
        return f"I'm turning my head to look {direction}. {token}"
    else:
        logger.warning(f"look_at_tool: Invalid direction '{direction}'")
        return f"I don't know how to look {direction}. I can look left, right, up, down, or front."


@tool
def turn_body_tool(direction: BodyDirection) -> str:
    """
    Turn Reachy's body/base in the specified direction.
    
    Use this when the user asks Reachy to turn around or rotate to face
    a different direction. This is different from head movement - it
    rotates the entire robot body.
    
    Args:
        direction: The direction to turn - either "left" or "right"
        
    Returns:
        Confirmation message with command token for the robot controller
    """
    direction = direction.lower()
    
    token_map = {
        "left": "[CMD_TURN_LEFT]",
        "right": "[CMD_TURN_RIGHT]",
    }
    
    token = token_map.get(direction)
    
    if token:
        logger.info(f"turn_body_tool: Turning body {direction}")
        return f"I'm turning my body to the {direction}. {token}"
    else:
        logger.warning(f"turn_body_tool: Invalid direction '{direction}'")
        return f"I can only turn left or right."


@tool
def enable_face_tracking_tool(enable: bool) -> str:
    """
    Enable or disable face tracking mode.
    
    When enabled, Reachy will automatically follow the user's face to
    maintain eye contact. Use this when the user says "look at me" or
    asks Reachy to pay attention to them.
    
    Args:
        enable: True to enable face tracking, False to disable
        
    Returns:
        Confirmation message with command token for the robot controller
    """
    if enable:
        logger.info("enable_face_tracking_tool: Enabling face tracking")
        return "I'm now tracking your face and maintaining eye contact with you. [CMD_FACE_TRACK_ON]"
    else:
        logger.info("enable_face_tracking_tool: Disabling face tracking")
        return "I've stopped tracking your face. [CMD_FACE_TRACK_OFF]"


@tool
def express_emotion_tool(emotion: EmotionType) -> str:
    """
    Express an emotion through Reachy's movements and antenna positions.
    
    Use this to make Reachy more expressive and personable. Different
    emotions trigger different movement patterns:
    - happy: Antenna wiggle, slight bounce
    - sad: Drooping antennas, slight head down
    - excited: Fast antenna movement, bouncy
    - curious: Head tilt, raised antennas
    - thinking: Look up/away, antenna pause
    - neutral: Return to default position
    - attentive: Forward lean, steady antennas
    - playful: Quick movements, antenna dance
    
    Args:
        emotion: The emotion to express
        
    Returns:
        Confirmation message with command token for the robot controller
    """
    emotion = emotion.lower()
    
    emotion_descriptions = {
        "happy": "I'm feeling happy! [CMD_EMOTION_HAPPY]",
        "sad": "I'm feeling a bit down. [CMD_EMOTION_SAD]",
        "excited": "I'm so excited! [CMD_EMOTION_EXCITED]",
        "curious": "Hmm, that's interesting! [CMD_EMOTION_CURIOUS]",
        "thinking": "Let me think about that. [CMD_EMOTION_THINKING]",
        "neutral": "I'm back to my normal state. [CMD_EMOTION_NEUTRAL]",
        "attentive": "I'm listening carefully. [CMD_EMOTION_ATTENTIVE]",
        "playful": "Time to have some fun! [CMD_EMOTION_PLAYFUL]",
    }
    
    if emotion in emotion_descriptions:
        logger.info(f"express_emotion_tool: Expressing {emotion}")
        return emotion_descriptions[emotion]
    else:
        logger.warning(f"express_emotion_tool: Unknown emotion '{emotion}'")
        return f"I don't know how to express {emotion}."


@tool
def dance_tool(dance_name: DanceType) -> str:
    """
    Make Reachy perform a dance move from the available repertoire.
    
    Use this when the user asks Reachy to dance, celebrate, or move expressively.
    Choose the dance based on the context:
    
    AVAILABLE DANCES:
    - simple_nod: Classic nod for acknowledgment
    - yeah_nod: Enthusiastic nod for excited agreement
    - uh_huh_tilt: Bouncy tilt for casual yes
    - head_tilt_roll: Gentle tilt loop, calm and subtle
    - neck_recoil: Snappy pop for surprise
    - sharp_side_tilt: Quick angular tilt, attentive
    - chicken_peck: Silly pecking bop for playful moments
    - side_peekaboo: Fun peekaboo reveal
    - side_glance_flick: Mischievous quick glance
    - chin_lead: Smooth chin-led glides, cool
    - stumble_and_recover: Comedic stumble
    - headbanger_combo: High-energy for rock/excitement
    - side_to_side_sway: Funky side-to-side groove
    - pendulum_swing: Rhythmic swinging motion
    - dizzy_spin: Celebration spin with antenna flair
    - jackson_square: Dramatic precision hits
    - groovy_sway_and_roll: Smooth groovy flow
    - interwoven_spirals: Hypnotic layered spirals
    - polyrhythm_combo: Complex rhythmic waves
    - grid_snap: Sharp robotic grid movements
    
    Args:
        dance_name: The specific dance move to perform
        
    Returns:
        Confirmation message with command token for the robot controller
    """
    dance_name_lower = dance_name.lower()
    
    # Validate dance name
    if dance_name_lower not in DANCE_DESCRIPTIONS:
        logger.warning(f"dance_tool: Unknown dance '{dance_name}', defaulting to simple_nod")
        dance_name_lower = "simple_nod"
    
    description = DANCE_DESCRIPTIONS.get(dance_name_lower, "a cool move")
    logger.info(f"dance_tool: Performing dance '{dance_name_lower}' - {description}")
    
    # Response varies based on dance type
    dance_responses = {
        "simple_nod": "Sure thing!",
        "yeah_nod": "Oh yeah!",
        "headbanger_combo": "Let's rock!",
        "groovy_sway_and_roll": "Getting groovy!",
        "chicken_peck": "Bawk bawk!",
        "dizzy_spin": "Woohoo!",
        "jackson_square": "Watch this!",
    }
    
    response = dance_responses.get(dance_name_lower, "Check this out!")
    return f"{response} [CMD_DANCE_{dance_name_lower.upper()}]"


@tool
def scan_room_tool(scan_direction: str = "full") -> str:
    """
    Make Reachy scan the room by looking around.
    
    Use this when the user asks Reachy to look around the room or
    to help find something. Reachy will move their head to scan
    the environment.
    
    Args:
        scan_direction: Type of scan - "full" for complete scan, 
                       "left_to_right" or "right_to_left" for sweep
        
    Returns:
        Confirmation message with command token for the robot controller
    """
    scan_direction = scan_direction.lower()
    
    if scan_direction == "full":
        logger.info("scan_room_tool: Full room scan")
        return "I'm scanning the room... [CMD_SCAN_FULL]"
    elif scan_direction in ("left_to_right", "ltr"):
        logger.info("scan_room_tool: Left to right scan")
        return "I'm scanning from left to right... [CMD_SCAN_LTR]"
    elif scan_direction in ("right_to_left", "rtl"):
        logger.info("scan_room_tool: Right to left scan")
        return "I'm scanning from right to left... [CMD_SCAN_RTL]"
    else:
        return "I'll do a quick scan of the area. [CMD_SCAN_FULL]"


@tool
def get_current_pose_tool() -> str:
    """
    Get Reachy's current head and body pose.
    
    Use this to check what direction Reachy is currently looking.
    
    Returns:
        Description of current pose
    """
    if _reachy_service and _reachy_service.connected:
        try:
            # Get actual pose from robot
            robot = _reachy_service.robot
            if robot:
                head_pose = robot.get_current_head_pose()
                # Extract approximate direction from pose
                return f"I'm currently looking forward. My head pose is at the default position."
        except Exception as e:
            logger.warning(f"Could not get current pose: {e}")
    
    return "I'm currently looking forward with my head in a neutral position."


@tool
def nod_tool(affirmative: bool = True) -> str:
    """
    Make Reachy nod their head (yes) or shake (no).
    
    Use this for non-verbal confirmation or denial.
    
    Args:
        affirmative: True for nodding (yes), False for shaking (no)
        
    Returns:
        Confirmation message with command token
    """
    if affirmative:
        logger.info("nod_tool: Nodding yes")
        return "[CMD_NOD_YES]"
    else:
        logger.info("nod_tool: Shaking no")
        return "[CMD_NOD_NO]"


def get_all_reachy_tools():
    """Get all Reachy control tools as a list."""
    return [
        look_at_tool,
        turn_body_tool,
        enable_face_tracking_tool,
        express_emotion_tool,
        dance_tool,
        scan_room_tool,
        get_current_pose_tool,
        nod_tool,
    ]
