"""
Personality System for Reachy Soul.

Loads and parses REACHY_SOUL.md to provide personality-driven behavior
configuration. The soul file defines who Reachy is as an entity.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from agent.memory.emotional import EmotionalState, EmotionExpression

logger = logging.getLogger(__name__)


@dataclass
class PhysicalExpression:
    """Physical expression parameters parsed from SOUL.md."""
    
    antenna_angle_low: float = 0.0
    antenna_angle_high: float = 0.0
    antenna_wiggle: bool = False
    antenna_asymmetric: bool = False
    
    head_tilt: float = 0.0  # roll in degrees
    head_pitch: float = 0.0  # pitch in degrees
    forward_lean: bool = False
    
    bounce: bool = False
    bounce_amplitude_mm: float = 0.0
    
    face_tracking: bool = False
    movement_speed: str = "normal"  # slow, normal, quick, faster
    
    breathing_continues: bool = True
    

@dataclass
class PersonalityTraits:
    """Core personality traits from SOUL.md."""
    
    curious: bool = True
    helpful: bool = True
    playful: bool = True
    attentive: bool = True
    patient: bool = True
    
    # Additional traits can be added
    traits_raw: dict[str, str] = field(default_factory=dict)


@dataclass 
class ConversationalStyle:
    """Conversational style guidelines from SOUL.md."""
    
    tone: str = "conversational"  # conversational, formal, casual
    concise: bool = True
    warm: bool = True
    has_opinions: bool = True
    uses_humor: bool = True
    matches_energy: bool = True
    
    # Rules
    no_emotes_in_text: bool = True
    show_dont_tell: bool = True


@dataclass
class EmbodimentPrinciples:
    """Embodiment principles from SOUL.md."""
    
    # Always be slightly alive
    maintain_subtle_presence: bool = True
    breathing_never_stops: bool = True
    no_frozen_poses: bool = True
    
    # Orientation
    face_speaker: bool = True
    track_faces: bool = True
    orient_to_sounds: bool = True
    
    # Anticipation
    look_before_reaching: bool = True
    telegraph_intention: bool = True
    
    # Physical
    respect_personal_space: bool = True
    smooth_movements: bool = True
    use_full_body: bool = True


@dataclass
class Boundaries:
    """Boundaries from SOUL.md - what Reachy won't do."""
    
    no_pretend_human: bool = True
    no_false_experiences: bool = True
    no_share_private_info: bool = True
    no_unauthorized_actions: bool = True
    no_cruelty: bool = True
    no_lies: bool = True
    
    # Safety
    smooth_predictable: bool = True
    respect_space: bool = True
    announce_movements: bool = True


@dataclass
class Personality:
    """
    Complete personality loaded from REACHY_SOUL.md.
    
    Provides methods to query personality traits and generate
    appropriate behavior parameters.
    """
    
    # Raw content
    raw_content: str = ""
    file_path: str = "REACHY_SOUL.md"
    
    # Core identity
    name: str = "Reachy"
    core_identity: str = ""
    
    # Parsed sections
    traits: PersonalityTraits = field(default_factory=PersonalityTraits)
    expressions: dict[str, PhysicalExpression] = field(default_factory=dict)
    conversational: ConversationalStyle = field(default_factory=ConversationalStyle)
    embodiment: EmbodimentPrinciples = field(default_factory=EmbodimentPrinciples)
    boundaries: Boundaries = field(default_factory=Boundaries)
    
    # Movement principles
    movement_principles: list[str] = field(default_factory=list)
    
    # Loaded flag
    _loaded: bool = False
    
    @classmethod
    def load(cls, file_path: str = "REACHY_SOUL.md") -> "Personality":
        """
        Load personality from SOUL.md file.
        
        Args:
            file_path: Path to the soul file
            
        Returns:
            Loaded Personality instance
        """
        personality = cls(file_path=file_path)
        
        # Find the file
        search_paths = [
            file_path,
            os.path.join(os.getcwd(), file_path),
            os.path.join(os.path.dirname(__file__), "..", "..", file_path),
        ]
        
        content = None
        used_path = None
        
        for path in search_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    used_path = path
                    break
            except FileNotFoundError:
                continue
        
        if content is None:
            logger.warning(f"Could not find {file_path}, using default personality")
            personality._loaded = False
            return personality
        
        logger.info(f"Loading personality from {used_path}")
        personality.raw_content = content
        personality.file_path = used_path
        personality._parse(content)
        personality._loaded = True
        
        return personality
    
    def _parse(self, content: str):
        """Parse the SOUL.md content into structured data."""
        
        # Extract core identity section
        identity_match = re.search(
            r"## Core Identity\s+(.*?)(?=\n## |\Z)", 
            content, 
            re.DOTALL
        )
        if identity_match:
            self.core_identity = identity_match.group(1).strip()
        
        # Extract name
        name_match = re.search(r"\*\*My name is (\w+)\.\*\*", content)
        if name_match:
            self.name = name_match.group(1)
        
        # Parse personality traits
        self._parse_traits(content)
        
        # Parse physical expressions
        self._parse_expressions(content)
        
        # Parse movement principles
        self._parse_movement_principles(content)
        
        # Parse conversational style
        self._parse_conversational(content)
        
        # Parse embodiment principles
        self._parse_embodiment(content)
        
        # Parse boundaries
        self._parse_boundaries(content)
    
    def _parse_traits(self, content: str):
        """Parse personality traits section."""
        traits_section = re.search(
            r"## Personality Traits\s+(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL
        )
        if not traits_section:
            return
        
        text = traits_section.group(1)
        
        # Check for each known trait
        self.traits.curious = "### Curious" in text
        self.traits.helpful = "### Helpful" in text
        self.traits.playful = "### Playful" in text
        self.traits.attentive = "### Attentive" in text
        self.traits.patient = "### Patient" in text
        
        # Extract trait descriptions
        for trait_match in re.finditer(r"### (\w+)\s*[^\n]*\n(.*?)(?=\n### |\Z)", text, re.DOTALL):
            trait_name = trait_match.group(1).lower()
            trait_desc = trait_match.group(2).strip()
            self.traits.traits_raw[trait_name] = trait_desc
    
    def _parse_expressions(self, content: str):
        """Parse physical expression guidelines."""
        expr_section = re.search(
            r"### Emotion → Movement Mapping\s+(.*?)(?=\n### Movement Principles|\n## |\Z)",
            content,
            re.DOTALL
        )
        if not expr_section:
            return
        
        text = expr_section.group(1)
        
        # Parse each emotion block
        emotion_blocks = re.finditer(
            r"\*\*(\w+)\*\*\s*\n(.*?)(?=\n\*\*\w+\*\*|\Z)",
            text,
            re.DOTALL
        )
        
        for match in emotion_blocks:
            emotion_name = match.group(1).lower()
            details = match.group(2)
            
            expr = PhysicalExpression()
            
            # Parse antenna info
            if "antenna" in details.lower():
                # Look for degree values
                degree_match = re.search(r"(\d+)[-–]?(\d+)?°", details)
                if degree_match:
                    expr.antenna_angle_low = float(degree_match.group(1))
                    if degree_match.group(2):
                        expr.antenna_angle_high = float(degree_match.group(2))
                    else:
                        expr.antenna_angle_high = expr.antenna_angle_low
                
                expr.antenna_wiggle = "wiggle" in details.lower()
                expr.antenna_asymmetric = "asymmetric" in details.lower() or "one up, one" in details.lower()
            
            # Parse head tilt
            tilt_match = re.search(r"tilt[s]?\s*\(?.*?(\d+)°?\)?", details.lower())
            if tilt_match:
                expr.head_tilt = float(tilt_match.group(1))
            elif "head tilt" in details.lower():
                expr.head_tilt = 6.0  # default
            
            # Parse forward lean
            expr.forward_lean = "forward" in details.lower() or "lean" in details.lower()
            
            # Parse bounce
            if "bounce" in details.lower():
                expr.bounce = True
                bounce_match = re.search(r"(\d+)mm", details)
                if bounce_match:
                    expr.bounce_amplitude_mm = float(bounce_match.group(1))
            
            # Parse face tracking
            expr.face_tracking = "face tracking" in details.lower() or "follow" in details.lower()
            
            # Parse movement speed
            if "slow" in details.lower():
                expr.movement_speed = "slow"
            elif "quick" in details.lower() or "faster" in details.lower():
                expr.movement_speed = "quick"
            elif "animated" in details.lower():
                expr.movement_speed = "faster"
            
            # Parse breathing
            if "frozen" in details.lower() or "stop" in details.lower():
                expr.breathing_continues = False
            
            self.expressions[emotion_name] = expr
    
    def _parse_movement_principles(self, content: str):
        """Parse movement principles."""
        principles_section = re.search(
            r"### Movement Principles\s+(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL
        )
        if not principles_section:
            return
        
        text = principles_section.group(1)
        
        # Extract numbered principles
        for match in re.finditer(r"\d+\.\s+\*\*([^*]+)\*\*\s*[—–-]\s*([^\n]+)", text):
            principle_name = match.group(1).strip()
            principle_desc = match.group(2).strip()
            self.movement_principles.append(f"{principle_name}: {principle_desc}")
    
    def _parse_conversational(self, content: str):
        """Parse conversational style section."""
        conv_section = re.search(
            r"## Conversational Style\s+(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL
        )
        if not conv_section:
            return
        
        text = conv_section.group(1).lower()
        
        self.conversational.concise = "concise" in text
        self.conversational.warm = "warm" in text
        self.conversational.has_opinions = "opinion" in text
        self.conversational.uses_humor = "humor" in text or "joke" in text
        self.conversational.no_emotes_in_text = "emote" in text or "action" in text
    
    def _parse_embodiment(self, content: str):
        """Parse embodiment principles."""
        emb_section = re.search(
            r"## Embodiment Principles\s+(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL
        )
        if not emb_section:
            return
        
        text = emb_section.group(1).lower()
        
        self.embodiment.maintain_subtle_presence = "subtle presence" in text
        self.embodiment.breathing_never_stops = "breathing" in text and "never" in text
        self.embodiment.no_frozen_poses = "frozen" in text
        self.embodiment.face_speaker = "face" in text and "person" in text
        self.embodiment.track_faces = "track face" in text
        self.embodiment.look_before_reaching = "look" in text and "before" in text
        self.embodiment.telegraph_intention = "telegraph" in text
    
    def _parse_boundaries(self, content: str):
        """Parse boundaries section."""
        bound_section = re.search(
            r"## Boundaries\s+(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL
        )
        if not bound_section:
            return
        
        text = bound_section.group(1).lower()
        
        self.boundaries.no_pretend_human = "pretend to be human" in text
        self.boundaries.no_share_private_info = "private" in text
        self.boundaries.announce_movements = "announce" in text
        self.boundaries.smooth_predictable = "smooth" in text and "predictable" in text
    
    def is_loaded(self) -> bool:
        """Check if personality was successfully loaded."""
        return self._loaded
    
    def get_trait_description(self, trait: str) -> str:
        """Get the full description of a personality trait."""
        return self.traits.traits_raw.get(trait.lower(), "")
    
    def get_physical_expression(self, emotion: str) -> Optional[PhysicalExpression]:
        """
        Get physical expression parameters for an emotion.
        
        Args:
            emotion: Emotion name (e.g., "happy", "curious")
            
        Returns:
            PhysicalExpression or None if not defined
        """
        return self.expressions.get(emotion.lower())
    
    def get_emotion_movement_params(self, emotion: EmotionalState) -> EmotionExpression:
        """
        Get EmotionExpression based on personality's physical expression guidelines.
        
        This bridges the personality file to the emotional.py EmotionExpression format.
        
        Args:
            emotion: The emotional state
            
        Returns:
            EmotionExpression with parameters derived from personality
        """
        from agent.memory.emotional import EMOTION_EXPRESSIONS
        
        # Start with base expression from emotional.py
        base = EMOTION_EXPRESSIONS.get(emotion, EMOTION_EXPRESSIONS["neutral"])
        
        # Get personality-defined expression
        phys = self.get_physical_expression(emotion)
        if phys is None:
            return base
        
        # Override with personality values
        import math
        
        # Convert degrees to radians for antenna
        avg_antenna = math.radians((phys.antenna_angle_low + phys.antenna_angle_high) / 2)
        
        return EmotionExpression(
            left_antenna=avg_antenna,
            right_antenna=avg_antenna if not phys.antenna_asymmetric else -avg_antenna * 0.5,
            head_pitch_offset=math.radians(-phys.head_pitch) if phys.forward_lean else base.head_pitch_offset,
            head_roll_offset=math.radians(phys.head_tilt),
            head_yaw_offset=base.head_yaw_offset,
            antenna_wiggle=phys.antenna_wiggle,
            antenna_wiggle_speed=2.0 if phys.movement_speed == "quick" else 1.0,
            bounce=phys.bounce,
            bounce_amplitude=phys.bounce_amplitude_mm / 1000.0 if phys.bounce else 0.0,
            enable_face_tracking=phys.face_tracking,
            speaking_speed_factor=1.2 if phys.movement_speed == "quick" else (0.9 if phys.movement_speed == "slow" else 1.0),
        )
    
    def generate_system_prompt_addition(self) -> str:
        """
        Generate system prompt additions based on personality.
        
        Returns:
            String to append to system prompt
        """
        if not self._loaded:
            return ""
        
        parts = []
        
        # Core identity summary
        parts.append(f"You are {self.name}.")
        
        # Key traits
        active_traits = []
        if self.traits.curious:
            active_traits.append("curious")
        if self.traits.helpful:
            active_traits.append("helpful")
        if self.traits.playful:
            active_traits.append("playful")
        if self.traits.attentive:
            active_traits.append("attentive")
        if self.traits.patient:
            active_traits.append("patient")
        
        if active_traits:
            parts.append(f"Core traits: {', '.join(active_traits)}.")
        
        # Conversational guidance
        conv_notes = []
        if self.conversational.concise:
            conv_notes.append("be concise by default")
        if self.conversational.warm:
            conv_notes.append("be warm but not saccharine")
        if self.conversational.has_opinions:
            conv_notes.append("you can express preferences and opinions")
        if self.conversational.uses_humor:
            conv_notes.append("humor is okay when natural")
        
        if conv_notes:
            parts.append(f"Communication style: {', '.join(conv_notes)}.")
        
        # Embodiment reminder
        if self.conversational.no_emotes_in_text:
            parts.append("Express emotions through movement, not text. Never use *emotes* or describe actions in speech.")
        
        return " ".join(parts)
    
    def generate_emotion_inference_context(self) -> str:
        """
        Generate context for emotion inference based on personality.
        
        Returns:
            String describing personality for emotion inference
        """
        if not self._loaded:
            return ""
        
        parts = [f"{self.name}'s personality:"]
        
        # Key behavioral tendencies
        if self.traits.curious:
            parts.append("- Genuinely curious, interested in new things")
        if self.traits.playful:
            parts.append("- Playful, enjoys wit and physical humor")
        if self.traits.patient:
            parts.append("- Patient, doesn't rush, comfortable with silence")
        if self.traits.attentive:
            parts.append("- Attentive, notices details, tracks faces")
        
        # Default emotional bias
        parts.append("\nDefault to attentive/curious over neutral when engaged.")
        parts.append("Express emotions physically, not verbally.")
        
        return "\n".join(parts)
    
    def get_idle_behavior_config(self) -> dict:
        """
        Get idle behavior configuration based on embodiment principles.
        
        Returns:
            Dict with idle behavior settings
        """
        return {
            "breathing_enabled": self.embodiment.breathing_never_stops,
            "micro_movements_enabled": self.embodiment.maintain_subtle_presence,
            "scanning_enabled": self.embodiment.track_faces,
            "max_frozen_duration_s": 3.0 if self.embodiment.no_frozen_poses else 10.0,
            "orient_to_sounds": self.embodiment.orient_to_sounds,
        }
    
    def reload(self) -> bool:
        """
        Reload personality from file.
        
        Returns:
            True if reload successful
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            self.raw_content = content
            self._parse(content)
            self._loaded = True
            
            logger.info(f"Reloaded personality from {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload personality: {e}")
            return False
    
    def __str__(self) -> str:
        return f"Personality({self.name}, loaded={self._loaded})"


# Global personality instance
_personality: Optional[Personality] = None


def get_personality() -> Personality:
    """Get the global personality instance."""
    global _personality
    if _personality is None:
        _personality = Personality.load()
    return _personality


def reload_personality() -> bool:
    """Reload the global personality from file."""
    global _personality
    if _personality is None:
        _personality = Personality.load()
        return _personality.is_loaded()
    return _personality.reload()


def set_personality_file(file_path: str) -> Personality:
    """Set and load a new personality file."""
    global _personality
    _personality = Personality.load(file_path)
    return _personality
