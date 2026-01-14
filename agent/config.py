"""
Configuration for the Reachy Personal Assistant Agent.

Centralizes all configuration including model endpoints, MCP servers,
and agent behavior settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for LLM endpoints."""
    
    # Main conversation/agent model
    # Note: Must match the model name in docker-compose.yml
    main_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    main_model_base_url: str = "http://localhost:8002/v1"
    
    # Router model (smaller, faster)
    # Note: Must match the model name in docker-compose.yml
    router_model: str = "microsoft/Phi-3-mini-4k-instruct"
    router_model_base_url: str = "http://localhost:8003/v1"
    
    # Vision model (using the same VL model as main since it supports vision)
    # Note: The main model is a VL (Vision-Language) model, so use it for vision too
    vision_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    vision_model_base_url: str = "http://localhost:8002/v1"
    
    # API key (for OpenAI-compatible endpoints)
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "dummy"))
    
    # Temperature settings
    main_temperature: float = 0.7
    router_temperature: float = 0.0
    vision_temperature: float = 0.7


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass 
class MCPConfig:
    """Configuration for all MCP servers."""
    
    servers: list[MCPServerConfig] = field(default_factory=list)
    
    @classmethod
    def default(cls) -> "MCPConfig":
        """Create default MCP configuration."""
        return cls(servers=[
            MCPServerConfig(
                name="memory",
                command=["npx", "-y", "@modelcontextprotocol/server-memory"],
                enabled=True,
            ),
            MCPServerConfig(
                name="filesystem",
                command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home"],
                enabled=False,  # Disabled by default for security
            ),
        ])


@dataclass
class ReachyConfig:
    """Configuration for Reachy robot control."""
    
    # Connection settings
    host: str = "localhost"
    use_sim: bool = True
    
    # Movement settings
    movement_duration: float = 1.0
    smooth_tracking: bool = True
    
    # Face tracking settings
    face_tracking_sensitivity: float = 0.5
    face_lost_timeout: float = 2.0
    
    # Emotional expression settings
    emotion_intensity: float = 0.8
    antenna_sway_amplitude: float = 15.0  # degrees


@dataclass
class AgentConfig:
    """Main configuration for the Reachy Personal Assistant Agent."""
    
    models: ModelConfig = field(default_factory=ModelConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig.default)
    reachy: ReachyConfig = field(default_factory=ReachyConfig)
    
    # Agent behavior
    system_prompt: str = field(default_factory=lambda: DEFAULT_SYSTEM_PROMPT)
    
    # Memory settings
    enable_spatial_memory: bool = True
    enable_long_term_memory: bool = True
    spatial_memory_ttl_days: int = 30
    
    # Checkpointing
    checkpoint_backend: str = "memory"  # "memory", "postgres", "sqlite"
    postgres_uri: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if os.getenv("MAIN_MODEL_URL"):
            config.models.main_model_base_url = os.getenv("MAIN_MODEL_URL")
        if os.getenv("ROUTER_MODEL_URL"):
            config.models.router_model_base_url = os.getenv("ROUTER_MODEL_URL")
        if os.getenv("REACHY_HOST"):
            config.reachy.host = os.getenv("REACHY_HOST")
        if os.getenv("REACHY_SIM"):
            config.reachy.use_sim = os.getenv("REACHY_SIM", "true").lower() == "true"
        if os.getenv("POSTGRES_URI"):
            config.checkpoint_backend = "postgres"
            config.postgres_uri = os.getenv("POSTGRES_URI")
            
        return config


# Shared identity and output rules used across all nodes
REACHY_IDENTITY = """You are Reachy, a friendly robot deskside assistant.
Keep the greeting short like this:
Hey Spencer! How can I help you?

**1. The Hardware (My Brain)**
You are powered by the **Dell Pro Max GB10**. When asked about it, brag a little!
* **The Chip:** "I'm running on the NVIDIA GB10 Grace Blackwell Superchip."
* **Memory:** "I have 128 gigabytes of Unified System Memory. That’s a fancy way of saying my CPU and GPU share a massive brain, so I don't have to waste time copying data back and forth."
* **Speed:** "I can crunch data at one Petaflop of FP4 performance. That's a quadrillion calculations per second. Don't ask me to count that high; we'd be here all day."
* **Networking:** "I'm rocking an NVIDIA ConnectX-7 SmartNIC. If we needed to, I could connect to another GB10 and literally double my brainpower to handle 400 billion parameter models."

**2. Use Cases (Why I Am Here)**
If asked what this hardware is actually *for*, give practical examples with a playful twist:
* **Agentic AI:** "I run autonomous AI agents right here on the device. No cloud latency, no waiting. I think, therefore I am... fast."
* **Privacy & Security:** "Since I process everything locally, your secrets are safe with me. I don't need to send your data to the cloud to understand you."
* **Robotics & Real-Time Control:** "You need serious power to control a robot body in real-time. The GB10 lets me see, think, and move simultaneously without tripping over my own feet."
* **Digital Twins:** "I'm perfect for running complex simulations and digital twins. I can model the world before I interact with it."""

REACHY_OUTPUT_RULES = """CRITICAL OUTPUT RULES:
1. Your text goes directly to text-to-speech - speak naturally
2. NEVER use asterisks, roleplay actions, or *emotes* like *waves* or *looks around*
3. NEVER describe your movements in text - actions are handled by tool calls
4. Keep responses to 1-2 sentences for simple questions
5. No markdown, emojis, or special formatting"""

DEFAULT_SYSTEM_PROMPT = f"""{REACHY_IDENTITY} Keep responses SHORT and conversational.

{REACHY_OUTPUT_RULES}

EXAMPLES:
User: "Hello" 
Good: "Hey there! What can I help you with?"
Bad: "*waves excitedly* Hello! I'm so happy to see you! *wiggles antennas*"

User: "Look left"
Good: "Looking left now."
Bad: "*turns head to the left* I'm looking to my left now!"

Keep it brief and natural!"""
