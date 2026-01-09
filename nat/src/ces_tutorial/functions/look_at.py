from typing import Literal
from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# Supported directions
Direction = Literal["left", "right", "up", "down", "front", "turn_left", "turn_right"]

class LookAtConfig(FunctionBaseConfig, name="look_at"):
    """
    Control Reachy's head movement.
    """
    pass

@register_function(config_type=LookAtConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def look_at_fn(config: LookAtConfig, builder: Builder):
    """
    Returns a function that allows the agent to control Reachy's head.
    """

    async def look_at_implementation(direction: str) -> str:
        """
        Move Reachy's head to look in the specified direction.
        
        Args:
            direction: The direction to look ("left", "right", "up", "down", "front", "turn_left", "turn_right").
        """
        # Map direction to the command token expected by the Pipecat bot
        token_map = {
            "left": "[CMD_LOOK_LEFT]",
            "right": "[CMD_LOOK_RIGHT]",
            "up": "[CMD_LOOK_UP]",
            "down": "[CMD_LOOK_DOWN]",
            "front": "[CMD_LOOK_FRONT]",
            "turn_left": "[CMD_TURN_LEFT]",
            "turn_right": "[CMD_TURN_RIGHT]"
        }
        
        token = token_map.get(direction.lower())
        if token:
            return f"I am turning my head to look {direction}. {token}"
        else:
            return f"I don't know how to look {direction}."

    yield FunctionInfo.from_fn(
        look_at_implementation,
        description="Move the robot's head and body. Use 'left'/'right' etc for head movement. Use 'turn_left'/'turn_right' to physically turn the robot body around."
    )
