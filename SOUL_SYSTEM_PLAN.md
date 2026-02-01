# Reachy Soul System - Implementation Plan

*Created: 2026-02-01*
*Branch: `feature/soul-system`*

---

## Overview

Transform Reachy from a reactive chatbot into a **living presence** — an entity that exists in the room, has continuous awareness, builds spatial memory, and expresses emotions naturally without explicit tool calls.

---

## Problem Statement

Current issues with the existing implementation:

1. **Hardcoded command tokens** — Tools return strings like `[CMD_EMOTION_HAPPY]` parsed downstream. Fragile and not contextually aware.

2. **Emotions are tool-based** — LLM must explicitly call `express_emotion_tool()`. Unnatural.

3. **Disconnected emotion system** — `EmotionalStateManager` exists but isn't integrated into the graph flow.

4. **No ambient presence** — No idle movements, breathing, or reactions while listening. Reachy only moves when commanded.

5. **No spatial awareness** — Reachy doesn't remember where things are in the room.

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EMBODIMENT LOOP (always running)                 │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │ Idle Gen     │    │ Emotion      │    │ Movement     │        │
│   │ (breathing,  │───►│ State        │───►│ Blender      │───► Reachy
│   │  micro-moves)│    │ Machine      │    │ (smooth      │        │
│   └──────────────┘    └──────────────┘    │  transitions)│        │
│          ▲                   ▲            └──────────────┘        │
│          │                   │                                     │
│          │    ┌──────────────┴──────────────┐                     │
│          │    │      Soul LLM (async)       │                     │
│          │    │  - Analyzes conversation    │                     │
│          │    │  - Infers emotional state   │                     │
│          │    │  - Runs every ~200ms        │                     │
│          │    └─────────────────────────────┘                     │
│          │                   ▲                                     │
└──────────│───────────────────│─────────────────────────────────────┘
           │                   │
           │     ┌─────────────┴─────────────┐
           │     │   Conversation Events     │
           │     │   - user_speaking         │
           │     │   - user_silent           │
           │     │   - bot_responding        │
           │     │   - message_content       │
           └─────┴───────────────────────────┘
                            ▲
                            │
     User ──► STT ──► Main Agent (Qwen3-VL) ──► TTS ──► Speaker
```

---

## Hardware Context

**Dell Pro Max with GB10 (NVIDIA Grace Blackwell)**
- 128GB unified LPDDR5X memory
- 1 petaflop FP4 compute
- Currently using FP8 quantization (NVFP4 not yet available for Qwen3-VL)
- Single unified memory pool — no CPU↔GPU transfer bottleneck

**Current vLLM Setup:**
- `agent-llm`: Qwen3-VL-30B-A3B-Instruct-FP8 on port 8002 (60% GPU)
- `routing-llm`: Phi-3-mini on port 8003 (15% GPU)
- ~25% headroom available

**Target vLLM Config:**
```yaml
command: >
  vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
  --port 8000 
  --host 0.0.0.0 
  --gpu-memory-utilization 0.85
  --max-model-len 64000
  --max-num-seqs 4  # Enable parallel requests for soul + main agent
  --enable-auto-tool-choice
  --tool-call-parser hermes
  --enable-prefix-caching
  --quantization fp8
```

---

## Implementation Phases

### Phase 1: Soul Loop (Core) ✅ TODO

**Goal:** Get the basic embodiment loop running with emotion inference.

**Components:**
- [ ] `soul/loop.py` — Async embodiment loop
- [ ] `soul/config.py` — Soul configuration dataclass
- [ ] `soul/emotion_inference.py` — LLM-based emotion inference
- [ ] `soul/movement_blender.py` — Smooth movement transitions
- [ ] `soul/idle_generator.py` — Breathing, micro-movements

**Key Behaviors:**
- Soul loop runs async, polls every 200ms (configurable)
- Infers emotional state from conversation context
- Generates movement commands blended smoothly
- Idle behaviors run continuously (breathing, micro-saccades)

**Remove/Deprecate:**
- `express_emotion_tool` — Soul handles this automatically
- Command token pattern (`[CMD_EMOTION_X]`) — Replace with movement requests

### Phase 2: Streaming Awareness ⏳ TODO

**Goal:** Soul reacts to response tokens as they stream, not just after.

**Components:**
- [ ] Hook into Pipecat's `LLMTextFrame` streaming
- [ ] Soul receives partial response text
- [ ] Expression evolves during speech (e.g., gets more excited as joke builds)

### Phase 3: Spatial Memory ⏳ TODO

**Goal:** Reachy remembers where things are in the room.

**Components:**
- [ ] `soul/spatial_memory.py` — Object/person tracking
- [ ] `soul/room_model.py` — Room layout representation
- [ ] Vision integration — Update spatial memory from camera
- [ ] Directed attention — "Look at the whiteboard" → knows it's to the left

**Key Behaviors:**
- Track objects relative to Reachy (direction, distance, last_seen)
- Track people (face_id, position, last_seen)
- Query: "Where is X?" → returns look direction
- Decay old memories over time (configurable)

### Phase 4: Autonomous Presence ⏳ TODO

**Goal:** Reachy has presence even when user isn't engaging.

**Components:**
- [ ] Sleep/wake states
- [ ] Autonomous room scanning when idle
- [ ] Building spatial model proactively
- [ ] Idle curiosity behaviors

---

## Configuration

```python
@dataclass
class SoulConfig:
    # Polling
    poll_interval_ms: int = 200
    
    # Model
    soul_model_url: str = "http://localhost:8002/v1"  # Same as main agent
    soul_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    
    # Idle behavior
    idle_breathing_enabled: bool = True
    idle_breathing_frequency_hz: float = 0.5
    idle_micro_movements_enabled: bool = True
    idle_scanning_enabled: bool = True
    idle_scan_interval_s: float = 30.0
    
    # Spatial memory
    spatial_memory_enabled: bool = True
    object_memory_decay_hours: float = 24.0
    
    # Streaming reaction
    react_to_streaming: bool = True
    
    # Event triggers (rule-based, no LLM)
    user_speaking_triggers_attention: bool = True
    user_silence_triggers_processing_look: bool = True
```

---

## File Structure

```
agent/
├── soul/
│   ├── __init__.py
│   ├── config.py          # SoulConfig dataclass
│   ├── loop.py            # Main embodiment loop
│   ├── emotion_inference.py  # LLM-based emotion detection
│   ├── movement_blender.py   # Smooth pose transitions
│   ├── idle_generator.py     # Breathing, micro-movements
│   ├── spatial_memory.py     # Object/person tracking
│   ├── room_model.py         # Room layout
│   └── events.py             # Event types and handlers
├── memory/
│   └── emotional.py       # UPDATE: Integrate with soul
├── tools/
│   └── reachy_tools.py    # UPDATE: Remove emotion tool, keep movement tools
└── graph.py               # UPDATE: Wire in soul loop
```

---

## Soul LLM Prompt (Emotion Inference)

Small, focused — runs every poll interval:

```
You are analyzing conversation context to determine emotional state.

Current state:
- User is: {user_state}  # speaking/silent/waiting
- Last user message: "{last_user_message}"
- Last bot response: "{last_bot_response}"
- Current emotional state: {current_emotion}
- Time since last interaction: {time_since_ms}ms

What should Reachy's emotional state be now?

Emotions: neutral, happy, curious, attentive, thinking, excited, helpful, playful

Respond with JSON only:
{"emotion": "...", "intensity": 0.0-1.0, "movement_hint": "..."}
```

---

## Event Triggers (Rule-Based, No LLM)

Immediate reactions without waiting for inference:

| Event | Reaction |
|-------|----------|
| User starts speaking | Face tracking ON, attentive pose |
| User stops speaking | Slight head tilt (processing) |
| Bot starts responding | Slight forward lean |
| Loud/sudden sound | Quick look toward source |
| Face detected | Track face, wave antenna greeting |
| Face lost | Return to neutral scan |

---

## Testing Checklist

### Phase 1
- [ ] Soul loop starts and runs async without blocking main agent
- [ ] Emotion inference returns valid states
- [ ] Movement blender smoothly transitions between poses
- [ ] Idle breathing is visible and natural
- [ ] No performance degradation on main agent responses

### Phase 2
- [ ] Soul receives streaming tokens
- [ ] Expression changes mid-response appropriately
- [ ] No race conditions between soul and response

### Phase 3
- [ ] Spatial memory stores object locations
- [ ] "Where is X?" queries return correct directions
- [ ] Idle scanning updates spatial memory
- [ ] Memory decays appropriately

### Phase 4
- [ ] Sleep mode activates after prolonged inactivity
- [ ] Wake on sound/motion
- [ ] Autonomous scanning builds room model

---

## PR Strategy

1. **PR #1: Phase 1** — Soul loop, emotion inference, movement blender, idle behaviors
2. **PR #2: Phase 2** — Streaming awareness
3. **PR #3: Phase 3** — Spatial memory
4. **PR #4: Phase 4** — Autonomous presence

Each PR should be independently testable and not break existing functionality.

---

## Notes & Decisions

- **2026-02-01**: Decided on Option C (Always-On Embodiment Agent)
- **2026-02-01**: Using same Qwen3-VL model with concurrency for soul (not separate model)
- **2026-02-01**: NVFP4 not available yet for Qwen3-VL, using FP8
- **2026-02-01**: Poll interval configurable, starting at 200ms
- **2026-02-01**: Soul should react to streaming response tokens
- **2026-02-01**: Spatial memory should track objects so Reachy knows where to look

---

## References

- Current repo: `spencerbull/reachy-personal-assistant` branch `spark-local-langgraph`
- Docker compose: `local/docker-compose.yml`
- Main graph: `agent/graph.py`
- Reachy tools: `agent/tools/reachy_tools.py`
- Emotional memory: `agent/memory/emotional.py`
- Dance/emotion moves: `bot/services/dance_emotion_moves.py`
