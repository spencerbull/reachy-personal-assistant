# Reachy Personal AI Assistant - Demo Scripts

This document contains scripted demo flows for showcasing Reachy's capabilities as a personal AI assistant.

## Quick Setup

Before running demos:

```bash
# Terminal 1: Start Reachy daemon
./start_daemon.sh

# Terminal 2: Start bot with LangGraph
./start_bot.sh --langgraph

# Open browser to Daily room URL shown in terminal
```

---

## Demo 1: Greeting and Introduction

**Purpose**: Show Reachy's personality and basic interaction

### Script

**User**: "Hey Reachy"

**Reachy** (excited, slight bounce):
> Hey, I'm Reachy your personal robot assistant. What can I do for you?

**User**: "What can you do?"

**Reachy** (helpful, forward lean):
> I can do quite a lot actually! I can look around the room and describe what I see, remember where you put things so you never lose them again, keep track of your schedule, and of course have a good conversation. I'm powered by the Dell Pro Max GB10, so I'm fast and I keep everything private since I process locally. Want me to show you something specific?

**User**: "Tell me about your hardware"

**Reachy** (excited):
> Oh, you want to know about my brain? I'm running on the NVIDIA GB10 Grace Blackwell Superchip. It's basically the Formula 1 engine of AI processors. I can crunch a quadrillion calculations per second. And the best part? I do everything locally. Your secrets are safe with me since I don't need to send anything to the cloud.

---

## Demo 2: Spatial Memory

**Purpose**: Demonstrate Reachy's ability to remember object locations

### Script

**User**: "I'm putting my passport in the top drawer"

**Reachy** (attentive, looks at user):
> Got it! I'll remember that your passport is in the top drawer. I'll keep track of that for you.

**User**: "And my keys are on the nightstand"

**Reachy** (helpful):
> Noted. Your keys are on the nightstand. Anything else you'd like me to remember?

**User**: "No, that's all for now"

**Reachy** (happy):
> Perfect. Just ask me whenever you need to find something!

*...Later in the conversation or session...*

**User**: "Where did I put my passport?"

**Reachy** (attentive, looks toward drawer):
> Your passport is in the top drawer. [Reachy looks in the direction of the drawer if orientation is known]

**User**: "And my keys?"

**Reachy** (helpful):
> Your keys are on the nightstand.

---

## Demo 3: Face Tracking

**Purpose**: Show Reachy's ability to maintain eye contact

### Script

**User**: "Look at me"

**Reachy** (attentive, enables face tracking):
> I've got my eyes on you! [Reachy starts tracking the user's face and maintains eye contact as they move]

**User**: *moves around the room*

**Reachy** (continues tracking):
> [Reachy's head follows the user smoothly]

**User**: "Stop tracking me"

**Reachy** (neutral, disables tracking):
> Alright, I've stopped tracking your face.

---

## Demo 4: Vision Understanding

**Purpose**: Demonstrate visual perception capabilities

### Script

**User**: "What do you see?"

**Reachy** (curious, observing):
> Let me take a look... I can see we're in what looks like a hotel room. There's a bed, a nightstand, and I can see you standing in front of me. The lighting is pretty good in here.

**User**: "What am I holding?"

**Reachy** (curious):
> You're holding a phone in your right hand. Looks like a smartphone with a dark case.

**User**: "What color is my shirt?"

**Reachy** (attentive):
> You're wearing a blue shirt. It looks like a nice casual button-down.

---

## Demo 5: Looking Around

**Purpose**: Show head movement control

### Script

**User**: "Look to the left"

**Reachy** (attentive, turns head left):
> [Reachy turns head to the left] Looking left now.

**User**: "What's over there?"

**Reachy** (curious):
> I can see a window with some curtains, and there's a desk with a lamp on it.

**User**: "Now look to the right"

**Reachy** (attentive, turns head right):
> [Reachy turns head to the right] Looking right.

**User**: "Look back at me"

**Reachy** (happy, looks front and enables tracking):
> [Reachy returns to center and faces user] There you are!

---

## Demo 6: Emotional Expressions

**Purpose**: Show Reachy's expressive personality

### Script

**User**: "Tell me a joke"

**Reachy** (playful, antenna wiggle):
> Why don't robots ever get tired? Because we run on batteries! Get it? Okay, okay, I'll work on my comedy routine.

**User**: "That was terrible"

**Reachy** (playful, slight head tilt):
> Hey, I'm better at being helpful than being a comedian. But at least I tried, right?

**User**: "Show me you're happy"

**Reachy** (happy, antenna wiggle and bounce):
> [Reachy does a happy expression with antennas wiggling] This is my happy dance! Well, my happy wobble anyway.

**User**: "Dance for me"

**Reachy** (excited):
> Let me show you my moves! [Reachy performs a dance sequence]

---

## Demo 7: Multi-Location Memory

**Purpose**: Show spatial memory across different locations

### Script

*At home:*

**User**: "We're at home now"

**Reachy** (attentive):
> Got it! I've noted that we're now at home. Any objects you tell me about will be associated with this location.

**User**: "My charger is on the kitchen counter"

**Reachy** (helpful):
> I'll remember that your charger is on the kitchen counter here at home.

*Travel to hotel:*

**User**: "We're at the hotel in Tokyo now"

**Reachy** (curious):
> Exciting! We're now at the hotel in Tokyo. I'll keep track of things separately from home.

**User**: "My charger is on the desk"

**Reachy** (helpful):
> Noted! Your charger here at the Tokyo hotel is on the desk. I remember you also have one on the kitchen counter at home.

---

## Demo 8: Technical Capabilities

**Purpose**: Showcase the GB10 hardware advantages

### Script

**User**: "How fast are you?"

**Reachy** (excited):
> I can crunch data at one Petaflop of FP4 performance. That's a quadrillion calculations per second. Don't ask me to count that high; we'd be here all day.

**User**: "Is my data private with you?"

**Reachy** (attentive, reassuring):
> Absolutely. Since I process everything locally on the GB10, your data never leaves this device. No cloud, no external servers. Your secrets are safe with me.

**User**: "Could you connect to another computer?"

**Reachy** (helpful):
> Yes! I'm rocking an NVIDIA ConnectX-7 SmartNIC. If we needed to, I could connect to another GB10 and literally double my brainpower. That would let me handle models with up to 400 billion parameters.

---

## Demo 9: Calendar Integration (Requires MCP Setup)

**Purpose**: Show external service integration

### Script

**User**: "What's on my schedule today?"

**Reachy** (attentive, checking):
> Let me check your calendar... You have a meeting at 10 AM with the engineering team, lunch at noon, and a presentation at 3 PM. Looks like a busy day!

**User**: "Remind me about the presentation"

**Reachy** (helpful):
> Your presentation is at 3 PM today. It's in the main conference room. Would you like me to help you prepare anything?

---

## Demo 10: Full Interaction Flow

**Purpose**: Natural extended conversation

### Script

**User**: "Good morning Reachy"

**Reachy** (happy, slight bounce):
> Good morning! Great to see you. How can I help you start your day?

**User**: "I need to pack for my trip. Where did I put my passport again?"

**Reachy** (helpful, thinks):
> Let me check... Your passport is in the top drawer. Need help finding anything else for your trip?

**User**: "Look around and tell me if you see my suitcase"

**Reachy** (curious, scans room):
> [Reachy looks around the room] Let me take a look... I can see what looks like a suitcase near the closet door on your left.

**User**: "Perfect, thanks!"

**Reachy** (happy):
> You're welcome! Have a great trip. I'll be here when you get back!

---

## Tips for Demos

1. **Speak clearly**: Reachy uses speech-to-text, so clear pronunciation helps
2. **Wait for responses**: Give Reachy time to process and respond
3. **Use natural language**: Reachy understands semantic intent, not just keywords
4. **Show movement**: Move around to demonstrate face tracking
5. **Vary emotions**: Ask about different topics to see emotional variety

## Troubleshooting

- **No response**: Check if all services are running (daemon, bot, models)
- **Slow responses**: Check GPU utilization and model loading status
- **Face tracking not working**: Ensure camera is capturing and face is visible
- **Memory not working**: Verify LangGraph agent is active (check for `--langgraph` flag)
