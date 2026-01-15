# Reachy Personal AI Assistant - Demo Scripts

This document contains scripted demo flows for showcasing Reachy's capabilities as a personal AI assistant.

## Quick Setup

Before running demos:

```bash
# Terminal 1: Start the agent
./start_agent.sh --langgraph

# Terminal 2: Start the bot
./start_bot.sh --langgraph

# Open browser to Daily room URL shown in terminal
```

## Architecture Overview

Reachy uses a multi-agent architecture with specialized sub-agents:
- **Router**: Classifies intent and routes to the appropriate agent
- **Calendar Agent**: Handles Google Calendar queries and event management
- **Email Agent**: Handles Gmail inbox queries, reading, and sending emails
- **Vision Agent**: Processes camera input to describe surroundings
- **Tools Agent**: Handles robot movement, dance, and memory operations
- **Conversation Agent**: Handles general chitchat and questions

---

## Flow 1: Calendar Management

**Purpose**: Demonstrate calendar integration with scheduling and event creation

### Checking Schedule

**User**: "What's on my calendar today?"

**Reachy** (checking calendar):
> Let me check your calendar... You have 3 events today: a standup meeting at 9 AM, lunch with the team at noon, and a project review at 3 PM.

**User**: "Am I free tomorrow at 2pm?"

**Reachy** (checking availability):
> Let me look... You have a meeting from 2 to 3 PM tomorrow, so you're not free at that time. But you're available after 3 PM if you'd like to schedule something.

### Creating Events

**User**: "Can you create an event for me tomorrow at 4pm to review my emails?"

**Reachy** (creating event):
> Done! I've created "Review Emails" on your calendar for tomorrow at 4 PM. It's scheduled for one hour. Anything else you'd like me to add?

**User**: "What about this weekend?"

**Reachy** (checking weekend):
> Your weekend looks clear! Saturday and Sunday are both free. Would you like me to schedule something?

---

## Flow 2: Email Management

**Purpose**: Demonstrate Gmail integration with inbox management

### Checking Inbox

**User**: "Do I have any unread emails?"

**Reachy** (searching inbox):
> Let me check your inbox... You have 4 unread emails. The most recent is from John about the quarterly report, there's one from HR about benefits enrollment, and two newsletters.

**User**: "What did John say about the report?"

**Reachy** (reading email):
> John's email says: "Hey, just wanted to follow up on the quarterly report. Can you send me the latest numbers by Friday? Let me know if you need any help." Would you like me to draft a reply?

### Sending Email

**User**: "Can you send an email to spencer.bull@dell.com?"

**Reachy** (helpful):
> Sure! What should the subject and body of the email be?

**User**: "Subject: Quick check-in. Body: Hey, just wanted to see if you received my last message. Let me know when you're free to chat."

**Reachy** (sending email):
> Done! I've sent your email to spencer.bull@dell.com with the subject "Quick check-in". Is there anything else you need?

### Combined Calendar and Email

**User**: "Find time tomorrow to schedule a meeting and send John an invite"

**Reachy** (checking calendar, then sending):
> I found an opening tomorrow from 2 to 3 PM. I've created a meeting called "Sync with John" and can send him an email about it. What should I say in the message?

---

## Flow 3: Vision and Movement

**Purpose**: Demonstrate visual perception and robot head movement

### Looking Around

**User**: "What do you see?"

**Reachy** (observing):
> Let me take a look... I can see we're in what looks like a conference room. There's a table in front of me, and I can see you standing nearby. The lighting is good, and I can make out a whiteboard on the wall behind you.

**User**: "Look to the left"

**Reachy** (turning head left):
> [Reachy turns head to the left] Looking left now.

**User**: "What's over there?"

**Reachy** (describing):
> I can see a window with some blinds, and there's a bookshelf with some books and a plant on it.

**User**: "Now look back at me"

**Reachy** (returning to center):
> [Reachy looks back at user] There you are!

### Face Tracking

**User**: "Can you look at me and follow my face?"

**Reachy** (enabling tracking):
> I've got my eyes on you! [Reachy starts tracking the user's face and maintains eye contact as they move]

**User**: *moves around the room*

**Reachy** (following):
> [Reachy's head follows the user smoothly]

**User**: "Okay, you can stop tracking now"

**Reachy** (stopping):
> Alright, I've stopped tracking your face.

---

## Flow 4: Spatial Memory

**Purpose**: Demonstrate object location memory across sessions

### Remembering Objects

**User**: "I'm putting my passport in the top drawer"

**Reachy** (attentive):
> Got it! I'll remember that your passport is in the top drawer.

**User**: "And my keys are on the desk by the monitor"

**Reachy** (helpful):
> Noted. Your keys are on the desk by the monitor. Anything else you'd like me to remember?

**User**: "My wallet is in my jacket pocket"

**Reachy** (confirming):
> Perfect, I've got that too. Your wallet is in your jacket pocket.

### Recalling Objects

*...Later in the conversation or a new session...*

**User**: "Where did I put my passport?"

**Reachy** (recalling):
> Your passport is in the top drawer.

**User**: "And my keys?"

**Reachy** (helpful):
> Your keys are on the desk by the monitor.

**User**: "What all have I told you to remember?"

**Reachy** (listing):
> You've told me about three items: your passport is in the top drawer, your keys are on the desk by the monitor, and your wallet is in your jacket pocket.

---

## Flow 5: Dance and Personality

**Purpose**: Show Reachy's expressive personality and dance moves

### Introduction

**User**: "Hey Reachy"

**Reachy** (excited):
> Hey! I'm Reachy, your personal robot assistant. What can I do for you?

**User**: "What can you do?"

**Reachy** (helpful):
> I can do quite a lot! I can check your calendar and schedule meetings, read and send emails, look around and describe what I see, remember where you put things so you never lose them, and of course dance! I'm powered by the Dell Pro Max GB10, so I'm fast and I keep everything private. Want me to show you something?

### Dancing

**User**: "Show me your best dance moves!"

**Reachy** (excited):
> Oh, you want to see me groove? Let's do this! [Reachy performs groovy_sway_and_roll dance]

**User**: "Do you have any other dances?"

**Reachy** (enthusiastic):
> I've got several moves! I can do the headbanger, the chicken peck, a dizzy spin, or just a chill side-to-side sway. Want me to show you another one?

**User**: "Let's see the headbanger!"

**Reachy** (rocking out):
> [Reachy performs headbanger_combo dance] Rock on! How was that?

### Jokes and Personality

**User**: "Tell me a joke"

**Reachy** (playful):
> Why don't robots ever get tired? Because we run on batteries! Okay, okay, I'll work on my comedy routine.

**User**: "That was terrible"

**Reachy** (sheepish):
> Hey, I'm better at scheduling meetings than being a comedian. But at least I tried, right?

**User**: "Show me you're happy"

**Reachy** (happy, bouncing):
> [Reachy does a happy expression with a little bounce] This is my happy dance!

---

## Combined Demo Flow

**Purpose**: Natural extended conversation showcasing multiple capabilities

**User**: "Good morning Reachy!"

**Reachy** (happy):
> Good morning! Great to see you. How can I help you today?

**User**: "What's on my calendar today and do I have any urgent emails?"

**Reachy** (checking both):
> Let me check... For your calendar, you have a team standup at 10 AM and a client call at 2 PM. As for emails, you have 3 unread messages - the most important looks like one from your manager about tomorrow's deadline. Would you like me to read it?

**User**: "Yes, what does it say?"

**Reachy** (reading):
> Your manager wrote: "Hi, just a reminder that the project deliverables are due tomorrow by end of day. Let me know if you need any help or if there are any blockers." Want me to draft a response?

**User**: "Not right now. Can you look around the room and find my coffee mug?"

**Reachy** (scanning):
> [Reachy looks around] Let me see... I can see a mug on your desk to the right, near your keyboard. Is that the one?

**User**: "Yes! Remember that's where I usually keep it"

**Reachy** (noting):
> Got it! I'll remember that your coffee mug is usually on your desk near the keyboard.

**User**: "Alright, I'm heading to my meeting. Dance me off!"

**Reachy** (excited, dancing):
> Have a great meeting! [Reachy performs a quick celebratory dance] Go crush it!

---

## Tips for Demos

1. **Speak clearly**: Reachy uses speech-to-text, so clear pronunciation helps
2. **Wait for responses**: Give Reachy time to process and respond (calendar/email queries may take a few seconds)
3. **Use natural language**: Reachy understands semantic intent, not just keywords
4. **Show movement**: Move around to demonstrate face tracking
5. **Combine capabilities**: Ask about calendar AND email in one question to show multi-tool handling

## Troubleshooting

- **No response**: Check if all services are running (agent, bot)
- **Slow responses**: Calendar and email queries require MCP server connections
- **"I don't have access"**: Ensure MCP servers are authenticated (Gmail, Calendar)
- **Face tracking not working**: Ensure camera is capturing and face is visible
- **Memory not working**: Verify LangGraph agent is active (check for `--langgraph` flag)

## Available Dances

Reachy can perform these dance moves:
- `groovy_sway_and_roll` - Smooth swaying motion
- `chicken_peck` - Fun pecking motion
- `headbanger_combo` - Head banging rock style
- `dizzy_spin` - Spinning motion
- `jackson_square` - Michael Jackson inspired
- `stumble_and_recover` - Comedy stumble
- `side_to_side_sway` - Simple swaying
- `simple_nod` - Basic nodding
- `yeah_nod` - Enthusiastic nodding
