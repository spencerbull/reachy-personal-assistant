# Google Calendar MCP Setup

This guide explains how to set up the Google Calendar MCP integration for Reachy.

## Package

We use [@cocal/google-calendar-mcp](https://github.com/nspady/google-calendar-mcp) which provides comprehensive calendar tools.

## Available Tools

| Tool | Description |
|------|-------------|
| `list-calendars` | List all available calendars |
| `list-events` | List events with date filtering |
| `get-event` | Get details of a specific event by ID |
| `search-events` | Search events by text query |
| `create-event` | Create new calendar events |
| `update-event` | Update existing events |
| `delete-event` | Delete events |
| `respond-to-event` | Respond to event invitations (Accept, Decline, Maybe) |
| `get-freebusy` | Check availability across calendars |
| `get-current-time` | Get current date and time in calendar's timezone |
| `list-colors` | List available event colors |
| `manage-accounts` | Add, list, or remove connected Google accounts |

## Setup Instructions

### Step 1: Google Cloud Project Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Calendar API**:
   - Go to "APIs & Services" → "Library"
   - Search for "Google Calendar API"
   - Click "Enable"

### Step 2: Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in the required fields (app name, support email)
   - Add your email as a test user
4. For Application type, select **"Desktop app"**
5. Give it a name (e.g., "Reachy Calendar")
6. Click "Create"
7. Download the credentials JSON file

### Step 3: Configure Credentials

The MCP server uses `~/.config/google-calendar-mcp/` for storing credentials and tokens.

If you have a credentials file, you can set it via environment variable:

```bash
export GOOGLE_OAUTH_CREDENTIALS="/path/to/your/gcp-oauth.keys.json"
```

Or just run the auth command and it will handle credential setup automatically.

### Step 4: Authenticate

Run the authentication flow:

```bash
npx @cocal/google-calendar-mcp auth
```

This will:
1. Open your browser for Google OAuth consent
2. Ask you to authorize calendar access
3. Save the authentication tokens

### Step 5: Verify Setup

Test that the MCP server works:

```bash
npx @cocal/google-calendar-mcp
```

If successful, the server will start and wait for MCP connections.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_OAUTH_CREDENTIALS` | Path to OAuth credentials file (optional if using default location) |
| `GOOGLE_CALENDAR_MCP_TOKEN_PATH` | Custom token storage location (optional) |
| `CALENDAR_MCP_ENABLED` | Set to "true" to force-enable even without credentials |

## Re-authentication

If you're in test mode (default for new projects), tokens expire after 7 days.

To re-authenticate:

```bash
npx @cocal/google-calendar-mcp auth
```

To avoid weekly re-authentication, publish your app to production mode:
1. Go to Google Cloud Console → "APIs & Services" → "OAuth consent screen"
2. Click "PUBLISH APP" and confirm
3. Your tokens will no longer expire after 7 days

## Example Usage with Reachy

Once configured, you can ask Reachy:

- "What's on my calendar today?"
- "Am I free tomorrow at 3pm?"
- "Schedule a meeting with John for Friday at 2pm"
- "What events do I have this week?"
- "Create an event called 'Team Standup' for tomorrow at 10am"
- "Check my availability for next Monday"

## Troubleshooting

### "OAuth Credentials File Not Found"
- Ensure credentials are saved to `~/.gcal-mcp/gcp-oauth.keys.json`
- Or set `GOOGLE_OAUTH_CREDENTIALS` environment variable

### "Authentication Errors"
- Ensure your credentials file is for a **Desktop App** type
- Verify your email is added as a **Test User** in OAuth consent screen
- Try deleting tokens and re-authenticating

### "User Rate Limit Exceeded"
- Ensure your `gcp-oauth.keys.json` file includes `project_id`
- Re-download credentials from Google Cloud Console

## References

- [google-calendar-mcp GitHub](https://github.com/nspady/google-calendar-mcp)
- [Google Calendar API Documentation](https://developers.google.com/calendar)
