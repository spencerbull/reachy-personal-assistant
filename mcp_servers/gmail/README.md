# Gmail MCP Integration

This directory documents the Gmail MCP integration for the Reachy Personal Assistant.

## Overview

We use the [`@gongrzhe/server-gmail-autoauth-mcp`](https://github.com/GongRzhe/Gmail-MCP-Server) npm package which provides Gmail functionality through the Model Context Protocol (MCP).

## Available Tools

| Tool | Description |
|------|-------------|
| `send_email` | Send emails with subject, body, recipients, and attachments |
| `read_email` | Read email content by ID with MIME handling |
| `search_emails` | Search using Gmail query syntax (from, subject, date, etc.) |
| `list_labels` | List all Gmail labels |
| `create_label` | Create new labels |
| `update_label` | Update existing labels |
| `delete_label` | Delete labels |

## Setup Instructions

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., "Reachy Gmail Integration")
3. Enable the Gmail API:
   - Navigate to **APIs & Services > Library**
   - Search for "Gmail API"
   - Click **Enable**

### Step 2: Configure OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. Choose **External** (or Internal if using Google Workspace)
3. Fill in the required fields:
   - App name: "Reachy Assistant"
   - User support email: Your email
   - Developer contact: Your email
4. Add scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
   - `https://www.googleapis.com/auth/gmail.compose`
   - `https://www.googleapis.com/auth/gmail.modify`
5. Add your email as a test user

### Step 3: Create OAuth Credentials

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth client ID**
3. Application type: **Desktop app**
4. Name: "Reachy Gmail"
5. Click **Create**
6. Download the JSON file

### Step 4: Set Up Authentication

```bash
# Create the Gmail MCP config directory
mkdir -p ~/.gmail-mcp

# Copy your downloaded credentials (rename to gcp-oauth.keys.json)
cp ~/Downloads/client_secret_*.json ~/.gmail-mcp/gcp-oauth.keys.json

# Run authentication (opens browser for Google sign-in)
npx @gongrzhe/server-gmail-autoauth-mcp auth
```

After successful authentication, credentials will be saved to `~/.gmail-mcp/credentials.json`.

### Step 5: Verify Integration

The Reachy agent will automatically detect Gmail MCP configuration on startup. Look for this log message:

```
Gmail MCP is configured and ready
```

## Usage Examples

Once configured, you can ask Reachy:

- "Check my email"
- "Do I have any unread emails?"
- "Read my latest email from John"
- "Send an email to john@example.com about the meeting tomorrow"
- "Search for emails about project updates"

## Troubleshooting

### "Gmail MCP not configured"

Run the authentication command:
```bash
npx @gongrzhe/server-gmail-autoauth-mcp auth
```

### OAuth errors

1. Ensure your OAuth credentials are in `~/.gmail-mcp/gcp-oauth.keys.json`
2. Verify the Gmail API is enabled in your Google Cloud project
3. Check that you've added yourself as a test user in the OAuth consent screen

### Token expired

Delete the old credentials and re-authenticate:
```bash
rm ~/.gmail-mcp/credentials.json
npx @gongrzhe/server-gmail-autoauth-mcp auth
```

## Security Notes

- OAuth credentials are stored locally in `~/.gmail-mcp/`
- Tokens auto-refresh using offline access
- Review and revoke access at: https://myaccount.google.com/permissions
- Never commit credentials to version control
