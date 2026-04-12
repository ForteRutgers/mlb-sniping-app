# MLB Sniping Discord Bot

Control your MLB sniping workflows directly from Discord!

## Commands

| Command | Description |
|---------|-------------|
| `!run daily` | Run daily predictions only |
| `!run bootstrap` | Run full data bootstrap (30+ min) |
| `!run training` | Retrain ML models |
| `!run full` | Bootstrap + training |
| `!status` | Check latest workflow runs |
| `!help` | Show available commands |

## Setup Guide

### Step 1: Create a Discord Application & Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **"New Application"** and give it a name (e.g. `MLB Sniping Bot`)
3. Click **"Create"**
4. Go to the **"Bot"** tab on the left sidebar
5. Click **"Add Bot"** → **"Yes, do it!"**
6. Under **"Privileged Gateway Intents"**, enable **"Message Content Intent"**
7. Click **"Save Changes"**
8. Click **"Reset Token"** → copy and save this token (this is your `DISCORD_BOT_TOKEN`)

### Step 2: Invite the Bot to Your Server

1. Go to the **"OAuth2"** tab → **"URL Generator"**
2. Under **"Scopes"**, check `bot`
3. Under **"Bot Permissions"**, check `Send Messages` and `Read Messages/View Channels`
4. Copy the generated URL and open it in your browser
5. Select your Discord server and click **"Authorize"**

### Step 3: Create a GitHub Personal Access Token

1. Go to [GitHub Settings → Developer Settings → Personal access tokens → Tokens (classic)](https://github.com/settings/tokens)
2. Click **"Generate new token (classic)"**
3. Give it a name (e.g. `MLB Discord Bot`)
4. Select the **`repo`** scope (this allows triggering workflows)
5. Click **"Generate token"** → copy and save this token (this is your `GITHUB_TOKEN`)

### Step 4: Deploy to Render

1. Go to [Render](https://render.com) and log in
2. Click **"New +"** → **"Web Service"** (or **"Background Worker"**)
3. Connect your GitHub account and select the `mlb-sniping-app` repository
4. Configure the service:
   - **Root Directory**: `discord_bot`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python bot.py`
5. Scroll to **"Environment Variables"** and add:
   - `DISCORD_BOT_TOKEN` → paste your Discord bot token
   - `GITHUB_TOKEN` → paste your GitHub personal access token
6. Click **"Create Web Service"**

Render will build and deploy your bot automatically. Once deployed, it will appear online in your Discord server.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DISCORD_BOT_TOKEN` | Your Discord bot token from the Developer Portal |
| `GITHUB_TOKEN` | GitHub Personal Access Token with `repo` scope |

## How It Works

The bot listens for commands in any channel it has access to. When you type `!run daily`, it calls the GitHub Actions API to trigger the `mlb_engine.yml` workflow with the appropriate inputs. The `!status` command fetches the latest workflow runs from GitHub and displays them in Discord.
