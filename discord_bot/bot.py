import discord
from discord.ext import commands
import requests
import os
from datetime import datetime

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Configuration from environment variables
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO_OWNER = os.environ.get('REPO_OWNER', 'ForteRutgers')
REPO_NAME = os.environ.get('REPO_NAME', 'mlb-sniping-app')
WORKFLOW_FILE = os.environ.get('WORKFLOW_FILE', 'mlb_engine.yml')

@bot.event
async def on_ready():
    print(f'🤖 {bot.user} is online and ready!')

@bot.command(name='help')
async def help_command(ctx):
    """Show available commands"""
    help_text = """
🤖 **MLB Sniping Bot Commands**

`!run daily` - Run daily predictions only
`!run bootstrap` - Run full data bootstrap (30+ min)
`!run training` - Retrain ML models with latest data
`!run full` - Run bootstrap + training
`!status` - Check latest workflow status
`!help` - Show this message
    """
    await ctx.send(help_text)

@bot.command(name='run')
async def run_workflow(ctx, workflow_type: str = 'daily'):
    """Trigger MLB workflow"""
    
    workflow_configs = {
        'daily': {'run_bootstrap': 'false', 'run_training': 'false'},
        'bootstrap': {'run_bootstrap': 'true', 'run_training': 'false'},
        'training': {'run_bootstrap': 'false', 'run_training': 'true'},
        'full': {'run_bootstrap': 'true', 'run_training': 'true'}
    }
    
    if workflow_type not in workflow_configs:
        await ctx.send(f"❌ Unknown workflow type: `{workflow_type}`\nUse: `!run daily`, `!run bootstrap`, `!run training`, or `!run full`")
        return
    
    if not GITHUB_TOKEN:
        await ctx.send("❌ GitHub token not configured!")
        return
    
    await ctx.send(f"⏳ Triggering **{workflow_type}** workflow...")
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    payload = {
        'ref': 'main',
        'inputs': workflow_configs[workflow_type]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 204:
            await ctx.send(f"✅ **{workflow_type.upper()}** workflow triggered successfully!\n🔗 Check progress: https://github.com/{REPO_OWNER}/{REPO_NAME}/actions")
        elif response.status_code == 401:
            await ctx.send("❌ Authentication failed. Check your GitHub token.")
        elif response.status_code == 404:
            await ctx.send("❌ Workflow not found. Check the workflow file name.")
        else:
            await ctx.send(f"❌ Failed to trigger workflow. Status: {response.status_code}\n```{response.text}```")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name='status')
async def check_status(ctx):
    """Check the latest workflow run status"""
    
    if not GITHUB_TOKEN:
        await ctx.send("❌ GitHub token not configured!")
        return
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs?per_page=5"
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            runs = response.json().get('workflow_runs', [])
            
            if not runs:
                await ctx.send("📊 No workflow runs found.")
                return
            
            status_message = "📊 **Recent Workflow Runs**\n\n"
            
            for run in runs[:3]:  # Show last 3 runs
                status = run['status']
                conclusion = run.get('conclusion') or 'in progress'
                created = run['created_at'][:10]  # Just the date
                
                # Status emoji
                if conclusion == 'success':
                    emoji = '✅'
                elif conclusion == 'failure':
                    emoji = '❌'
                elif status == 'in_progress':
                    emoji = '🔄'
                else:
                    emoji = '⏸️'
                
                status_message += f"{emoji} **{created}** - {status} ({conclusion})\n"
            
            latest = runs[0]
            status_message += f"\n🔗 [View Latest Run]({latest['html_url']})"
            
            await ctx.send(status_message)
        else:
            await ctx.send(f"❌ Failed to fetch status. Status: {response.status_code}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

# Run the bot
if __name__ == '__main__':
    token = os.environ.get('DISCORD_BOT_TOKEN')
    if not token:
        print("❌ DISCORD_BOT_TOKEN not set!")
    else:
        bot.run(token)
