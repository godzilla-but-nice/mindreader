# Work with Python 3.6
import discord

TOKEN = 'NjYwNjEyODc3MDI3MTE1MDI1.XgfZ1w.9fwIg3831MheR3dDjBOfZXD7sz8'

client = discord.Client()

@client.event
async def on_message(message):
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    if message.content.startswith('!hello'):
        msg = 'Hello {0.author.mention}'.format(message)
        await client.send_message(message.channel, msg)

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(TOKEN)