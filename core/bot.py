import discord
import json
import pickle
import pymongo
import numpy as np
import os
import random
import requests
from core.online_classifier import train_classifier, predict_sentiment

# config
TOKEN = os.environ['TOKEN']
THRESHOLD = 0.81
MIN_CHARS = 15
EMOTIONS = {0: 'neg', 2: 'neu', 1: 'pos'}  # TODO: make sure keys are correct
TRAINING_DATA = os.path.abspath('core/training_data/smaller_tweets.csv')
PICKLE_VECT = os.path.abspath('core/pkl_objects/vectorizer.pkl')
PICKLE_CLF = os.path.abspath('core/pkl_objects/classifier.pkl')

# insults
INSULT_ENDPOINT = 'https://insult.mattbas.org/api/insult'

def get_insult(who='Noah'):
    payload = {'who': who}
    resp = requests.get(INSULT_ENDPOINT, params=payload)
    return resp.text



async def init_server_db(server, servers):  # TODO: enforce schema
    if servers.find_one({'server_id': server.id}) != None:
        return
    servers.insert_one({
        'server_id': server.id,
        'server_name': server.name,
        'reactions': {
            'pos': '😀',
            'neg': '☹️',
            'neu': '😐'
        }
    })


# db stuff
mongo_client = pymongo.MongoClient(
    'mongodb+srv://' + os.environ['DB_USERNAME'] + ':' + os.environ['DB_PASS'] +
    '@mindreader-fzoou.mongodb.net/test?retryWrites=true&w=majority')
db = mongo_client.test
servers = db.servers

# discord
client = discord.Client()

# load classifier
if os.path.exists(PICKLE_VECT) and os.path.exists(PICKLE_CLF):
    print('Classifier and vectorizer found!')
    vect = pickle.load(open(PICKLE_VECT, 'rb'))
    clf = pickle.load(open(PICKLE_CLF, 'rb'))
else:
    print('Saved Classifier not found! Training classifier')
    clf, vect = train_classifier(TRAINING_DATA)

# random game


def randomItem(ls):
    return ls[random.randint(0, len(ls) - 1)]


@client.event
async def on_message(message):
    global EMOTIONS, THRESHOLD, MIN_CHARS
    server_id = message.guild.id
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    # noah history
    if message.content.startswith('!deleted'):
        member = message.mentions[0]
        msg = (f"Messages deleted by {member.mention}:"
                "> I am very smelly and once pooped up the back of a toilet.\n"
                "> I am plotting a mutiny to overthrow the captain of our sub. This is the beginning of a new era on Europa.\n"
                "> Andrew hates Dune and I hate him. Terrible taste in film.\n")
        msg.format(message)
        await message.channel.send(msg)
        return

    # noah insult
    if client.user.mentioned_in(message):
        if message.content.lower().contains('noah'):
            insult = get_insult('Noah')
            await message.channel.send(insult)
    
    # dav bowling
    if message.content.startswith('!dav'):
        msg = "Don't worry, @GalaxyQuest said he's going bowling"
        msg.format(message)
        await message.channel.send(msg)
        return

    # pick a game (or item from list)
    if message.content.startswith('!pick '):
        message.content = message.content[6:].strip()
        ls = message.content.split(',')
        for str in ls:
            str = str.strip()
        msg = '{0.author.mention} ' + randomItem(ls)
        msg = msg.format(message)
        await message.channel.send(msg)

    # change emojis
    if message.content.startswith('!change'):
        # need admin permissions
        if message.author.guild_permissions.administrator != True:
            msg = '{0.author.mention}, you do not have permission to issue this command.'
            msg = msg.format(message)
            await client.send_message(message.channel, msg)
            return

        parsed_mes = message.content.split()
        try:
            if parsed_mes[1] not in EMOTIONS.values():
                raise Exception('Invalid sentiment.')
            await message.add_reaction(parsed_mes[2])
            servers.update_one(
                {'server_id': server_id},
                {'$set': {
                    'reactions.' + parsed_mes[1]: parsed_mes[2]
                }
                })
            return

        except Exception as e:
            msg = '{0.author.mention} ' + str(e)
            msg = msg.format(message)
            await message.channel.send_message(msg)
            return

    # emoji test
    if message.content.startswith('!test'):
        parsed_mes = message.content.split()
        try:
            if parsed_mes[1] not in EMOTIONS.values():
                doc = servers.find_one({'server_id': server_id})
            reaction = servers.find_one(
                {'server_id': server_id})['reactions'][parsed_mes[1]]
            await message.add_reaction(reaction)
            return

        except Exception as e:
            msg = '{0.author.mention} ' + str(e)
            msg = msg.format(message)
            await message.channel.send_message(msg)
            return

    # predict sentiment and react
    if len(message.content) >= MIN_CHARS:
        emotion, proba = predict_sentiment(
            clf, vect, EMOTIONS, message.content)
        print('Sentiment:', emotion, 'Proba:', proba)
        if proba > THRESHOLD:
            reaction = servers.find_one({'server_id': server_id})[
                'reactions'][emotion]
            await message.add_reaction(reaction)


@client.event
async def on_server_join(server):
    print('Connecting to new server:')
    print(server.id, ':', server.name)
    await init_server_db(server, servers)
    msg = ('>>> **Thank you for inviting me to your server!**'
           '\nBeep Boop.'
           '\n\nCommands:'
           '\n```!change <pos, neg, or neu> <emoji>``` '
           '- Changes reaction to positive, negative, or neutral comments'
           '\n  (need to be admin)'
           '\n```!test <pos, neg, neu>```-  Tests reactions.'
           '\n\nNote: this bot and its sentiment analysis are still under'
           ' development. More "emotions" will be added. Expect wonky behavior...')
    for channel in server.channels:
        if str(channel.type) == 'text':
            await channel.send_message(msg)
            break


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')
    print('Servers connected:')
    for server in client.guilds:
        if servers.find_one({'server_id': server.id}) == None:
            await init_server_db(server, servers)
        print(server.id, ':', server.name)


async def startup():
    await client.run(TOKEN)

if __name__ == '__main__':
    client.run(TOKEN)
