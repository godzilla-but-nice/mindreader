import discord
import json
import pickle
import pymongo
import numpy as np
import os
from online_classifier import train_classifier, predict_sentiment

# config
TOKEN = os.environ['TOKEN']
THRESHOLD = 0.81
MIN_CHARS = 15
EMOTIONS = {0: 'neg', 2: 'neu', 1: 'pos'}  # TODO: make sure keys are correct
TRAINING_DATA = 'training_data/smaller_tweets.csv'
PICKLE_VECT = 'pickle_objects/vectorizer.pkl'
PICKLE_CLF = 'pickle_objects/classifier.pkl'


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



@client.event
async def on_message(message):
    global EMOTIONS, THRESHOLD, MIN_CHARS
    server_id = message.server.id
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    # change emojis
    if message.content.startswith('!change'):
        # need admin permissions
        if message.author.server_permissions.administrator != True:
            msg = '{0.author.mention}, you do not have permission to issue this command.'
            msg = msg.format(message)
            await client.send_message(message.channel, msg)
            return

        parsed_mes = message.content.split()
        try:
            if parsed_mes[1] not in EMOTIONS.values():
                raise Exception('Invalid sentiment.')
            await client.add_reaction(message, parsed_mes[2])
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
            await client.send_message(message.channel, msg)
            return

    # emoji test
    if message.content.startswith('!test'):
        parsed_mes = message.content.split()
        try:
            if parsed_mes[1] not in EMOTIONS.values():
                doc = servers.find_one({'server_id': server_id})
            reaction = servers.find_one(
                {'server_id': server_id})['reactions'][parsed_mes[1]]
            await client.add_reaction(message, reaction)
            return

        except Exception as e:
            msg = '{0.author.mention} ' + str(e)
            msg = msg.format(message)
            await client.send_message(message.channel, msg)
            return

    # predict sentiment and react
    if len(message.content) >= MIN_CHARS:
        emotion, proba = predict_sentiment(
            clf, vect, EMOTIONS, message.content)
        print('Sentiment:', emotion, 'Proba:', proba)
        if proba > THRESHOLD:
            reaction = servers.find_one({'server_id': server_id})[
                'reactions'][emotion]
            await client.add_reaction(message, reaction)


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
            await client.send_message(channel, msg)
            break


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')
    print('Servers connected:')
    for server in client.servers:
        if servers.find_one({'server_id' : server.id}) == None:
            await init_server_db(server, servers)
        print(server.id, ':', server.name)


client.run(TOKEN)
client.close()
