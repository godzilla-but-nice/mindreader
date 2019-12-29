import discord
import json
import pickle
import pymongo
import numpy as np

# config
TOKEN = 'NjYwNzI2MzA4Mzg0NDA3NTgz.Xgj-AA._ZpocAc1iNRbN4C9gKvVb0p-5jM'
THRESHOLD = 0.7
MIN_CHARS = 15
EMOTIONS = {0: 'neg', 2: 'neu', 1: 'pos'}  # TODO: make sure keys are correct


def init_server_db(server, servers):
    if servers.find_one({'server_id': server.id}) != None:
        return
    servers.insert_one({
        'server_id': server.id,
        'name': server.name.lower(),
        'reactions': {
            'pos': '😀',
            'neg': '☹️',
            'neu': '😐'
        }
    })


def predict_sentiment(classifier, class_dict, input):
    probs = classifier.predict_proba([input])
    sent_i = np.argmax(probs)
    confidence = np.max(probs)
    sentiment = class_dict[sent_i]

    return (sentiment, confidence)


def tokenizer(text):
    return text.split()


# db stuff
mongo_client = pymongo.MongoClient(
    'mongodb+srv://mindreader_bot:3VihUx5OHnMHO3xP@mindreader-fzoou.mongodb.net/test?retryWrites=true&w=majority')
db = mongo_client.test
servers = db.servers

# discord
client = discord.Client()

# load classifier
classifier = pickle.load(open('classifier.pkl', 'rb'))


@client.event
async def on_message(message):
    global EMOTIONS, THRESHOLD, MIN_CHARS
    server = message.server.name.lower()
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
                {"name": server},
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
                raise Exception('Invalid sentiment.')
            doc = servers.find_one({"name": server})
            reaction = servers.find_one(
                {"name": server})['reactions'][parsed_mes[1]]
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
            classifier, EMOTIONS, message.content)
        print('Sentiment:', emotion, "Proba:", proba)
        if proba > THRESHOLD:
            reaction = servers.find_one({'name': server})['reactions'][emotion]
            await client.add_reaction(message, reaction)

@client.event
async def on_server_join(server):
    init_server_db(server, servers)


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')
    print('Servers connected to:')
    for server in client.servers:
        print(server.id)

    # debug and tests
    db_cursor = servers.find()
    for item in db_cursor:
        print(item)
    print(servers.find_one({"name": "phantasmalpup"})['reactions']['pos'])


client.run(TOKEN)
client.close()
