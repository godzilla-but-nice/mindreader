import discord
import json
import pickle
import numpy as np

TOKEN = 'NjYwNjEyODc3MDI3MTE1MDI1.XggBgQ.jBjDSD1LitWqSGqHZQQMqqqegCQ'


def predict_sentiment(classifier, class_dict, input):
    probs = classifier.predict_proba([input])
    sent_i = np.argmax(probs)
    confidence = np.max(probs)
    sentiment = class_dict[sent_i]

    return (sentiment, confidence)


def tokenizer(text):
    return text.split()


# load reactions
data = open('reactions.json', 'r')
reactions = json.load(data)
data.close()
client = discord.Client()

# load classifier
classifier = pickle.load(open('classifier.pkl', 'rb'))
emotions = {0: 'neg', 1: 'pos'}


@client.event
async def on_message(message):
    global reactions, emotions
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    # change emojis
    if message.content.startswith('!change'):
        if message.author.server_permissions.administrator != True:
            msg = '{0.author.mention}, you do not have permission to issue this command.'
            msg = msg.format(message)
            await client.send_message(message.channel, msg)
            return

        parsed_mes = message.content.split()
        try:
            if parsed_mes[1] not in reactions.keys():
                raise Exception('Invalid sentiment.')
            await client.add_reaction(message, parsed_mes[2])
            reactions[parsed_mes[1]] = parsed_mes[2]
            data = open('reactions.json', 'w')
            json.dump(reactions, data, indent=4)
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
            if parsed_mes[1] not in reactions.keys():
                raise Exception('Invalid sentiment.')

        except Exception as e:
            msg = '{0.author.mention} ' + str(e)
            msg = msg.format(message)
            await client.send_message(message.channel, msg)
            return

        await client.add_reaction(message, reactions[parsed_mes[1]])
        return

    # predict sentiment and react
    if len(message.content) > 15:
        sentiment, proba = predict_sentiment(
            classifier, emotions, message.content)
        print('Sentiment:', sentiment, "Proba:", proba)
        if proba > 0.7:
            await client.add_reaction(message, reactions[sentiment])


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(TOKEN)
client.close()
