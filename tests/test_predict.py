from core import bot, online_classifier

bot.startup()

def func(x):
    return x + 1

def test_answer():
    assert func(1) == 2
