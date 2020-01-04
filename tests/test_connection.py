import pytest
from core import bot


def test_connection():
    bot.startup()
    return bot.client.is_logged_in


def test_close():
    bot.client.close()
    return bot.client.is_closed
