import pytest
from core.online_classifier import predict_sentiment
from core.bot import clf, vect, EMOTIONS

@pytest.mark.parametrize("test_input,expected",
                          [("happy happy happy good good", "pos"),
                           ("sad sad sad bad bad", "neg")])
def test_extreme_prediction(test_input, expected):
    emotion, _ = predict_sentiment(clf, vect, EMOTIONS, test_input)
    assert emotion == expected
