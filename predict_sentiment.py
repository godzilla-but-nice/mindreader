import numpy as np

def predict_sentiment(classifier, class_dict, input):
    probs = classifier.predict_proba([input])
    sent_i = np.argmax(probs)
    confidence = np.max(probs)
    sentiment = class_dict[sent_i]

    return (sentiment, confidence)
