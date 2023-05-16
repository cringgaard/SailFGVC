from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm

class Labeller:
    def __init__(self, labels , threshold = 0.8):
        self.labels = [label.lower() for label in labels]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(labels)
        self.embedded_labels = self.vectorizer.transform(labels).toarray()
    
    def cosine_similarity(self, a, b):
        return (np.dot(a,b)/(norm(a)*norm(b)))

    def add_label(self, text):
        boat_name = self.vectorizer.transform(text).toarray()
        similarities = [self.cosine_similarity(boat_name, label) for label in self.embedded_labels]
        closest = np.argmax(similarities)
        if similarities[closest] > 0.8:
            return True , closest , similarities[closest]
        else:
            return False , None, None
