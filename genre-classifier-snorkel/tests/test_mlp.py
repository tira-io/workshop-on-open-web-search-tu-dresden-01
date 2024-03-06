import unittest
from joblib import load
from src.__init__ import CLASSIFIER_PATH
from src.run_mlp import classify, label

vectorizer = load(CLASSIFIER_PATH / f"vectorizer_{'plain_text'}_{'mlp'}_{'english'}.joblib")
mlp_classifier = load(CLASSIFIER_PATH / f"classifier_{'plain_text'}_{'mlp'}_{'english'}.joblib")

class mlp_test(unittest.TestCase):
    def test_general(self):
        article = "this article is about the student life in Dresden. in summery the paper talkes about important things you should lern in your life"
        shop = "if you buy 2 sets of this item you can get 30 euro offer on the USA T-shirts"
        discussion = "forum and talk about the process. i would like to discuss the issue that we had yesterday"

        data = {'text': [article, shop, discussion], 'docno': [0,1,2]}

        res = classify(data)
        expected = [label.ARTICLES, label.SHOP, label.DISCUSSION]
        print(res)
        self.assertListEqual(list(res['predicted_labels']), expected)



