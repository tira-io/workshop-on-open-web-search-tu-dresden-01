import unittest
from src.snorkel_genre_classifier import process_documents
from approvaltests import verify_as_json

def doc(doc_id, default_text):
    class MockedDocument:
        def __init__(self, doc_id, text):
            self.doc_id = doc_id
            self.text = text

        def default_text(self):
            return self.text
        
    return MockedDocument(doc_id, default_text)


class SnorkelTest(unittest.TestCase):
    def test_super_strict(self):
        # a super strict test :)
        self.assertEqual(1, 1)

    def test_with_dummy_documents(self):
        document_iter = [doc(1, 'discussion'), doc(2, 'quantity')]
        actual = process_documents(document_iter)

        verify_as_json([i.to_dict() for _, i in actual.iterrows()])
