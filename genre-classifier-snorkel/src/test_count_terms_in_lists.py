import unittest
from src.util import extract_overlapping_terms

class TestCountTermsInLists(unittest.TestCase):
    def test_discussion_01(self):
        text = 'The title of my messages is always hello.'
        terms_in_discussion_list = ['title', 'messages']
        actual = extract_overlapping_terms(text, 'discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)

    def test_discussion_02(self):
        text = 'my emails, were,posted.'
        terms_in_discussion_list = ['emails', 'posted']
        actual = extract_overlapping_terms(text, 'discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)

    def test_discussion_03(self):
        text = 'The title of my messages is always hello.'
        terms_in_discussion_list = ['messages']
        actual = extract_overlapping_terms(text, 'vocabulary-popescul-modified-discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)

    def test_discussion_04(self):
        text = 'my mails, were,posted.'
        terms_in_discussion_list = ['mails', 'posted']
        actual = extract_overlapping_terms(text, 'vocabulary-popescul-modified-discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)