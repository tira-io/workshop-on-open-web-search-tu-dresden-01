import unittest
from src.util import extract_overlapping_terms, preprocess_document

class TestCountTermsInLists(unittest.TestCase):
    def test_discussion_01(self):
        text = preprocess_document('The of my messages is always hello.')
        terms_in_discussion_list = ['messag']
        actual = extract_overlapping_terms(text, 'tokens_with_count_75', 'vocabulary-popescul-modified-discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)

    def test_discussion_02(self):
        text = preprocess_document('my mails, were,posted.')
        terms_in_discussion_list = ['mail', 'post']
        actual = extract_overlapping_terms(text, 'tokens_with_count_75', 'vocabulary-popescul-modified-discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)

    def test_discussion_03(self):
        text = preprocess_document('The title of my messages is always hello.')
        terms_in_discussion_list = ['messag']
        actual = extract_overlapping_terms(text, 'tokens_with_count_75', 'vocabulary-popescul-modified-discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)

    def test_discussion_04(self):
        text = preprocess_document('my mails, were,posted.')
        terms_in_discussion_list = ['mail', 'post']
        actual = extract_overlapping_terms(text, 'tokens_with_count_75', 'vocabulary-popescul-modified-discussion.txt')
        self.assertEquals(terms_in_discussion_list, actual)


    def test_download(self):
        text = preprocess_document('download something on windows.')
        terms_in_download_list = ['download', 'window']
        actual = extract_overlapping_terms(text, 'tokens_with_count_75', 'vocabulary-popescul-modified-download.txt')
        self.assertEquals(terms_in_download_list, actual)

    def test_article(self):
        text = preprocess_document('something is wrong with the system')
        term_in_article = ['system']
        actual = extract_overlapping_terms(text, 'tokens_with_count_75', 'vocabulary-popescul-modified-articles.txt')
        self.assertEquals(term_in_article, actual)