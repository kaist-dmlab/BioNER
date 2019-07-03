import os
import unittest
import sys
from dataset import PreprocessingPOS 
                 
class TestPreprocessingPOS(unittest.TestCase):
    def setUp(self):
        self.pos_processor = PreprocessingPOS()

    def test_AnnotatingPosTagsGivenSimpleSentence(self):
        given_sentence = ['I', 'am', 'a', 'student', '.']
        pos_sentence = self.pos_processor.annotate_pos(given_sentence)
        self.assertEqual(pos_sentence, ['PRON', 'VERB', 'DET', 'NOUN', 'PUNCT'])

    def test_GetTagsViaClassMethod(self):
        pos_tags = PreprocessingPOS.get_pos_tags()
        self.assertEqual(len(pos_tags), 19)

    def test_GetTagsLengthViaClassMethod(self):
        self.assertEqual(len(self.pos_processor), 19)
        
    def test_PosDictionary(self):
        pos_tags = PreprocessingPOS.get_pos_tags()
        self.assertEqual(self.pos_processor.pos_dict[pos_tags[0]], 0)
        self.assertEqual(self.pos_processor.pos_dict[pos_tags[2]], 2)

    def test_ConvertedPosTags(self):
        given_sentence = ['I', 'am', 'a', 'student', '.']
        target_pos_tags = ['PRON', 'VERB', 'DET', 'NOUN', 'PUNCT']
        pos_sentence = self.pos_processor(given_sentence)
        converted_pos_tags = list(map(lambda tag: self.pos_processor.pos_dict[tag], target_pos_tags))
        self.assertEqual(pos_sentence, converted_pos_tags)
    
    def test_UnknownPosTag(self):
        self.assertEqual(self.pos_processor.pos_dict['X'], 10)
                
    def tearDown(self):
        pass
