import tensorflow as tf
import numpy as np
import re

class DataSet():
    def __init__(self, new_code=None):
        
        # user's customizing data
        self.new_code = new_code
        
        # built-in dataset
        self.seq_data = ['tuple', 'list', 'dic', 'return', 'print', 
                         'for', 'range', 'while', 'not', 'is', 'sort']
        
        # the longest word's length among dataset
        self.max_word_len = max([len(word) for word in self.seq_data])
        
        # update built-in dataset with given data from a user
        self.add_new_code2seq_data()
    
    # preprocess new codes(new data)
    def clean_new_code(self):
        if self.new_code:
            assert type(self.new_code) == str
            new_code_ls = re.sub('[ "\']+', ' ', re.sub('[\t]', ' ', self.new_code)).split('\n')
            new_code_ls = [word for sent in new_code_ls for word in sent.split(' ')]
            new_code_ = re.sub('[ ]+', ' ', re.sub('[\W]', ' ', self.new_code))
            new_code_ls2 = new_code_.split(' ')
            new_code_ls2 = [word for sent in new_code_ls2 for word in sent.split(' ')]
            new_code_ls.extend(new_code_ls2)
            return new_code_ls
        else:
            print('There is no new code to preprocess!')
            return None
    
    def add_new_code2seq_data(self):
        new_code_ls = self.clean_new_code()
        if new_code_ls:
            assert type(new_code_ls) == list
            self.seq_data.extend(new_code_ls)
            self.max_word_len = max([len(word) for word in self.seq_data])
    
    # make word dictionary
    def make_word2idx_idx2word(self, get_vocab_size=True):
        """
        return : (word2idx(dic type), idx2word(dic type), vocab_size(int type))
        """
        # base characters
        words_arr = set("")
        for i in range(len(self.seq_data)):
            x = [self.seq_data[i][:j] for j in range(1, len(self.seq_data[i]))]
            y = [self.seq_data[i][j:] for j in range(1, len(self.seq_data[i]))]
            words_arr.update(set(list(self.seq_data[i])+ x + y))

        word2idx = {word: i for i, word in enumerate(words_arr)}
        idx2word = {i: word for i, word in enumerate(words_arr)}
        
        # add '[UKN]' token in the both word dictionaries
        word2idx['[UKN]'] = len(word2idx)
        idx2word[word2idx['[UKN]']] = ' '
        
        if get_vocab_size:
            vocab_size = len(word2idx)
            return word2idx, idx2word, vocab_size
        else:
            return word2idx, idx2word
        
    def get_dataset(self):
        """
        return : x, y, max word length, vocabulary size
        x shape : (dataset length, max word length)
        y shape : (dataset length, length of word dictionary)
        """
        
        word2idx, idx2word, vocab_size = self.make_word2idx_idx2word()
        
        inputs = list(list(list(self.seq_data[i][:j])for j in range(1, len(self.seq_data[i]))) for i in range(len(self.seq_data)))
        outputs = list(list(self.seq_data[i][j:] for j in range(1, len(self.seq_data[i]))) for i in range(len(self.seq_data)))
        inputs_vec = [[word2idx[char_ls[i]] for i in range(len(char_ls))] for word_ls in inputs for char_ls in word_ls]
        x = tf.keras.preprocessing.sequence.pad_sequences(inputs_vec, maxlen=self.max_word_len, padding='pre')
        y = np.array([word2idx[word] for word_ls in outputs for word in word_ls])
        y = np.eye(vocab_size)[y]
        return x, y, self.max_word_len, vocab_size
    