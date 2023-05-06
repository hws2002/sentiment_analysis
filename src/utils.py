#%% import modules
from collections import Counter
import numpy as np
import gensim
import time

#%% build Dictionary
def build_vocab(List : list) -> dict: 
    """
    word to index
    used for creating dataset
    """
    print("Starting to build Dictionary..."); start_time = time.time()
    
    Vocab = Counter()
    for file in List: # train.txt and validation.txt
        with open(file,"r",encoding="utf-8") as f:
            for line in f.readlines(): # 1	死囚 爱 刽子手 女贼 爱 衙役 我们 爱 你们 难道 还有 别的 选择 没想到 ...
                sentence = line.strip().split() # ['1', '死囚', '爱', '刽子手', '女贼', '爱', '衙役', ..
                for voca in sentence[1:]: # first word is label
                    if voca not in Vocab.keys():
                        Vocab[voca] = len(Vocab)
    end_time = time.time(); elapsed_time = end_time - start_time
    print("Building vocabulary ended! Elapsed time: {:.2f} seconds".format(elapsed_time))
    return Vocab

vocab =  build_vocab(["../Dataset/train.txt"])

#%%build sentence vector
n_exist = 0
def build_vector(List : list, Vocab : dict) -> np.ndarray: 
    """
    word to vector (index from Vocab)
    only used in model.py
    """
    print("Starting to build vector..."); start_time = time.time()
    
    global n_exist
    # loaded pre-trained model
    preModel = gensim.models.KeyedVectors.load_word2vec_format("../Dataset/wiki_word2vec_50.bin",binary=True)
    vector = np.array([np.zeros(preModel.vector_size)]*(len(Vocab)+1)) # +1 for padding (although there are already 0 row vector for words not in vocab)

    for voca in Vocab: # some words don't exist in preModel -> 
        try:
            vector[Vocab[voca]] = preModel[voca]
        except Exception as e:
            # TODO : Make a better way to handle this exception
            # - pre process the data? idk...
            n_exist += 1
            pass
            # print("An exception occurred: " + str(e))
            
    end_time = time.time(); elapsed_time = end_time - start_time
    print("Building vector ended! Elapsed time: {:.2f} seconds".format(elapsed_time))
    print("There are ",n_exist," words that doesn't exist in Model (include duplication)")
    print("Length of sentence vector is ",len(vector)) 
    return vector

s_vectors = build_vector(["../Dataset/train.txt"],vocab)

#%% parse data function to build dataset
def build_dataset(path : str,vocab : dict,max_length=50): # length of single sentence to use from data
    """
    returns contents and labels in numpy array
    from train, test, validation
    """
    print("Starting to parse data from ",path,"..."); start_time = time.time()
    
    words,labels = np.array([0]*max_length), np.array([],dtype=np.float64) # can't use integer here
    with open(path,encoding='utf-8',errors='ignore') as f:
        for line in f.readlines():
            sentence = line.strip().split() # ['1', '如果', '我', '无聊',...
            stripped_sentence = np.asarray([vocab.get(word,len(vocab)) for word in sentence [1:]])[:max_length] # strip only first max_length elements
                                    # index vector of sentence
                                    # if key doesn't exist in vocab, return 0
            # pad the content to match length
            padding = max(max_length - len(stripped_sentence), 0)
            stripped_sentence = np.pad(stripped_sentence, pad_width=(0, padding), mode='constant', constant_values=len(vocab))
            
            # append label, pos -> 1, neg ->0
            labels = np.append(labels, int(sentence[0])) 
            
            # append content
            words = np.vstack([words,stripped_sentence])
    # delete the first row of contents (to match its length with labels)
    words = np.delete(words,0,axis=0)
    
    end_time = time.time();elapsed_time = end_time - start_time
    print("Parsing data ended! Elapsed time: {:.2f} seconds-------------------------------".format(elapsed_time))
    return words, labels