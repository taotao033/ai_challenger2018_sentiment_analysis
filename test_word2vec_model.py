from gensim.models import  Word2Vec
import pandas as pd
model = Word2Vec.load('./word2vec.model')
#print(model.wv['热情'])
word_embedding_matrix = pd.read_pickle('./gensim_data_word_embedding_matrix_new.pkl')
print(word_embedding_matrix.ndim)