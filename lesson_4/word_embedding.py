from gensim.models import Word2Vec

# Sample corpus (list of sentences, each split into words)
sentences = [
    ["king", "queen", "man", "woman"],
    ["paris", "france", "berlin", "germany"],
    ["apple", "banana", "fruit"],
    ["cat", "dog", "pet", "animal"],
]


# Train 
model = Word2Vec(sentences,vector_size=10, window=2,min_count=1,workers=1)

print("Vector for 'king': \n",model.wv['king'])

# Find most similar words to 'king'
print("Most similar to 'king'\n: ",model.wv.most_similar('king'))