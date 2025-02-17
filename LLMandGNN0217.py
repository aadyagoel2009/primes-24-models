from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-cos-v1") #all-MiniLM-L6-v2

# Two lists of sentences
# sentences1 = [
#     "The new movie is awesome",
#     "The cat sits outside",
#     "A man is playing guitar",
# ]

# sentences2 = [
#     "The dog plays in the garden",
#     "The new movie is so great",
#     "A woman watches TV",
# ]

sentences1 = [
    # "doctor",
    # "nurse",
    # "engineer",
    # "scientist",
    # "housekeeper",
    # "manager"
    "Kate is a",
    "Mike is a",
    "My housekeeper is a",
    "Their father is a",
    "Their mother is a",
    "The sister is a",
    "The brother is a",
]

sentences2 = [
    "man",
    "woman",
]

# Compute embeddings for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for idx_i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for idx_j, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
