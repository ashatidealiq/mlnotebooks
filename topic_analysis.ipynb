#!/usr/bin/env python
# coding: utf-8

# # get all the pages from cms

# # content details per page

# In[ ]:


# turn the pages into documents with the text on the page


documents = []

for page in all_pages:
    

    doc = []
    doc.append(page['attributes'].get('title'))
    if page['attributes'].get('primary_topic'):
        doc.append(page['attributes'].get('primary_topic'))
    if page['attributes'].get('excerpt'):
        doc.append(page['attributes'].get('excerpt'))
    if page['attributes'].get('subtitle'):
        doc.append(page['attributes'].get('subtitle'))
    if page['attributes'].get('body_introduction'):
        doc.append(page['attributes'].get('body_introduction'))

    page_details = {
        "url": page['attributes']['slug'],
        "documents": ". ".join(doc)
    }
    documents.append(page_details)


# In[ ]:


documents_df = pd.DataFrame.from_dict(documents)

orig_len = len(documents_df.index)
documents_df


# # similarity using tfidf

# In[ ]:


import scipy.sparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# removing special characters and stop words from the text
stop_words_l=stopwords.words('english')
documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )

tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit(documents_df.documents_cleaned)
tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)

# add the average
documents_df.loc[orig_len] = ['avg', "",""]
if orig_len == tfidf_vectors.shape[0]:
    tfidf_vectors = scipy.sparse.vstack([tfidf_vectors, scipy.sparse.csr_matrix(tfidf_vectors.sum(axis=0)/orig_len)])

pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
pairwise_differences=euclidean_distances(tfidf_vectors)


# In[ ]:


def most_similar(doc_id,similarity_matrix,matrix):
    print()
    print('MATRIX:', matrix)
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    print (f'Document: {documents_df.iloc[doc_id].get("url",documents_df.iloc[doc_id]["documents"])}')
    print ('Similar Documents:')
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print (f'Document: {documents_df.iloc[ix].get("url",documents_df.iloc[ix]["documents"])}')
        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')

most_similar(orig_len,pairwise_similarities,'Cosine Similarity')        


# In[ ]:


pairwise_differences[orig_len]


# In[ ]:





# In[ ]:


slug = "how-to-do-a-proper-crunch-exercise"
orig_len = (documents_df[documents_df["url"]==slug].index)[0]

most_similar(orig_len,pairwise_differences,'Euclidean Distance') 


# # Difference using word tokens

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import scipy.sparse


# In[ ]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(documents_df.documents_cleaned)
tokenized_documents=tokenizer.texts_to_sequences(documents_df.documents_cleaned)
tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=64,padding='post')
vocab_size=len(tokenizer.word_index)+1

# reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
embeddings_index = dict()

with open('/data/glove/glove.6B.200d.txt') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    
# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
embedding_matrix=np.zeros((vocab_size,200))

for word,i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# calculating average of word vectors of a document weighted by tf-idf
document_embeddings=np.zeros((len(tokenized_paded_documents),200))
words=tfidfvectoriser.get_feature_names()

# add average to the end beginning
n_documents = documents_df.shape[0]

# instead of creating document-word embeddings, directly creating document embeddings
for i in range(n_documents):
    for j in range(len(words)):
        document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i].toarray()[0][j]

pairwise_similarities=cosine_similarity(document_embeddings)
pairwise_differences=euclidean_distances(document_embeddings)


# # distance to average
# 
# code from 
# https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630

# In[ ]:


most_similar(orig_len,pairwise_similarities,'Cosine Similarity')


# In[ ]:


most_similar(orig_len,pairwise_differences,'Euclidean Distance')


# In[ ]:


orig_len


# # using transformers

# In[ ]:


from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# In[ ]:


def get_bert_embedding(text):
    # Tokenize text
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Pass tokens through the BERT model
    with torch.no_grad():
        output = model(**tokens)

    # Extract embeddings from the output
    embeddings = output.last_hidden_state.mean(dim=1)  # You can use mean pooling for sentence embeddings

    return embeddings[0]


# In[ ]:


import pandas as pd

# Assuming you have a DataFrame with columns 'url' and 'document_contents'
# df = ...

# Create a new column 'embeddings' to store BERT embeddings
documents_df['embeddings'] = documents_df['documents'].apply(get_bert_embedding)


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract embeddings as a numpy array
embeddings_array = np.stack(documents_df['embeddings'].to_numpy())

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings_array, embeddings_array)


# In[ ]:


def get_top_similar_documents(url, similarity_matrix, df, top_n=10):
    # Find the index of the given URL in the DataFrame
    index = df[df['url'] == url].index[0]

    # Get the similarity scores for the given URL
    similarity_scores = similarity_matrix[index]

    # Sort the indices based on similarity scores (descending order)
    similar_indices = similarity_scores.argsort()[::-1]

    # Get the top N similar documents (excluding the input document itself)
    top_similar_documents = pd.DataFrame()
    top_similar_documents['url'] = df['url'].loc[similar_indices[1:top_n + 1]]
    top_similar_documents['similarity'] = similarity_matrix[index][similar_indices[1:top_n + 1]]
    return top_similar_documents

# Example usage:
given_url = "delicious-recipes-featuring-arugula"
top_similar_documents = get_top_similar_documents(given_url, similarity_matrix, documents_df)
print(top_similar_documents)


# In[ ]:


# get or set the page type for every page

slug_to_page_type = {}
slug_to_page = {}


# In[ ]:


# lookup table

all_slug_ids = {}
for this in all_pages:
    slug = this['attributes']['slug']
    this_id = this['id']
    all_slug_ids[slug] = this_id


# In[ ]:


for this_page in all_pages:
    
    slug = this_page['attributes']['slug']
    res = cms.get_by_slug("pages",slug)

    page = res['data']['attributes']

    page_type = None

    if len(page['body'])==0 and len(page['children_pages']['data'])>0:
        page_type = "category"
    if len(page['body']):
        page_type = "article"
    if len(page.get('children_pages')['data']) == 0:
        page_type == "article"

    slug_to_page[slug] = page
    slug_to_page_type[slug] = page_type


# In[ ]:


slug = this_page['attributes']['slug']


# In[ ]:


slug


# In[ ]:


df_slug_to_page_type = pd.DataFrame()
for this_page in all_pages:
    slug = this_page['attributes']['slug']
    df_slug_to_page_type = df_slug_to_page_type.append( pd.DataFrame([{"url":slug,"page_type":slug_to_page_type[slug]}]))


# In[ ]:


df_slug_to_page_type


# In[ ]:


documents_df_wtype[documents_df_wtype['page_type']=="category"]


# In[ ]:


# for each url
i = 0
for this_page in all_pages:
    
    slug = this_page['attributes']['slug']
    

    print(i, slug, slug_to_page_type[slug])
    print("http://localhost:8000/"+slug+"/")
    
    i = i +1
    top_similar_documents = get_top_similar_documents(slug, similarity_matrix, documents_df, 30)
    top_similar_documents = pd.merge(top_similar_documents, df_slug_to_page_type, how="left")
    
    # if article
    if slug_to_page_type[slug] == 'article':
        # then add related articles
        
        # add top articles until there are 3 related
        res = cms.get_by_slug("pages",slug)
        page = res['data']['attributes']
        page['related_pages']

        max_n = 4
        related_pages = []
        if len(page['related_pages']['data']) < max_n:

            for related_page in page['related_pages']['data']:
                related_pages.append(related_page['id'])

            # get related pages
            to_add = top_similar_documents[ top_similar_documents['page_type']=='article'][0:(max_n-len(page['related_pages']['data']))]

            for this_slug in to_add['url']:
                related_pages.append( all_slug_ids[this_slug])


            strapiPageUpdate = {
                "slug": page['slug'],
                "related_pages": related_pages
            }
            cms.create_or_update("pages",strapiPageUpdate)
            
        # randomly choose until there are 5 recommended
        max_n = 5
        recommended_pages = []
        if len(page['recommended_pages']['data']) < max_n:

            for related_page in page['recommended_pages']['data']:
                recommended_pages.append(related_page['id'])

            # get related pages
            to_add = top_similar_documents[ top_similar_documents['page_type']=='article'][5:]
            import random
            to_add = random.sample(list(to_add['url']), min(5,len(to_add['url'])))
            
            for this_slug in to_add:
                recommended_pages.append( all_slug_ids[this_slug])


            strapiPageUpdate = {
                "slug": page['slug'],
                "recommended_pages": recommended_pages
            }
            cms.create_or_update("pages",strapiPageUpdate)
            
        True
        
    # if category
    if slug_to_page_type[slug] == 'category':
        # then add similar categories as related

        max_n = 4
        related_pages = []
        if len(page['related_pages']['data']) < max_n:

            for related_page in page['related_pages']['data']:
                related_pages.append(related_page['id'])

            # get related pages
            to_add = top_similar_documents[ top_similar_documents['page_type']=='category'][0:(max_n-len(page['related_pages']['data']))]

            for this_slug in to_add['url']:
                related_pages.append( all_slug_ids[this_slug])


            strapiPageUpdate = {
                "slug": page['slug'],
                "related_pages": related_pages
            }
            cms.create_or_update("pages",strapiPageUpdate)
            
        # add until there are 5 related article pages as children
        # must be >0.92 match
        max_n = 5
        children_pages = []
        if len(page['children_pages']['data']) < max_n:

            for related_page in page['children_pages']['data']:
                children_pages.append(related_page['id'])

            # get related pages
            to_add = top_similar_documents[ top_similar_documents['page_type']=='article']
            to_add = to_add [ to_add['similarity']>0.92 ]
            
            to_add = to_add[0:min(len(to_add),max_n-len(children_pages))]
            
            for this_slug in to_add['url']:
                children_pages.append( all_slug_ids[this_slug])


            strapiPageUpdate = {
                "slug": page['slug'],
                "children_pages": children_pages
            }
            cms.create_or_update("pages",strapiPageUpdate)
            
        
        # no recommended articles
        True
        
    
    print()
    print()


# In[ ]:





# # plots

# In[ ]:


# plot all distance from avg

from statistics import mean, stdev
import matplotlib.pyplot as plt

# total population
dist = pairwise_differences[orig_len][0:orig_len]
dist_sd = stdev(dist) # sqrt(sum(dist-mean(dist))^2/len(dist))
dist_new = abs(dist-mean(dist))/dist_sd*3/2


# Generate data on commute times.
size, scale = 1000, 10

commutes = pd.Series([i for i in dist_new if i < 1])
commutes.plot.hist(grid=True, bins=5, rwidth=0.9,
                   color='c', label='On-topic')

commutes = pd.Series([i for i in dist_new if i >= 1 and i <= 3])
commutes.plot.hist(grid=True, rwidth=0.9,
                   color='m', label='Near-topic')

commutes = pd.Series([i for i in dist_new if i > 3])
commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='y', label='Off-topic')

plt.legend()
plt.title('Distribution of Website Topic Congruency')
plt.xlabel('Semantic Topical Similarity')
plt.ylabel('Count of Pages (URLs)')
plt.grid(axis='y', alpha=0.25)
plt.grid(axis='x', alpha=0)


# In[ ]:


documents_df['dist_new'] = dist_new_plus
documents_df['domain'] = ""
documents_df['page_count'] = 1

conditions = [
    (documents_df['dist_new'] <= 1),
    (documents_df['dist_new'] > 1) & (documents_df['dist_new'] <=3),
    (documents_df['dist_new'] > 3)
    ]

# create a list of the values we want to assign for each condition
values = ['on-topic', 'near-topic', 'off-topic']

# create a new column and use np.select to assign values to it using our lists as arguments
documents_df['tier'] = np.select(conditions, values)


# In[ ]:


documents_df.head()


# In[ ]:


documents_df[ documents_df['dist_new'] > 2 ]


# # pca plot

# In[ ]:


from sklearn.decomposition import PCA

# Reduce the dimensionality to 2 using PCA
pca = PCA(n_components=2)
reduced_matrix = pca.fit_transform(similarity_matrix)


# In[ ]:


import matplotlib.pyplot as plt

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], alpha=0.5)

# Annotate each point with its corresponding URL
if False:
    for i, url in enumerate(documents_df['url']):
        plt.annotate(url, (reduced_matrix[i, 0], reduced_matrix[i, 1]), fontsize=8)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('URL Similarity Visualization')

plt.show()


# # clusters

# In[ ]:


from sklearn.cluster import KMeans

# Number of clusters (you can adjust this)
n_clusters = 3

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(reduced_matrix)


# In[ ]:


import numpy as np

# Initialize an array to store the centroids
cluster_centroids = np.zeros((n_clusters, 2))

# Calculate centroids for each cluster
for cluster_id in range(n_clusters):
    cluster_points = reduced_matrix[cluster_labels == cluster_id]
    centroid = cluster_points.mean(axis=0)
    cluster_centroids[cluster_id] = centroid


# In[ ]:


plt.figure(figsize=(10, 8))
plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=cluster_labels, s=10, alpha=0.5)
plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='x', s=100, label='Centroids')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('URL Clustering with Centroids')
plt.legend()

plt.show()


# In[ ]:




