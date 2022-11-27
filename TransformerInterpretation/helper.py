'''
Prateek Gulati
11/1/2022
CS 7180 Advanced Perception
'''

'''
Includes helper functions used in the jupyter notebook
'''
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import torch
import seaborn as sns

# function to load the pandas dataframe
def load_df(text_path,label_path):
    with open(text_path,'rt') as fi:
        texts = fi.read().strip().split('\n')
    text_dfs = pd.Series(data=texts,name='text',dtype='str')
    labels_dfs = pd.read_csv(label_path,names=['label'],index_col=False).label
    ret_df = pd.concat([text_dfs,labels_dfs],axis=1)
    return ret_df

# Removing urls from text: part of preprocessing
def encode_urls(row):
    row.text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))","HTTPURL", row.text)
    return row

# Removing mentions and hashtags from text: part of preprocessing
def encode_mentions_hashtags(row):
    row.text = row.text.replace('@',' @')
    row.text = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)","@USER", row.text)
    row.text = row.text.replace('#',' ')
    return row

# Decode emojis from text: part of preprocessing
def encode_emojis(row):
    row.text = emoji.demojize(row.text)
    return row

# Remove extra spaces from text: part of preprocessing
def remove_extra_spaces(row):
    row.text = ' '.join(row.text.split())
    return row

# text to lower case: part of preprocessing
def lower_text(row):
    row.text = row.text.lower()
    return row

# main function to preprocess all data in a dataframe series
def preprocess_data_df(df):
    df = df.apply(encode_urls,axis=1)
    df = df.apply(encode_mentions_hashtags,axis=1)
    df = df.apply(encode_emojis,axis=1)
    df = df.apply(remove_extra_spaces,axis=1)
    df = df.apply(lower_text,axis=1)
    return df

# function to visualize embeddings given a dimensionality reducer, embeddings and layer numbers
def visualize_layerwise_embeddings(dim_reducer, hidden_states,masks,ys,title,layers_to_visualize=[0,1,-2,-1]):
    print('visualize_layerwise_embeddings for',title)
    num_layers = len(layers_to_visualize)
    fig = plt.figure(figsize=(24,(num_layers/4)*6)) #each subplot of size 6x6
    ax = [fig.add_subplot(num_layers//4,4,i+1) for i in range(num_layers)]
    ys = ys.numpy().reshape(-1)
    for i,layer_i in enumerate(layers_to_visualize):#range(hidden_states):
        layer_hidden_states = hidden_states[layer_i]
        averaged_layer_hidden_states = torch.div(layer_hidden_states.sum(dim=1),masks.sum(dim=1,keepdim=True))
        layer_dim_reduced_vectors = dim_reducer.fit_transform(averaged_layer_hidden_states.numpy())
        df = pd.DataFrame.from_dict({'x':layer_dim_reduced_vectors[:,0],'y':layer_dim_reduced_vectors[:,1],'label':ys})
        df.label = df.label.astype(int)
        sns.scatterplot(data=df,x='x',y='y',hue='label',ax=ax[i])
        fig.suptitle(f"{title}:")
        if layer_i < 0:
            ax[i].set_title(f"layer {len(hidden_states) + layer_i+1}")
        else:
            ax[i].set_title(f"layer {layer_i+1}")
    print()
    
# dataloader for transformer
def get_bert_encoded_data_in_batches(tokenizer, df,batch_size = 0,max_seq_length = 50):
    data = [(row.text,row.label,) for _,row in df.iterrows()]
    sampler = torch.utils.data.sampler.SequentialSampler(data)
    batch_sampler = torch.utils.data.BatchSampler(sampler,batch_size=batch_size if batch_size > 0 else len(data), drop_last=False)
    for batch in batch_sampler:
        encoded_batch_data = tokenizer.batch_encode_plus([data[i][0] for i in batch],max_length = max_seq_length,pad_to_max_length=True,truncation=True)
        seq = torch.tensor(encoded_batch_data['input_ids'])
        mask = torch.tensor(encoded_batch_data['attention_mask'])
        yield (seq,mask),torch.LongTensor([data[i][1] for i in batch])