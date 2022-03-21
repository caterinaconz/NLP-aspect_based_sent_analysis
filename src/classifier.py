import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score
# Load pre-trained model tokenizer (vocabulary)
print('Importing tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Classifier:
    """The Classifier"""

    def clean_aspect_cat(self,df):
        for index, row in df.iterrows():
            row['aspect_category'] = row['aspect_category'].lower()
            row['aspect_category'] = row['aspect_category'].replace('#', ', ')

        return df

    def embeddings(self, train_path, test_path):
        print('Loading data')
        train_data = pd.read_csv(train_path, sep='\t', header=None)
        train_data.rename(
            columns={0: 'polarity', 1: 'aspect_category', 2: 'target_term', 3: 'character_offsets', 4: 'sentence'},
            inplace=True)

        dev_data = pd.read_csv(test_path, sep='\t', header=None)
        dev_data.rename(
            columns={0: 'polarity', 1: 'aspect_category', 2: 'target_term', 3: 'character_offsets', 4: 'sentence'},
            inplace=True)

        train_data.name = 'train'
        dev_data.name = 'dev'
        print('Cleaning data')
        train_data = Classifier().clean_aspect_cat(df = train_data)
        dev_data = Classifier().clean_aspect_cat(df = dev_data)
        print('Concatenating inputs')
        train_data['merged'] = ' [CLS] ' + train_data['aspect_category'] + ' [SEP] ' + train_data[
            'target_term'] + ' [SEP] ' + train_data['sentence'] + ' [SEP] '
        dev_data['merged'] = ' [CLS] ' + dev_data['aspect_category'] + ' [SEP] ' + dev_data['target_term'] + ' [SEP] ' + \
                             dev_data['sentence'] + ' [SEP] '
        print('Tokenizing')
        tokens = []
        idxs = []
        segments_ids = []
        for index, row in train_data.iterrows():
            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(row['merged'])
            tokens.append(tokenized_text)
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            idxs.append(indexed_tokens)

            segments_ids.append([1] * len(tokenized_text))

        train_data['tokens'] = tokens
        train_data['indexes'] = idxs
        train_data['segment_ids'] = segments_ids
        train_data.head(2)

        tokens = []
        idxs = []
        segments_ids = []
        for index, row in dev_data.iterrows():
            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(row['merged'])
            tokens.append(tokenized_text)
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            idxs.append(indexed_tokens)

            segments_ids.append([1] * len(tokenized_text))

        dev_data['tokens'] = tokens
        dev_data['indexes'] = idxs
        dev_data['segment_ids'] = segments_ids

        # Load pre-trained model (weights)
        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states=True,  # Whether the model returns all hidden-states.
                                          )

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

        dfs = [dev_data,train_data]
        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.

        for df in dfs:
            print('Creating the embeddings for ' + df.name + 'data')
            hidden_states = []
            with torch.no_grad():
                for index, row in df.iterrows():
                    tokens_tensor = torch.tensor([row['indexes']])
                    segments_tensors = torch.tensor([row['segment_ids']])
                    outputs = model(tokens_tensor, segments_tensors)

                    hidden_states.append(outputs[2])

            # Concatenate the tensors for all layers. We use `stack` here to
            # create a new dimension in the tensor.
            token_embeddings = []
            for hidden_state in hidden_states:
                token_embeddings.append(torch.squeeze(torch.stack(hidden_state, dim=0), dim=1).permute(1, 0, 2))

            token_embeddings[0].size()

            # `hidden_states` has shape [13 x 1 x 22 x 768]
            sentence_embedding = []
            for hidden_state in hidden_states:
                # `token_vecs` is a tensor with shape [22 x 768]
                token_vecs = hidden_state[-2][0]

                # Calculate the average of all 22 token vectors.
                sentence_embedding.append(torch.mean(token_vecs, dim=0))

            sentence_embedding_np = []

            for sentence_embedding_ind in sentence_embedding:
                sentence_embedding_np.append(sentence_embedding_ind.detach().numpy())

            print('Saving the data')
            df['sentence_embedding'] = sentence_embedding_np
            features_df = pd.DataFrame(df['sentence_embedding'].to_list())
            features_df['labels'] = df['polarity']
            path_save = train_path.replace('traindata.csv', '')
            features_df.to_csv(path_save + df.name + "_embeddings_catboost.csv")


        #############################################
    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        #Call the embedding functions to create a save the sentence vectors which will form our features
        Classifier().embeddings(trainfile, devfile)
        ##Call the data
        datadir = "../data/"
        df = pd.read_csv(datadir + 'train_embeddings_catboost.csv', index_col=0)

        train_cols_x = list(df.drop(['labels'], axis=1).columns)
        train_x = df[train_cols_x]
        train_y = df['labels']
        train_y = np.transpose(np.array(train_y)).ravel()



        # define dataset used to train the model
        train_dataset = Pool(data=train_x,
                             label=train_y,
                             )


        if torch.cuda.is_available():
            device = "GPU"
        else:
            device = 'CPU'

        # set model parameters

        
        
        self.model = CatBoostClassifier(
            task_type=device,
            iterations=1000,
            # random_strength=0.5, #reduce overfitting
            depth=6,  # depth of the tree
            l2_leaf_reg=3,
            # border_count=32,
            bagging_temperature=2,
            learning_rate=0.05,
            sampling_unit='Object',
            sampling_frequency='PerTree',
            rsm=1,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            boosting_type='Plain',
            verbose=200
        )
        print('Fitting the model')
        self.model.fit(train_dataset, plot=False)



    def predict(self, testfile = None):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        ##Call the data
        datadir = "../data/"
        test_df = pd.read_csv(datadir + 'dev_embeddings_catboost.csv', index_col=0)
        test_cols_x = list(test_df.drop(['labels'], axis=1).columns)
        test_x = test_df[test_cols_x]
        test_y = test_df['labels']
        test_y = np.transpose(np.array(test_y)).ravel()

        y_pred = list(self.model.predict(test_x))
        #accuracy = accuracy_score(test_y, y_pred)

        #print("accuracy in TEST is " + str(accuracy))
        return y_pred

'''
c = Classifier()
datadir = "../data/"
c.train(datadir + 'traindata.csv', datadir + 'devdata.csv')
predictions = c.predict()
print(type(predictions))
print(len(predictions))
'''


