Contributors:

Arora, Aprajita - aprajita.arora@student-cs.fr
Bardi Vale, Yago - yago.bardi@student-cs.fr
Conz, Caterina - caterina.conz@student-cs.fr
Sever, Fatmanur - fatmanur.sever@student-cs.fr

Methodology:

We first clean the aspect category from (for example) 'FOOD#QUALITY' to 'food, quality'. Next we concatenate the 3 inputs
in the following way: '[CLS] ' + aspect_category + ' [SEP] ' + aspect_term + ' [SEP] ' + review_sentence + ' [SEP]'. This
concatenated string is then tokenized, and the token indexes are passed to a pre-trained BERT model which passes them
through a neural network. We then extract the the vectors which represent each sentence by retrieving the last 3 hidden layers
of this neural network and averaging them. this results in a 768 dimensions vector for each of the concatenated sentences.
Each of this dimensions will constitute a feature for our model, CatBoost. For this model we simply pass the 768 features
(one columns per dimension) and the labels. Once fitted on the training data, it results in 82.7% accuracy on the dev data.
The parameters for CatBoost where chosen after doing a gridsearch.
