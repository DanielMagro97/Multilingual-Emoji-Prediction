# Multilingual-Emoji-Prediction
This project was submitted for the Assignment component of the UoM LIN3012 - Data-Driven Natural Language Processing unit. A BiDirectional LSTM was used to solve the SemEval 2018 task of Emoji Prediction for multilingual text, namely tweets.

All code is described within the scripts as inline comments\
train_val_curve_US and train_val_curve_ES show the training and validation loss over epochs graph

EmojiPredictionUS.py can be run to train a BiDirectional LSTM over the US tweet data set. This will save the trained model to the 'models' folder.\
After training, the script will also evaluate the model according to the official evaluation script.\
EmojiPredictionES.py does the same process, however on the ES tweet data set.

scorer_semeval18.py is the official evaluation script.\
Evaluation.py is a slight modification of scorer_semeval18.py. The only difference is that Evaluation.py can be called directly from another script after a model has been trained, and does not need to be run from a command line with arguments as the official does.

RunModels.py can be run to load the models that have already been trained by the EmojiPredictionUS/ES.py scripts from the 'models' folder, and use them to predict the test data's labels, and then evaluate the model.\
The script also allows for qualitative evaluation of the model, by showing correctly and incorrectly classified instances.

EmojiPredictionUS_FHP.py and EmojiPredictionES_FHP.py are an application of the skopt package on the EmojiPredictionUS/ES.py scripts.\
Either of these scripts can be run so skopt can try out different hyper parameter combinations and then evaluates them, to fine tune the hyper parameters.

Due to file size restrictions, the US training data and the US trained models will not be uploaded. These can, however be downloaded from the following link:\
https://drive.google.com/drive/folders/1BRkHAFeTO4zNV99KT7Q8miMoNcklGyXA?usp=sharing

All the scripts can be run on a Python 3.6 environment. To install all the required packages, the following commands can be run for an anaconda environment:\
conda install keras matplotlib nltk\
pip install scikit-optimize	(only required for hyper parameter optimisation scripts (_FHP files))

The scripts can then be run from a terminal by running:\
python script_name.py

In order to run the RunModels.py script, which uses the trained models to predict the test data and then evaluates the performance, one must either download the us model and vocab from the provided google drive link (MultilingualEmojiPrediction/models/us_rnn_model.h5 and us_vocab.pkl) and place them in the same directory of the submitted project (same folder as es model and vocab)\
or download the entire MultilingualEmojiPrediction folder from Google Drive\
or simply comment out lines 79-109 and 160-176 of the script so as not to run the us model

To run the EmojiPredictionUS.py, the US training data must be downloaded from Google Drive\
(MultilingualEmojiPrediction/Semeval2018-Task2-EmojiPrediction/train/crawler/data/us)
