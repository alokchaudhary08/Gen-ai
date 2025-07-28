import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import io
import os
def load_data(path , num_samples=10000):
    #open the txt file split by space
    with open(path , 'r' , encoding ='utf-8') as f:
        lines = f.read().split('\n')
    input_texts = []
    target_texts = []
    #separately store input and target text
    for line in lines[:num_samples]:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        #now here the given dataset is french to english but i want eng to french trans so careful with the order
        eng , fra = parts[0] , parts[1]
        #add start and end for decoder
        target_text = "<start> " + fra + " <end>"
        input_texts.append(eng)
        target_texts.append(target_text)
        
    return input_texts, target_texts

input_texts, target_texts = load_data('fra.txt')
def preprocess_text(sentences):
    #convert in lowercase
    sentences =  [s.lower() for s in sentences]
    #remove everything except letters , digits , white spaces , and <>(remember the <start>)
    #dont forget ^ this its the negation symbol i forgot and spend 30 minutes looking for bug
    sentences = [re.sub(r"[^a-zA-Z0-9<>\s]", "", s) for s in sentences]
    return sentences
input_texts = preprocess_text(input_texts)
target_texts = preprocess_text(target_texts)
# Tokenize output
#as we have already cleaned the text filter is off('' - do not remove anything)
input_tokenizer = Tokenizer(filters = '')
#go through all sentences and build a word-index vocab
input_tokenizer.fit_on_texts(input_texts)
#use the vocab and replce each word with its integer ID
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
#extract the word to index vocab or dict. for future uses
input_word_index = input_tokenizer.word_index
#find the max.sequence length
max_input_len = max(len(seq) for seq in input_sequences)
#pad with 0 if length is lower than max length
encoder_input_data = pad_sequences(input_sequences , maxlen = max_input_len ,padding='post')
## Creating decoder targets

#create a numpy array of same shape as decoder_input_data
decoder_target_data = np.zeros_like(decoder_input_data)
#now we take all the columns from 1 to end from decoder_input_data and fill it in decoder_target_data from 0 to end - 1 
#so basically we shifted left
decoder_target_data[:,:-1] = decoder_input_data[:, 1:]
#well this is actually not necessary but i still did that just to be safe , it does nothing but make sure last value is zero
decoder_target_data[:, -1] = 0
#embedding dimensions
embedding_dim = 256 
#LSTM hidden units
lstm_units = 512
#vocab size , +1 for padding token
input_vocab_size = len(input_word_index)+1
target_vocab_size = len(target_word_index)+1
#Encoder Model
#encoder inputs
encoder_inputs = Input(shape = (None , ))
#embedding layer take encoder inputs of size vocab size as defined above and embed them and convert into dimensions of embedding_dim
enc_emb = Embedding(input_vocab_size ,embedding_dim , mask_zero = True)(encoder_inputs)
#a LSTM layer with hidden units = lstm_units and we want final hidden state as well as cell state so return_state is true
encoder_outputs , state_h , state_c = LSTM(lstm_units, return_state = True)(enc_emb)
#store the stats
encoder_states = [state_h , state_c]
#Decoder Model
#No change so no comment in next two lines
decoder_inputs = Input(shape = (None , ))
decoder_emb_layer = Embedding(target_vocab_size, embedding_dim, mask_zero=True, name='dec_emb')
dec_emb = decoder_emb_layer(decoder_inputs)
#just defining the LSTM layer and in addition to states we want output at each state/time step so return_sequence = True
decoder_lstm = LSTM(lstm_units,return_sequences = True, return_state = True)
#get the outputs by giving dec_emb as input and passing encoder's states as initial states
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = encoder_states)
#create a dense layer as softmax as activation
decoder_dense = Dense(target_vocab_size, activation = 'softmax')
#pass previous outputs as inputs in dense layer and get new outputs 
decoder_outputs =  decoder_dense(decoder_outputs)
#Training Model
#combined model with teacher forcing
model = Model([encoder_inputs,decoder_inputs],decoder_outputs)
#we are using 'sparse_categorical_crossentropy' cause we didnt one-hot encode targets
model.compile(optimizer = 'adam' , loss ='sparse_categorical_crossentropy' , metrics=['accuracy'])
#lets check our model 
model.summary()
#Train the Model
#... maens all the dimensions
history = model.fit([encoder_input_data,decoder_input_data], decoder_target_data[...,np.newaxis], 
                    batch_size = 64, epochs = 30,
                   validation_split = 0.2)
# Inference encoder model: just takes input and gives out hidden states
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the embedding and LSTM
dec_emb2 = decoder_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model for inference
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)
reverse_target_index = {i: word for word, i in target_tokenizer.word_index.items()}
reverse_target_index[0] = ''  # For padding

target_index_word = target_tokenizer.word_index
start_token = target_index_word['<start>']
end_token = target_index_word['<end>']

def decode_sequence(input_seq):
    # Encode the input
    states_value = encoder_model.predict(input_seq)

    # Prepare start token as first input to decoder
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample the best word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_index.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

        # Update decoder input for next time step
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return ' '.join(decoded_sentence)
#lets check
def predict_sample(index):
    input_seq = encoder_input_data[index:index+1]
    decoded = decode_sequence(input_seq)
    print("Input:", input_texts[index])
    print("Target:", target_texts[index])
    print("Predicted:", decoded)


predict_sample(0)
# lets translate and check 5 sentences
def predict_samples(start_index, num_samples=5):
    for i in range(start_index, start_index + num_samples):
        input_seq = encoder_input_data[i:i+1]
        decoded = decode_sequence(input_seq)
        print(f"Sample {i}:")
        print("Input: ", input_texts[i])
        print("Target:", target_texts[i])
        print("Predicted:", decoded)
        print("="*50)

# predict 5 samples from index 0
predict_samples(0, 5)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4  
def evaluate_bleu_score(num_samples=100):
    total_score = 0.0
    individual_scores = []

    for i in range(num_samples):
        input_seq = encoder_input_data[i:i+1]
        decoded_sentence = decode_sequence(input_seq).strip().split()
        reference_sentence = target_texts[i].replace('<start>', '').replace('<end>', '').strip().split()

        score = sentence_bleu([reference_sentence], decoded_sentence, smoothing_function=smoothie)
        individual_scores.append(score)
        total_score += score

    avg_bleu = total_score / num_samples
    print(f"\nAverage BLEU score on {num_samples} samples: {avg_bleu:.4f}")
    return avg_bleu, individual_scores
avg_bleu, scores = evaluate_bleu_score(100)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy and Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()