import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM


tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_basic_tokenize=False)
model = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')

def get_next_word_probability(sentence, next_word):
    sentence[-1] = "[MASK]"
    masked_sentence = ' '.join(sentence)
    tokenized_sentence = tokenizer.tokenize(masked_sentence)

    token_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)

    mask_index = tokenized_sentence.index("[MASK]")

    token_tensor = torch.tensor([token_ids])
    with torch.no_grad():
        outputs = model(token_tensor)
        predictions = outputs.logits[0, mask_index]

    next_word_index = tokenizer.convert_tokens_to_ids(next_word)
    next_word_probability = predictions[next_word_index].item()
    tokenizer.convert_tokens_to_string
    return next_word_probability

PATH = "./texts/"
if __name__ == '__main__':    
    with open(PATH+'bela_vista.txt', encoding='utf-8') as f:
        text = f.read()
    words = text.split(' ')

    # # Token Embeddings
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # input_tensor = torch.tensor([input_ids])
    # with torch.no_grad():
    #     outputs = model(input_tensor)
    #     embeddings = outputs[0][0]  # Extract token embeddings from the last layer

    for id, word in enumerate(words):
        sentence = words[:id+2]
        next_word = words[id+1]
        
        probability = get_next_word_probability(sentence, next_word)

        # curr_embedding = embeddings[id+1]
        # next_embedding = embeddings[id+2]
        # if probability > 1 or torch.cosine_similarity(next_embedding, curr_embedding, dim=0) >= 0.8:
        #     tokens[id+2] = tokens[id+2].replace('##', '')
            #TODO combine words that are in multiple tokens (guloso?)
                
        print(f"A probabilidade de '{next_word}' ser a próxima palavra: {probability}")
