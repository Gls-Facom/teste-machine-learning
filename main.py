from transformers import BertTokenizer, BertForMaskedLM
import torch
from find import BruteForceTokenizer

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_special_tokens=True)
model = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
model.eval()

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
    return next_word_probability

def find_longest_token(tokens, words=[]):
    longest_token = ''.join(tokens).replace('##','')
    if longest_token in ptBR_dictionary or tokenizer.convert_tokens_to_ids(longest_token) != tokenizer.all_special_ids[0]:
        words.append(longest_token)
        return words
    else:
        find_longest_token(tokens[:-1], words)
        find_longest_token(tokens[-1], words)
        return words

PATH = "./texts/"
if __name__ == '__main__':    
    with open(PATH+'bela_vista.txt', encoding='utf-8') as f:
        text = f.read()
    words = text.split(' ')

    with open(PATH+'dicionario_ptBR.txt', encoding='utf-8') as f:
        ptBR_dictionary = set(f.read().split('\n'))
    
    brute_force_tokenizer = BruteForceTokenizer(PATH)
    # Token Embeddings
    input_ids = tokenizer.encode(words, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0][0]  # Extract token embeddings from the last layer

    #assumes that first word is correct
    final_words = [words[0]]
    for id in range(len(words)-1):
        sentence = final_words + [words[id+1]]
        next_word = sentence[-1]
        
        probability = get_next_word_probability(sentence, next_word)
        
        if probability > 1: 
            longest_token = find_longest_token(tokenizer.tokenize(next_word), [])
            final_words.extend(longest_token)
        else:
            if next_word in ptBR_dictionary:
                print('should never print this')
            else:
                tokens = brute_force_tokenizer(next_word, words=[])[0].replace('##','')
                print(tokens)
                
                
        print(' '.join(final_words))