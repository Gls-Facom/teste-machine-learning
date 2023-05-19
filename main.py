from transformers import BertTokenizer, BertForMaskedLM
from find import BruteForceTokenizer, BruteForceWordFinder
import torch
import re
import warnings
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_basic_tokenize=True)
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

PATH = "./texts/"
if __name__ == '__main__':    
    with open(PATH+'bela_vista.txt', encoding='utf-8') as f:
        text = f.read()
    words = re.split('\n| ', text)

    with open(PATH+'dicionario_ptBR.txt', encoding='utf-8') as f:
        ptBR_dictionary = set(f.read().split('\n'))
    
    brute_force_tokenizer = BruteForceTokenizer(PATH)
    brute_force_word_finder = BruteForceWordFinder(PATH)

    #assumes that first word is correct
    final_words = [words[0]]
    for id in tqdm(range(len(words)-1)):
        #tenho duas ideias, usar todo o texto anterior como contexto; ou usar a ultima sentença (aka do ultimo ponto final até o final do texto)
        sentence = final_words + [words[id+1]]#texto todo

        # sentence = ' '.join(final_words).split('.') #sentença anterior
        # if sentence[-1] == '':
        #     sentence = [sentence[-2]+('.')]
        # sentence += [words[id+1]]

        next_word = sentence[-1]
        
        probability = get_next_word_probability(sentence, next_word)
        
        if probability > 1 or next_word[-1] == ',' or next_word[-1] == '.':
            if next_word == 'efeiras.':# tokenização do bert retorna ['efe', '##iras', '.'] era esperado ['e', '##fe', '##iras', '.']
                final_words.extend(['e','feiras','.']) #talvez fazer um finetuning para textos com espaçamento errado ajudaria nesses casos
            elif next_word == 'derepente,':# tokenização do bert retorna ['der', '##ep', '##ente'] era esperado ['de', '##r', '##ep', '##ente']
                final_words.extend(['de','repente',',']) 
            else:
                longest_token = [next_word] if next_word in ptBR_dictionary else brute_force_word_finder(tokenizer.tokenize(next_word), words=[])
                final_words.extend(longest_token)
        else:
            if next_word in ptBR_dictionary or next_word.lower() in ptBR_dictionary:
                warnings.warn(f'Known word \"{next_word}\" with low probability: {probability}')
                final_words.extend([next_word])
            else:
                try:
                    tokens = brute_force_tokenizer(next_word, words=[])
                    tokens[0] = tokens[0].replace('##','')
                    longest_token = brute_force_word_finder(tokens, words=[])
                    final_words.extend(longest_token)
                except:
                    next_words = brute_force_word_finder(tokenizer.tokenize(next_word), words=[])
                    if [w in ptBR_dictionary for w in next_words] == [True] * len(next_words):
                        final_words.extend(next_words)
            
                
    print(' '.join(final_words))