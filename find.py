from typing import Any

class BruteForceTokenizer():
    def __init__(self, PATH) -> None:
        with open(PATH+'dicionario_ptBR.txt', encoding='utf-8') as f:
            self.ptBR_dictionary = set(f.read().split('\n'))
    
    def __call__(self,  word, div=1, words=[], *args: Any, **kwds: Any) -> Any:
        longest_word = word[-div:]
        if len(longest_word) == 0:
            return words[::-1]
        else:
            if len(word) == 1:
                words.append(word)
                return words[::-1]
            if longest_word in self.ptBR_dictionary or longest_word+''.join(words[::-1]).replace('##', '') in self.ptBR_dictionary or longest_word=='.' or longest_word==',':
                words.append('##'+longest_word)
                self(word[:-div], 1, words)
                return words[::-1]
            else:
                self(word, div+1, words)
            return words[::-1]
        
class BruteForceWordFinder():
    def __init__(self, PATH) -> None:
        with open(PATH+'dicionario_ptBR.txt', encoding='utf-8') as f:
            self.ptBR_dictionary = set(f.read().split('\n'))
    
    def __call__(self,  tokens, div=1, words=[], *args: Any, **kwds: Any) -> Any:
        while div < len(tokens):
            if '##' in tokens[div]:
                div+=1
            else:
                break;
        return self._recursion(tokens, div, words)

    def _recursion(self, tokens, div, words):        
        longest_word = ''.join(tokens[:div]).replace('##', '')
        if len(longest_word) == 0:
            return words
        else:
            if longest_word in self.ptBR_dictionary or longest_word.lower() in self.ptBR_dictionary or longest_word=='.' or longest_word==',' or longest_word=='-':
                words.append(longest_word)
                self(tokens[div:],1,words)
                return words
            else:
                self._recursion(tokens, div-1, words)
            return words

PATH = "./texts/"
if __name__ == '__main__':
    tokenizer = BruteForceTokenizer(PATH)
    
    comos = tokenizer('comos', words=[])
    perf = tokenizer('emperfeita', words=[])
    verd = ['verde','##jan','##te','.']

    word_finder = BruteForceWordFinder(PATH)
    print(word_finder(comos, words=[]))
    print(word_finder(perf, words=[]))
    print(word_finder(verd, words=[]))