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
            if longest_word in self.ptBR_dictionary or longest_word+''.join(words[::-1]).replace('##', '') in self.ptBR_dictionary:
                words.append('##'+longest_word)
                self(word[:-div], 1, words)
                return words[::-1]
            else:
                self(word, div+1, words)
            return words[::-1]

PATH = "./texts/"
if __name__ == '__main__':
    tokenizer = BruteForceTokenizer(PATH)
    
    print(tokenizer('comos', words=[]))
    print(tokenizer('emperfeita', words=[]))