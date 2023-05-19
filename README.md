![Logo AI Solutions](http://aisolutions.tec.br/wp-content/uploads/sites/2/2019/04/logo.png)

# AI Solutions

## Teste para novos candidatos

### Introdução

O teste consiste em arrumar o espaçamento entre as palavras do arquivo bela_vista.txt.

O candidato pode usar qualquer técnica para realizar esse teste, mas deve colocar o código no repositório e descrever detalhadamente as técnicas que utilizou.

#### Explicação da solução
Usei o BERT pre-treinado em textos da lingua portuguesa, o [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased).

A ideia era usar o modelo para linguagem com máscara para predizer a próxima palavra dado o contexto, e comparar se essa palavra é a que está no texto. Caso a palavra presente no texto não seja uma palavra contida no dicionário da língua portuguesa, essa palavra era tokenizada e separada em palavras presentes no dicionário. Os algortitmos feitos para achar as melhores palavras foram baseados na tokenização do BERT, existem casos que a tokenização não é feita de maneira ótima e o algoritmo deixa a desejar. Para produzir um texto coeerente foram adicionadas correções hardcoded para produzir um resultado. Creio que fosse realizado um fine-tuning no modelo, não teriamos esses problemas.

Dois tipos de contexto foram testados. O contexto da ultima sentença até o ultimo ponto; E o contexto completo com todas as palavras até o momento. O resultado produzido foi suficiente para as duas abordagens, porém a vertente com o contexto completo tende a realizar menos predições erradas (predizer que a palavra não é a proxima mesmo ela sendo uma palavra válida da lingua portuguesa), apesar de ser incrementalmente mais lenta conforme o contexto aumenta.

##### Possíveis melhorias
- Finetuning em textos sem espaçamento.
- Usar embeddings para escolher as melhores palavras, ao invés de pegar palavras que só precisam existir no dicionário.