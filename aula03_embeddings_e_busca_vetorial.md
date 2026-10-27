# Aula 3 — Embeddings e Busca Vetorial

**Projeto:** Assistente Virtual RAG da TechStore Conecta
**Duração sugerida:** ~100 min
**Entregas da aula:** `indice.faiss` e módulo `busca.py` no Google Drive

> 💡 A GPU acelera a geração dos embeddings, mas com poucos documentos a CPU também dá conta.

---

## 🎯 Objetivos

Ao final desta aula, o aluno será capaz de:

1. Explicar o que é um **embedding** e o que é **similaridade de cosseno**.
2. Comparar a busca por palavras (TF-IDF) com a **busca semântica**.
3. Gerar embeddings dos chunks com `sentence-transformers`.
4. Construir e salvar um **índice vetorial FAISS**.
5. Avaliar a recuperação com a métrica **hit rate @k**.

## ⏱️ Roteiro da aula

| Tempo | Atividade |
|-------|-----------|
| 0–15 min | Conceito: embeddings e espaço vetorial |
| 15–35 min | Primeiros embeddings e matriz de similaridade |
| 35–45 min | TF-IDF × embeddings: a diferença na prática |
| 45–65 min | Indexação dos chunks com FAISS |
| 65–75 min | Módulo `busca.py` |
| 75–90 min | Avaliação da busca (hit rate @k) |
| 90–100 min | Exercícios e checklist |

---

## 📚 Parte 1 — Conceitos (15 min)

### Embedding

Um **embedding** é uma lista de números (vetor) que representa o **significado** de um texto. Textos com sentidos parecidos ficam **próximos** no espaço vetorial, mesmo que usem palavras diferentes.

```
        "meu fone pifou"  ●  ● "o aparelho parou de funcionar"
                            ↖ próximos (sentido parecido)

                                         ● "aceitam pix?"
                                              ↖ distante (outro assunto)
```

O modelo que usaremos (`paraphrase-multilingual-MiniLM-L12-v2`) transforma qualquer frase em um vetor de **384 números** e entende mais de 50 idiomas, incluindo o português.

### Similaridade de cosseno

Mede o ângulo entre dois vetores:

$$\text{cos}(A, B) = \frac{A \cdot B}{\|A\|\,\|B\|}$$

| Valor | Interpretação |
|-------|---------------|
| próximo de 1 | sentidos muito parecidos |
| próximo de 0 | sem relação |

> 💡 Se os vetores forem **normalizados** (comprimento 1), o cosseno é simplesmente o **produto escalar**. Vamos usar esse truque com o FAISS.

### TF-IDF (projeto anterior) × Embeddings

| | TF-IDF | Embeddings |
|---|---|---|
| Representa | Palavras presentes | Significado |
| Sinônimos | ❌ "pifou" ≠ "estragou" | ✅ reconhece |
| Vetor | Esparso, tamanho = vocabulário | Denso, tamanho fixo (384) |
| Precisa de treino? | Sim, na sua base | Não, o modelo já vem pré-treinado |

---

## ⚙️ Parte 2 — Primeiros embeddings (20 min)

Crie um notebook `aula03_rag.ipynb`.

```python
# Célula 1 — Instalação e Drive
!pip install -q sentence-transformers faiss-cpu

from google.colab import drive
drive.mount('/content/drive')

import sys, json
PASTA = '/content/drive/MyDrive/chatbot_rag'
sys.path.append(PASTA)
```

```python
# Célula 2 — Carregar o modelo de embeddings
from sentence_transformers import SentenceTransformer

MODELO_EMB = 'paraphrase-multilingual-MiniLM-L12-v2'
emb_modelo = SentenceTransformer(MODELO_EMB)

vetor = emb_modelo.encode('Quero devolver meu fone de ouvido')
print('Dimensões:', vetor.shape)
print('Primeiros valores:', vetor[:8].round(3))
```

```python
# Célula 3 — Matriz de similaridade entre frases
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import util

frases = [
    "meu fone pifou",
    "o aparelho parou de funcionar",
    "quero devolver o produto",
    "aceitam pix?",
    "quais as formas de pagamento?",
    "quanto tempo demora a entrega?",
]
vetores = emb_modelo.encode(frases, normalize_embeddings=True)
sim = util.cos_sim(vetores, vetores).numpy()

df_sim = pd.DataFrame(sim.round(2), index=frases, columns=frases)
plt.figure(figsize=(7, 6))
plt.imshow(df_sim, cmap='Blues')
plt.xticks(range(len(frases)), frases, rotation=45, ha='right')
plt.yticks(range(len(frases)), frases)
for i in range(len(frases)):
    for j in range(len(frases)):
        plt.text(j, i, df_sim.iloc[i, j], ha='center', va='center', fontsize=8)
plt.title('Similaridade de cosseno')
plt.tight_layout()
plt.show()
```

> 🗣️ **Discussão:** quais pares têm maior similaridade? "meu fone pifou" e "o aparelho parou de funcionar" não compartilham **nenhuma** palavra. Como o modelo sabe que são parecidas?

---

## ⚖️ Parte 3 — TF-IDF × Embeddings na prática (10 min)

```python
# Célula 4 — Mesma busca com as duas técnicas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base = ["O produto apresentou defeito e parou de funcionar",
        "Aceitamos pagamento via Pix com desconto",
        "O prazo de entrega varia conforme a região"]
consulta = "meu aparelho pifou"

tfidf = TfidfVectorizer().fit(base + [consulta])
s_tfidf = cosine_similarity(tfidf.transform([consulta]), tfidf.transform(base))[0]

s_emb = util.cos_sim(emb_modelo.encode(consulta, normalize_embeddings=True),
                     emb_modelo.encode(base, normalize_embeddings=True))[0].numpy()

pd.DataFrame({'texto': base, 'TF-IDF': s_tfidf.round(2), 'Embeddings': s_emb.round(2)})
```

> 📌 O TF-IDF dá **zero** para todas as frases, pois nenhuma palavra da consulta aparece na base. Os embeddings encontram o trecho correto pelo **sentido**.

---

## 🗄️ Parte 4 — Indexando os chunks com FAISS (20 min)

### O que é um índice (ou banco) vetorial?

Com milhões de chunks, comparar a pergunta com cada um deles fica lento. Bancos e índices vetoriais (FAISS, Chroma, Qdrant, pgvector…) organizam os vetores para buscas rápidas. O **FAISS**, criado pela Meta, é gratuito e roda localmente.

```python
# Célula 5 — Carregar os chunks da Aula 2 e gerar os embeddings
import numpy as np

with open(f'{PASTA}/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

textos = [c['texto'] for c in chunks]
emb_chunks = emb_modelo.encode(textos, normalize_embeddings=True,
                               show_progress_bar=True).astype('float32')
print('Matriz de embeddings:', emb_chunks.shape, '(chunks x dimensões)')
```

```python
# Célula 6 — Criar o índice FAISS e salvar no Drive
import faiss

dimensao = emb_chunks.shape[1]
indice = faiss.IndexFlatIP(dimensao)      # IP = produto interno (= cosseno com vetores normalizados)
indice.add(emb_chunks)

faiss.write_index(indice, f'{PASTA}/indice.faiss')
print('Vetores no índice:', indice.ntotal)
```

> ⚠️ **Importante:** a posição de cada vetor no índice corresponde à posição do chunk em `chunks.json`. Se os chunks forem regerados, o índice **precisa** ser refeito.

```python
# Célula 7 — Primeira busca semântica
def buscar_teste(pergunta, k=3):
    v = emb_modelo.encode([pergunta], normalize_embeddings=True).astype('float32')
    scores, ids = indice.search(v, k)
    for s, i in zip(scores[0], ids[0]):
        print(f"[{s:.2f}] {chunks[i]['id']}")
        print('     ', chunks[i]['texto'][:120].replace('\n', ' '), '...')

buscar_teste("o fone pode ser usado na piscina?")
```

---

## 🧩 Parte 5 — Módulo `busca.py` (10 min)

```python
%%writefile /content/drive/MyDrive/chatbot_rag/busca.py
# Módulo de busca semântica sobre os chunks indexados no FAISS
import json
import faiss
from sentence_transformers import SentenceTransformer

MODELO_EMB = 'paraphrase-multilingual-MiniLM-L12-v2'

_emb_modelo = None
_indice = None
_chunks = None


def carregar_busca(pasta, modelo=MODELO_EMB):
    global _emb_modelo, _indice, _chunks
    _emb_modelo = SentenceTransformer(modelo)
    _indice = faiss.read_index(f'{pasta}/indice.faiss')
    with open(f'{pasta}/chunks.json', encoding='utf-8') as f:
        _chunks = json.load(f)
    print(f'Busca pronta: {_indice.ntotal} chunks indexados.')


def buscar(pergunta, k=3):
    """Retorna os k chunks mais parecidos com a pergunta, com o score de similaridade."""
    vetor = _emb_modelo.encode([pergunta], normalize_embeddings=True).astype('float32')
    scores, ids = _indice.search(vetor, k)
    resultados = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        item = dict(_chunks[i])
        item['score'] = float(score)
        resultados.append(item)
    return resultados
```

```python
# Célula 9 — Testar o módulo
from busca import carregar_busca, buscar

carregar_busca(PASTA)
for r in buscar("Quantos pontos ganho por real gasto?", k=3):
    print(f"[{r['score']:.2f}] {r['fonte']}")
```

---

## 📏 Parte 6 — Avaliando a busca (15 min)

Se a busca não encontrar o trecho certo, **nenhuma LLM** conseguirá responder corretamente. Por isso avaliamos a recuperação **separadamente**.

**Hit rate @k:** percentual de perguntas em que o documento esperado aparece entre os *k* primeiros resultados.

```python
# Célula 10 — Conjunto de teste: pergunta -> documento que contém a resposta
teste = [
    ("Quantos pontos eu ganho a cada real gasto?",           "programa_conecta_mais.md"),
    ("Posso transferir meus pontos para minha mãe?",         "programa_conecta_mais.md"),
    ("Quanto tempo tenho para devolver algo que não gostei?", "trocas_devolucoes.md"),
    ("Em quanto tempo recebo o estorno no cartão?",          "trocas_devolucoes.md"),
    ("Quanto custa a garantia estendida de 24 meses?",       "garantia.md"),
    ("Qual o prazo de entrega para o Nordeste?",             "entregas.md"),
    ("A entrega expressa é grátis?",                         "entregas.md"),
    ("Tem desconto pagando no Pix?",                         "pagamentos.md"),
    ("Aceitam cartão de débito?",                            "pagamentos.md"),
    ("Quantas horas dura a bateria do Conecta Pro X?",        "manual_fone_conecta_pro_x.md"),
    ("Posso nadar com o fone?",                              "manual_fone_conecta_pro_x.md"),
    ("Vocês atendem no domingo?",                            "institucional_atendimento.md"),
    ("Vocês têm loja física?",                               "institucional_atendimento.md"),
]

def hit_rate(k):
    acertos = 0
    for pergunta, fonte_esperada in teste:
        fontes = [r['fonte'] for r in buscar(pergunta, k=k)]
        acertos += fonte_esperada in fontes
    return acertos / len(teste)

for k in [1, 3, 5]:
    print(f'Hit rate @{k}: {hit_rate(k):.0%}')
```

```python
# Célula 11 — Onde a busca errou no top-1?
for pergunta, esperado in teste:
    top = buscar(pergunta, k=1)[0]
    if top['fonte'] != esperado:
        print(f'❌ {pergunta}\n   esperado: {esperado} | encontrado: {top["fonte"]} ({top["score"]:.2f})')
```

> 🗣️ **Discussão:** vale mais a pena enviar 1 chunk (prompt curto, risco de faltar informação) ou 5 chunks (mais contexto, prompt maior, mais "ruído") para a LLM?

---

## 🏋️ Exercícios

1. **Perguntas difíceis:** crie 5 perguntas usando sinônimos e linguagem informal (ex.: *"rola parcelar em 12x?"*) e verifique se a busca acerta.
2. **Chunking × busca:** volte à Aula 2, gere os chunks com `tamanho_max=250`, refaça o índice e compare o hit rate @1 e @3.
3. **Outro modelo de embeddings:** teste `intfloat/multilingual-e5-small`. Esse modelo exige prefixos: `"query: "` na pergunta e `"passage: "` nos chunks. O hit rate melhora?
4. **Limiar de similaridade:** busque *"vocês vendem geladeira?"* e *"qual a capital da França?"*. Qual é o score do primeiro resultado? Proponha um limiar abaixo do qual o sistema deveria responder "não encontrei".
5. **Desafio — busca híbrida:** combine o score do TF-IDF com o dos embeddings (ex.: `0.3 * tfidf + 0.7 * embeddings`) e verifique se o hit rate melhora para perguntas com nomes próprios, como *"Conecta Pro X"*.

---

## ✅ Checklist de saída

- [ ] Sei explicar embedding e similaridade de cosseno
- [ ] Mostrei um caso em que embeddings acertam e o TF-IDF falha
- [ ] Arquivo `indice.faiss` salvo no Drive
- [ ] Módulo `busca.py` salvo e funcionando
- [ ] Calculei o hit rate @1, @3 e @5 e analisei os erros

**Próxima aula:** vamos ligar a busca à LLM, montar o pipeline RAG completo com citação de fontes e publicar o assistente em uma interface web.
