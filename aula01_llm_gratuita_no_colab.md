# Aula 1 — LLM Gratuita no Colab e o Problema da Alucinação

**Projeto:** Assistente Virtual RAG da TechStore Conecta
**Duração sugerida:** ~100 min
**Entrega da aula:** módulo `llm.py` salvo no Google Drive

---

## 🎯 Objetivos

Ao final desta aula, o aluno será capaz de:

1. Explicar o que é uma LLM, o que são **tokens** e o que é **janela de contexto**.
2. Executar uma LLM de código aberto gratuitamente no Colab com GPU.
3. Usar mensagens com papéis (`system`, `user`, `assistant`) e ajustar a **temperatura**.
4. Demonstrar o problema da **alucinação** e entender por que o RAG resolve parte dele.

## ⏱️ Roteiro da aula

| Tempo | Atividade |
|-------|-----------|
| 0–15 min | Conceitos: LLM, tokens, contexto, modelos abertos × APIs |
| 15–30 min | Configuração da GPU, Drive e instalação |
| 30–45 min | Criação do módulo `llm.py` e primeira geração |
| 45–60 min | Tokens e prompt de sistema |
| 60–75 min | Temperatura: criatividade × consistência |
| 75–90 min | Alucinação e "contexto manual": a ideia do RAG |
| 90–100 min | Exercícios e checklist |

---

## 📚 Parte 1 — Conceitos (15 min)

### O que é uma LLM?

Um **modelo de linguagem grande** é uma rede neural (Transformer) treinada para **prever o próximo token** de um texto. Repetindo essa previsão muitas vezes, ela gera respostas completas.

| Conceito | Explicação | Exemplo |
|----------|-----------|---------|
| **Token** | Pedaço de texto que o modelo processa (palavra, parte de palavra ou pontuação) | `"devolução"` pode virar `["dev", "olu", "ção"]` |
| **Janela de contexto** | Quantidade máxima de tokens que o modelo "enxerga" de uma vez | Prompt + resposta precisam caber nela |
| **Parâmetros** | "Tamanho" do modelo (pesos aprendidos) | 1,5 bilhão no modelo desta aula |
| **Instruct** | Versão ajustada para seguir instruções e conversar | `Qwen2.5-1.5B-Instruct` |
| **Temperatura** | Controla a aleatoriedade das respostas | 0 = previsível; 1 = criativo |

### Modelos abertos × APIs

| | Modelo aberto no Colab (nosso caso) | API (Gemini, GPT, Claude…) |
|---|---|---|
| Custo | Grátis (usa a GPU do Colab) | Plano gratuito limitado ou pago por uso |
| Privacidade | Os dados não saem da sessão | Os dados vão para o provedor |
| Qualidade | Boa para tarefas focadas | Geralmente superior |
| Configuração | Baixar o modelo (~3 GB) | Criar uma chave de API |

### Por que o Qwen2.5-1.5B-Instruct?

É um modelo aberto e gratuito do Hugging Face, com bom desempenho em português e pequeno o bastante para a GPU T4 gratuita do Colab. Alternativas: `Qwen/Qwen2.5-0.5B-Instruct` (mais rápido, menos preciso) ou `Qwen/Qwen2.5-3B-Instruct` (melhor, mais lento).

---

## ⚙️ Parte 2 — Configurando o ambiente (15 min)

1. Crie um notebook `aula01_rag.ipynb`.
2. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4 → Salvar**.

```python
# Célula 1 — Verificar se a GPU está ativa
import torch
print('GPU disponível:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Placa:', torch.cuda.get_device_name(0))
```

```python
# Célula 2 — Instalar/atualizar bibliotecas
!pip install -q -U transformers accelerate
```

```python
# Célula 3 — Conectar ao Drive e criar a pasta do projeto
from google.colab import drive
drive.mount('/content/drive')

import os, sys
PASTA = '/content/drive/MyDrive/chatbot_rag'
os.makedirs(PASTA, exist_ok=True)
sys.path.append(PASTA)
print('Pasta do projeto:', PASTA)
```

---

## 🧩 Parte 3 — O módulo `llm.py` (15 min)

Assim como no projeto anterior, vamos guardar o código reutilizável em um módulo no Drive. Na Aula 4, ele será apenas importado.

```python
%%writefile /content/drive/MyDrive/chatbot_rag/llm.py
# Módulo para carregar e usar uma LLM aberta do Hugging Face
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELO_PADRAO = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_modelo = None


def carregar_llm(nome=MODELO_PADRAO):
    """Baixa (na primeira vez) e carrega a LLM na GPU, se disponível."""
    global _tokenizer, _modelo
    tipo = torch.float16 if torch.cuda.is_available() else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(nome)
    _modelo = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=tipo, device_map="auto")
    print(f"LLM carregada: {nome} | dispositivo: {_modelo.device}")


def contar_tokens(texto):
    if _tokenizer is None:
        carregar_llm()
    return len(_tokenizer.encode(texto))


def gerar(mensagens, max_tokens=300, temperatura=0.2):
    """Recebe uma lista de mensagens [{'role': ..., 'content': ...}] e devolve a resposta."""
    if _modelo is None:
        carregar_llm()
    texto = _tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    entradas = _tokenizer([texto], return_tensors="pt").to(_modelo.device)

    parametros = {"max_new_tokens": max_tokens, "pad_token_id": _tokenizer.eos_token_id}
    if temperatura > 0:
        parametros.update(do_sample=True, temperature=temperatura, top_p=0.9)
    else:
        parametros.update(do_sample=False)

    with torch.no_grad():
        saida = _modelo.generate(**entradas, **parametros)

    novos_tokens = saida[0][entradas["input_ids"].shape[1]:]   # descarta o prompt
    return _tokenizer.decode(novos_tokens, skip_special_tokens=True).strip()
```

> ⚠️ `%%writefile` precisa ser a **primeira linha** da célula.
> 💡 Se aparecer um aviso dizendo que `torch_dtype` está obsoleto, troque por `dtype=tipo`. É só um aviso; o código continua funcionando.

```python
# Célula 5 — Carregar a LLM e fazer a primeira pergunta
from llm import carregar_llm, gerar, contar_tokens

carregar_llm()

mensagens = [{"role": "user", "content": "Explique em duas frases o que é inteligência artificial."}]
print(gerar(mensagens))
```

---

## 🔤 Parte 4 — Tokens e prompt de sistema (15 min)

```python
# Célula 6 — Quantos tokens tem um texto?
frases = [
    "Olá!",
    "Quero devolver o produto que comprei.",
    "A TechStore Conecta oferece garantia estendida de 24 meses.",
]
for f in frases:
    print(f'{contar_tokens(f):3} tokens -> {f}')
```

> 🗣️ **Discussão:** por que contar tokens importa? (Limite da janela de contexto, tempo de resposta e, nas APIs pagas, **custo**.)

### Papéis das mensagens

| Papel | Função |
|-------|--------|
| `system` | Instruções gerais: quem o assistente é, como deve se comportar |
| `user` | Mensagem do usuário |
| `assistant` | Respostas anteriores do modelo (histórico da conversa) |

```python
# Célula 7 — O mesmo pedido com prompts de sistema diferentes
pergunta = {"role": "user", "content": "O que é um fone Bluetooth?"}

sistemas = [
    "Você é um assistente prestativo.",
    "Você é um atendente de loja. Responda em no máximo 2 frases, de forma simpática.",
    "Você explica tecnologia para crianças de 8 anos.",
]
for s in sistemas:
    print('SYSTEM:', s)
    print(gerar([{"role": "system", "content": s}, pergunta]), '\n' + '-' * 60)
```

---

## 🌡️ Parte 5 — Temperatura (15 min)

```python
# Célula 8 — Mesma pergunta, 3 vezes, com temperaturas diferentes
msg = [{"role": "user", "content": "Crie um slogan curto para uma loja de eletrônicos."}]

for temp in [0, 0.7, 1.2]:
    print(f'\n🌡️ Temperatura {temp}')
    for _ in range(3):
        print('  -', gerar(msg, max_tokens=40, temperatura=temp))
```

> 🗣️ **Discussão:** para um assistente de **atendimento**, que precisa ser consistente e fiel às políticas da empresa, qual temperatura você escolheria? E para gerar ideias de marketing?

---

## 👻 Parte 6 — Alucinação e a ideia do RAG (15 min)

A LLM aprendeu com textos públicos da internet. Ela **não conhece** as regras internas da TechStore Conecta. Veja o que acontece:

```python
# Célula 9 — Perguntando algo que o modelo NÃO tem como saber
sistema = {"role": "system", "content": "Você é o assistente virtual da loja TechStore Conecta."}
pergunta = {"role": "user", "content": "Como funciona o programa de pontos Conecta+ da loja? Quantos pontos eu ganho por real gasto?"}

print(gerar([sistema, pergunta]))
```

> 😱 O modelo provavelmente **inventou** regras com muita confiança. Isso é **alucinação**: a LLM gera um texto plausível, mas sem base nos fatos.

Agora vamos entregar a informação correta **dentro do prompt**:

```python
# Célula 10 — "RAG manual": colando o documento no prompt
regulamento = """Programa de Fidelidade Conecta+:
- O cliente ganha 1 ponto a cada R$ 1,00 gasto em compras concluídas.
- Cada 100 pontos valem R$ 5,00 de desconto em compras futuras.
- Os pontos expiram 12 meses após a data da compra.
- Clientes que acumulam mais de 5.000 pontos no ano passam ao nível Ouro e ganham frete grátis em qualquer compra."""

sistema_rag = {"role": "system", "content": (
    "Você é o assistente virtual da loja TechStore Conecta. "
    "Responda SOMENTE com base no CONTEXTO fornecido. "
    "Se a informação não estiver no contexto, diga que não sabe.")}
pergunta_rag = {"role": "user", "content": f"CONTEXTO:\n{regulamento}\n\nPERGUNTA: {pergunta['content']}"}

print(gerar([sistema_rag, pergunta_rag]))
```

```python
# Célula 11 — E se a pergunta não estiver no contexto?
pergunta_fora = {"role": "user", "content": f"CONTEXTO:\n{regulamento}\n\nPERGUNTA: Os pontos podem ser transferidos para outra pessoa?"}
print(gerar([sistema_rag, pergunta_fora]))
```

### 💡 Chegamos à ideia do RAG

Colar o documento certo no prompt funcionou. Mas a TechStore tem **dezenas de documentos**, e uma empresa real pode ter **milhares**. Não cabem todos no prompt (janela de contexto, tempo e custo).

**Solução:** para cada pergunta, **buscar automaticamente apenas os trechos relevantes** e colocá-los no prompt. É isso que vamos construir nas próximas aulas:

| Aula | Peça do RAG |
|------|-------------|
| 2 | Preparar os documentos e dividi-los em trechos |
| 3 | Encontrar os trechos certos para cada pergunta |
| 4 | Juntar tudo: busca + prompt + LLM + interface |

---

## 🏋️ Exercícios

1. **Comparação de modelos:** troque para `carregar_llm("Qwen/Qwen2.5-0.5B-Instruct")`, repita a Célula 10 e compare a qualidade e o tempo de resposta.
2. **Engenharia de prompt:** melhore o prompt de sistema da Célula 10 para que o bot sempre termine perguntando "Posso ajudar em algo mais?".
3. **Teste de fidelidade:** faça 5 perguntas diferentes sobre o regulamento Conecta+ (incluindo 2 cuja resposta **não** está no texto). Em quantas o modelo foi fiel ao contexto?
4. **Tokens:** use `contar_tokens(regulamento)` e estime quantos documentos desse tamanho caberiam em uma janela de 32 mil tokens.
5. **Desafio:** crie a função `conversar()` com um laço `input()` que mantenha o **histórico** de mensagens (adicionando as respostas com o papel `assistant`), para que o bot lembre o que foi dito antes.

---

## ✅ Checklist de saída

- [ ] GPU T4 ativada e LLM carregada
- [ ] Arquivo `llm.py` salvo no Drive
- [ ] Sei explicar tokens, janela de contexto e temperatura
- [ ] Demonstrei uma alucinação e a corrigi com contexto no prompt
- [ ] Sei explicar com minhas palavras por que precisamos de RAG

**Próxima aula:** vamos criar a base de documentos da TechStore e dividi-la em trechos (chunks).
