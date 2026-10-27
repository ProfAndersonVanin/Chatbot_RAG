# 🧠 Projeto: Assistente Virtual com RAG e LLM Gratuita

**Empresa fictícia:** TechStore Conecta — loja online de eletrônicos (a mesma do projeto de chatbot com PLN clássico)
**Ambiente:** Google Colab com GPU T4 gratuita
**Duração:** 4 aulas (sugestão de ~100 min cada)
**Nível:** intermediário em Python

---

## 🎯 O que vamos construir

No projeto anterior, o chatbot só sabia responder ao que estava cadastrado como **intenção**. Agora vamos construir um assistente que **lê os documentos da empresa** (políticas, manuais, regulamentos) e responde a perguntas livres usando uma **LLM (Large Language Model)** de código aberto, rodando de graça no Colab.

A técnica se chama **RAG — Retrieval-Augmented Generation** (Geração Aumentada por Recuperação): antes de responder, o sistema **busca** os trechos relevantes nos documentos e os entrega à LLM como contexto.

```
                    ┌────────────── INDEXAÇÃO (feita uma vez) ──────────────┐
  Documentos  ──►   │  Leitura  ──►  Chunking  ──►  Embeddings  ──►  FAISS  │
  da empresa        └───────────────────────────────────────────────┬───────┘
                                                                     │
                    ┌────────────── CONSULTA (a cada pergunta) ──────▼───────┐
  Pergunta    ──►   │  Embedding da pergunta ──► Busca dos k trechos mais    │
  do cliente        │  parecidos ──► Prompt (instruções + contexto +         │
                    │  pergunta) ──► LLM ──► Resposta com fontes             │
                    └────────────────────────────────────────────────────────┘
```

## 🗂️ Estrutura das aulas

| Aula | Arquivo | Tema | Entrega |
|------|---------|------|---------|
| 1 | [`aula01_llm_gratuita_no_colab.md`](aula01_llm_gratuita_no_colab.md) | LLMs, prompts e o problema da alucinação | `llm.py` |
| 2 | [`aula02_base_documental_e_chunking.md`](aula02_base_documental_e_chunking.md) | Documentos da empresa e divisão em trechos (chunks) | `documentos/`, `chunking.py`, `chunks.json` |
| 3 | [`aula03_embeddings_e_busca_vetorial.md`](aula03_embeddings_e_busca_vetorial.md) | Embeddings, busca semântica e índice vetorial FAISS | `indice.faiss`, `busca.py` |
| 4 | [`aula04_pipeline_rag_e_interface.md`](aula04_pipeline_rag_e_interface.md) | Pipeline RAG completo, avaliação e interface web | Assistente funcionando no navegador |

## 🧰 Recursos utilizados (todos gratuitos)

| Recurso | Para quê |
|---------|----------|
| Google Colab (GPU T4) | Executar a LLM e os notebooks |
| Google Drive | Guardar os arquivos entre as aulas |
| **Qwen2.5-1.5B-Instruct** (Hugging Face) | LLM de código aberto, boa em português, cabe na GPU gratuita |
| **sentence-transformers** | Gerar embeddings multilíngues (modelo `paraphrase-multilingual-MiniLM-L12-v2`) |
| **FAISS** | Índice vetorial para busca rápida por similaridade |
| transformers / accelerate | Carregar e executar a LLM |
| pypdf | Ler PDFs (extensão opcional) |
| Gradio | Interface de chat no navegador |
| *Opcional:* Google Gemini API | Alternativa via API com plano gratuito (exige chave do Google AI Studio) |

> 💡 **Por que não usar LangChain ou LlamaIndex?** Esses frameworks são ótimos, mas escondem as etapas. Aqui cada peça do RAG é escrita "na mão", em poucas linhas, para que os alunos entendam o que acontece por dentro. No fechamento, apresentamos os frameworks como próximo passo.

## 📁 Estrutura de pastas no Google Drive

```
MyDrive/
└── chatbot_rag/
    ├── llm.py                 ← Aula 1
    ├── documentos/            ← Aula 2  (arquivos .md da empresa)
    ├── chunking.py            ← Aula 2
    ├── chunks.json            ← Aula 2
    ├── indice.faiss           ← Aula 3
    ├── busca.py               ← Aula 3
    └── log_rag.csv            ← Aula 4
```

## ⚠️ Antes de cada aula

Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4**.
O download da LLM (~3 GB) leva de 1 a 3 minutos na primeira execução de cada sessão.

## ✅ Pré-requisitos dos alunos

- Conta Google
- Python: listas, dicionários, funções e leitura de arquivos
- Desejável: ter feito o projeto de chatbot com PLN clássico (TF-IDF e classificação de intenções)

## 🧭 Competências trabalhadas

- Funcionamento de LLMs: tokens, janela de contexto, temperatura, papéis de mensagem
- Engenharia de prompt e controle de alucinação
- Preparação de bases documentais e estratégias de chunking
- Embeddings e similaridade de cosseno
- Bancos/índices vetoriais
- Montagem de um pipeline RAG com citação de fontes
- Avaliação de recuperação (hit rate) e de respostas
- Comparação entre LLM local e LLM via API

## 📝 Sugestão de avaliação

| Critério | Peso |
|----------|------|
| Base documental própria (mínimo 5 documentos) e estratégia de chunking justificada | 20% |
| Índice vetorial e avaliação da recuperação (hit rate) | 25% |
| Pipeline RAG respondendo com fontes e recusando perguntas fora do escopo | 30% |
| Interface Gradio funcional | 10% |
| Relatório: comparação com o chatbot de intenções, limitações e riscos | 15% |

**Projeto final sugerido:** cada grupo escolhe uma organização real ou fictícia (secretaria da escola com o regulamento escolar, uma clínica, uma prefeitura com a carta de serviços…) e cria um assistente RAG sobre os documentos dela.

## 🔄 Comparando os dois projetos

| | Chatbot de intenções (PLN clássico) | Assistente RAG + LLM |
|---|---|---|
| Conhecimento | Respostas escritas manualmente | Extraído dos documentos |
| Perguntas novas | Só se parecidas com os exemplos | Responde a perguntas livres |
| Atualização | Editar intenções e retreinar | Trocar documentos e reindexar |
| Hardware | CPU comum | GPU (ou API externa) |
| Risco | Responder a intenção errada | Alucinar se o prompt e a busca forem ruins |
| Controle | Total | Depende das instruções e do contexto |
