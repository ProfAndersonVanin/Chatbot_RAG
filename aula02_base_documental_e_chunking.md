# Aula 2 — Base Documental e Chunking

**Projeto:** Assistente Virtual RAG da TechStore Conecta
**Duração sugerida:** ~100 min
**Entregas da aula:** pasta `documentos/`, módulo `chunking.py` e arquivo `chunks.json` no Google Drive

> 💡 Esta aula **não precisa de GPU**. Pode usar o ambiente padrão (CPU) do Colab.

---

## 🎯 Objetivos

Ao final desta aula, o aluno será capaz de:

1. Organizar uma **base documental** para um sistema RAG.
2. Ler documentos de texto (e, opcionalmente, PDF) com Python.
3. Explicar o que é **chunking** e por que ele é necessário.
4. Comparar estratégias de divisão (tamanho fixo × por parágrafos) e o papel da **sobreposição**.
5. Gerar e salvar os chunks com **metadados** (fonte e título).

## ⏱️ Roteiro da aula

| Tempo | Atividade |
|-------|-----------|
| 0–10 min | Revisão: por que precisamos buscar trechos? |
| 10–30 min | Criação da base documental da TechStore |
| 30–40 min | Leitura e exploração dos documentos |
| 40–55 min | Conceito de chunking: tamanho fixo × parágrafos |
| 55–80 min | Módulo `chunking.py` e geração dos chunks |
| 80–90 min | Análise dos chunks e salvamento |
| 90–100 min | Extensão com PDF, exercícios e checklist |

---

## ⚙️ Parte 1 — Preparação (5 min)

Crie um notebook `aula02_rag.ipynb`.

```python
# Célula 1 — Drive e pastas
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json
PASTA = '/content/drive/MyDrive/chatbot_rag'
PASTA_DOCS = f'{PASTA}/documentos'
os.makedirs(PASTA_DOCS, exist_ok=True)
sys.path.append(PASTA)
```

---

## 📄 Parte 2 — Criando a base documental (20 min)

Em uma empresa real, esses documentos já existiriam (PDFs, páginas da intranet, manuais). Aqui vamos criá-los em **Markdown**: um formato de texto simples que preserva títulos e listas.

> 🗣️ **Para a turma:** repare que há informações que **nenhuma LLM conhece**, como o programa Conecta+ e o manual do fone Conecta Pro X. São elas que vão provar que o RAG funciona.

```python
# Célula 2 — Documentos da TechStore Conecta
documentos = {

"institucional_atendimento.md": """# Institucional e Canais de Atendimento

A TechStore Conecta é uma loja online de eletrônicos fundada em 2018, com sede em São Paulo (SP). Vendemos smartphones, notebooks, fones de ouvido, acessórios e produtos de casa inteligente para todo o Brasil.

O atendimento humano funciona de segunda a sexta-feira, das 8h às 20h, e aos sábados, das 9h às 14h. Não há atendimento humano aos domingos e feriados. O assistente virtual funciona 24 horas por dia.

Canais oficiais: chat no site, WhatsApp (11) 4000-0000 e e-mail atendimento@techstoreconecta.com.br. O prazo de resposta por e-mail é de até 1 dia útil.

A TechStore Conecta não possui lojas físicas. Todas as vendas são feitas pelo site e pelo aplicativo.""",

"pagamentos.md": """# Formas de Pagamento

Aceitamos Pix, boleto bancário e cartões de crédito das bandeiras Visa, Mastercard, Elo e American Express. Não aceitamos cartão de débito.

Pagamentos via Pix têm 5% de desconto sobre o valor dos produtos (o desconto não se aplica ao frete). O código Pix expira em 30 minutos.

O cartão de crédito pode ser parcelado em até 10 vezes sem juros, com parcela mínima de R$ 50,00. Acima de 10 parcelas não há opção de parcelamento.

O boleto vence em 2 dias úteis. O pedido só é enviado após a compensação, que pode levar até 3 dias úteis.""",

"entregas.md": """# Política de Entregas

Entregamos em todo o território nacional. O prazo começa a contar após a confirmação do pagamento.

Prazos por região (em dias úteis): Sudeste, de 2 a 4 dias; Sul, de 3 a 5 dias; Centro-Oeste, de 4 a 7 dias; Nordeste, de 5 a 8 dias; Norte, de 7 a 10 dias.

O frete é grátis para compras acima de R$ 299,00. Abaixo desse valor, o frete é calculado pelo CEP no carrinho.

A Entrega Expressa está disponível apenas para as capitais da região Sudeste, com entrega em 1 dia útil para pedidos pagos até as 12h. O custo da Entrega Expressa é de R$ 29,90 e ela não é gratuita, mesmo em compras acima de R$ 299,00.

O código de rastreio é enviado por e-mail em até 24 horas após o despacho e também aparece em "Meus Pedidos". Se o entregador não encontrar ninguém, serão feitas mais duas tentativas. Após três tentativas, o produto retorna ao nosso centro de distribuição.""",

"trocas_devolucoes.md": """# Política de Trocas e Devoluções

Arrependimento: conforme o Código de Defesa do Consumidor, o cliente pode desistir da compra em até 7 dias corridos após o recebimento, sem precisar justificar. O produto deve estar sem sinais de uso e, de preferência, na embalagem original. O reembolso é integral, incluindo o frete.

Produto com defeito: o cliente tem até 30 dias corridos após o recebimento para solicitar a troca diretamente com a TechStore Conecta. Após esse prazo, a solicitação deve ser feita pela garantia do fabricante.

Produto errado ou avariado na entrega: recuse o recebimento ou registre a ocorrência em até 48 horas pelo menu "Meus Pedidos".

Como solicitar: acesse "Meus Pedidos", escolha o item e clique em "Solicitar troca/devolução". Enviaremos um código de postagem gratuito para levar o produto a uma agência dos Correios.

Reembolso: no Pix e no boleto, o valor é devolvido por transferência em até 5 dias úteis após o recebimento do produto. No cartão de crédito, o estorno aparece em até 2 faturas.""",

"garantia.md": """# Garantia

Todos os produtos vendidos pela TechStore Conecta são novos e possuem garantia legal de 90 dias mais a garantia contratual do fabricante, totalizando normalmente 12 meses. O prazo exato está indicado na página de cada produto.

Para acionar a garantia do fabricante, envie a nota fiscal e uma descrição do problema para garantia@techstoreconecta.com.br. Nossa equipe informa a assistência técnica autorizada mais próxima em até 2 dias úteis.

Garantia Estendida Conecta: pode ser contratada na compra ou em até 30 dias após o recebimento. Há duas opções: mais 12 meses, por 10% do valor do produto, ou mais 24 meses, por 18% do valor do produto. A garantia estendida não cobre quedas, contato com líquidos nem mau uso.""",

"programa_conecta_mais.md": """# Programa de Fidelidade Conecta+

O Conecta+ é o programa de fidelidade gratuito da TechStore Conecta. Basta ter cadastro no site para participar automaticamente.

Acúmulo: o cliente ganha 1 ponto a cada R$ 1,00 gasto em compras concluídas. O frete não gera pontos. Compras canceladas ou devolvidas têm os pontos estornados.

Resgate: cada 100 pontos valem R$ 5,00 de desconto em compras futuras. O resgate mínimo é de 500 pontos. Os pontos podem ser usados em conjunto com o desconto do Pix.

Validade: os pontos expiram 12 meses após a data da compra que os gerou.

Níveis: clientes que acumulam mais de 5.000 pontos em um ano passam ao nível Ouro e ganham frete grátis em qualquer compra, inclusive abaixo de R$ 299,00, durante os 12 meses seguintes.

Os pontos são pessoais e não podem ser transferidos para outra pessoa nem trocados por dinheiro.""",

"manual_fone_conecta_pro_x.md": """# Manual do Fone Bluetooth Conecta Pro X

O Conecta Pro X é o fone sem fio da marca própria da TechStore Conecta, com cancelamento ativo de ruído (ANC) e Bluetooth 5.3.

Bateria: até 8 horas de uso contínuo no fone e até 30 horas no total com o estojo de carregamento. Com o ANC ligado, a autonomia do fone cai para 6 horas. A carga rápida de 10 minutos garante 2 horas de uso. A recarga é feita por cabo USB-C.

Pareamento: com os fones no estojo, abra a tampa e mantenha o botão do estojo pressionado por 5 segundos, até a luz piscar em azul. Em seguida, selecione "Conecta Pro X" no Bluetooth do celular.

Restaurar configurações de fábrica: mantenha o botão do estojo pressionado por 15 segundos, até a luz piscar em vermelho três vezes.

Resistência: certificação IPX4, ou seja, resistente a suor e respingos. O fone NÃO é à prova d'água e não deve ser usado no banho ou na natação.

Garantia: 12 meses contra defeitos de fabricação.""",
}

for nome, conteudo in documentos.items():
    with open(f'{PASTA_DOCS}/{nome}', 'w', encoding='utf-8') as f:
        f.write(conteudo)

print('Documentos salvos:', os.listdir(PASTA_DOCS))
```

---

## 🔍 Parte 3 — Lendo e explorando os documentos (10 min)

A partir daqui, o código **não depende** de como os documentos foram criados: ele lê qualquer `.md` ou `.txt` da pasta. Assim, os alunos podem trocar os arquivos pelos de outra empresa.

```python
# Célula 3 — Ler todos os documentos da pasta
import pandas as pd

def ler_documentos(pasta):
    docs = []
    for nome in sorted(os.listdir(pasta)):
        if nome.endswith(('.md', '.txt')):
            with open(f'{pasta}/{nome}', encoding='utf-8') as f:
                docs.append({'fonte': nome, 'texto': f.read()})
    return docs

docs = ler_documentos(PASTA_DOCS)

pd.DataFrame([{'fonte': d['fonte'],
               'caracteres': len(d['texto']),
               'palavras': len(d['texto'].split()),
               'paragrafos': d['texto'].count('\n\n') + 1} for d in docs])
```

---

## ✂️ Parte 4 — O que é chunking? (15 min)

**Chunking** é dividir os documentos em **trechos menores** (chunks), que serão os itens buscados e enviados à LLM.

| Chunk muito **grande** | Chunk muito **pequeno** |
|---|---|
| Mistura vários assuntos em um só trecho | Perde o contexto ("até 30 dias" — 30 dias de quê?) |
| A busca fica menos precisa | A informação fica espalhada em vários trechos |
| Gasta mais tokens no prompt | Precisa buscar mais trechos por pergunta |

### Estratégia 1 — Tamanho fixo com sobreposição

```python
# Célula 4 — Chunking por tamanho fixo (em caracteres)
def chunks_tamanho_fixo(texto, tamanho=300, sobreposicao=50):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + tamanho])
        inicio += tamanho - sobreposicao
    return chunks

exemplo = docs[1]['texto']           # um dos documentos
for i, c in enumerate(chunks_tamanho_fixo(exemplo)):
    print(f'--- chunk {i} ---\n{c}\n')
```

> 🗣️ **Observe:** os cortes acontecem no meio de palavras e frases. A **sobreposição** (os 50 caracteres repetidos entre um trecho e o seguinte) reduz o problema de uma informação ser cortada ao meio, mas não o resolve.

### Estratégia 2 — Por parágrafos (respeitando a estrutura)

Os documentos já vêm organizados em parágrafos, cada um com uma ideia. Vamos **agrupar parágrafos** até um tamanho máximo e repetir o último parágrafo no chunk seguinte (sobreposição).

---

## 🧩 Parte 5 — O módulo `chunking.py` (25 min)

```python
%%writefile /content/drive/MyDrive/chatbot_rag/chunking.py
# Funções de limpeza e divisão de documentos em chunks
import re


def limpar_texto(texto):
    texto = texto.replace('\r', '')
    texto = re.sub(r'[ \t]+', ' ', texto)        # espaços repetidos
    texto = re.sub(r'\n{3,}', '\n\n', texto)     # linhas em branco repetidas
    return texto.strip()


def extrair_titulo(texto, padrao='Sem título'):
    """Usa a primeira linha iniciada por '#' como título do documento."""
    for linha in texto.splitlines():
        if linha.startswith('#'):
            return linha.lstrip('#').strip()
    return padrao


def dividir_em_chunks(texto, tamanho_max=500, sobreposicao=1):
    """Agrupa parágrafos até tamanho_max caracteres.
    'sobreposicao' = quantos parágrafos do chunk anterior são repetidos no próximo."""
    paragrafos = [p.strip() for p in texto.split('\n\n')
                  if p.strip() and not p.strip().startswith('#')]
    chunks, atual = [], []
    for p in paragrafos:
        if atual and len('\n\n'.join(atual + [p])) > tamanho_max:
            chunks.append('\n\n'.join(atual))
            atual = atual[-sobreposicao:] if sobreposicao else []
        atual.append(p)
    if atual:
        chunks.append('\n\n'.join(atual))
    return chunks


def processar_documento(fonte, texto, tamanho_max=500, sobreposicao=1):
    """Gera a lista de chunks com metadados.
    O título é colocado no início de cada chunk para dar contexto ao trecho."""
    texto = limpar_texto(texto)
    titulo = extrair_titulo(texto, padrao=fonte)
    resultado = []
    for i, trecho in enumerate(dividir_em_chunks(texto, tamanho_max, sobreposicao)):
        resultado.append({
            'id': f'{fonte}#{i}',
            'fonte': fonte,
            'titulo': titulo,
            'texto': f'{titulo}\n\n{trecho}',
        })
    return resultado
```

> 💡 **Por que repetir o título em cada chunk?** O trecho *"A garantia estendida não cobre quedas…"* fica mais fácil de encontrar e de entender quando começa com *"Garantia"*. Essa técnica simples é chamada de **enriquecimento de contexto** do chunk.

```python
# Célula 6 — Gerar os chunks de todos os documentos
from chunking import processar_documento

chunks = []
for d in docs:
    chunks.extend(processar_documento(d['fonte'], d['texto'], tamanho_max=500, sobreposicao=1))

print('Total de chunks:', len(chunks))
for c in chunks[:3]:
    print(f"\n🆔 {c['id']}\n{c['texto']}")
```

---

## 📊 Parte 6 — Analisando e salvando (10 min)

```python
# Célula 7 — Tamanho dos chunks
import matplotlib.pyplot as plt

df_chunks = pd.DataFrame(chunks)
df_chunks['caracteres'] = df_chunks['texto'].str.len()

df_chunks['caracteres'].plot(kind='hist', bins=15, figsize=(7, 3), color='teal')
plt.title('Distribuição do tamanho dos chunks')
plt.xlabel('Caracteres')
plt.show()

df_chunks.groupby('fonte').size().rename('qtd_chunks')
```

```python
# Célula 8 — Salvar os chunks no Drive
with open(f'{PASTA}/chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f'{len(chunks)} chunks salvos em {PASTA}/chunks.json')
```

---

## 📎 Extensão — Lendo PDFs (opcional)

Documentos reais costumam estar em PDF. Com a biblioteca `pypdf`, é possível extrair o texto e reaproveitar o mesmo pipeline.

```python
!pip install -q pypdf

from google.colab import files
from pypdf import PdfReader

enviados = files.upload()                      # escolha um PDF do seu computador
for nome in enviados:
    leitor = PdfReader(nome)
    texto = '\n\n'.join(pagina.extract_text() or '' for pagina in leitor.pages)
    novos = processar_documento(nome, texto)
    print(f'{nome}: {len(leitor.pages)} páginas -> {len(novos)} chunks')
```

> ⚠️ PDFs escaneados (imagens) não têm texto extraível; seria necessário OCR. PDFs com colunas e tabelas podem gerar texto desorganizado. **Sempre inspecione o resultado.**

---

## 🏋️ Exercícios

1. **Comparar estratégias:** gere os chunks com `tamanho_max=200`, `500` e `1000`. Quantos chunks surgem em cada caso? Leia alguns e diga qual tamanho parece ideal para esses documentos.
2. **Sobreposição:** compare `sobreposicao=0` e `sobreposicao=1`. Encontre um exemplo de informação que só fica completa com sobreposição.
3. **Novo documento:** crie `cupons.md` com as regras de um cupom de primeira compra (valor, validade, condições). Gere os chunks novamente e salve.
4. **Qualidade dos dados:** encontre no documento de pagamentos uma frase que poderia gerar dúvida para a LLM e reescreva-a de forma mais clara. Por que a qualidade dos documentos importa tanto no RAG?
5. **Desafio:** escreva uma função de chunking por **sentenças** (separando por `. `), agrupando até 3 sentenças por chunk com sobreposição de 1 sentença.

---

## ✅ Checklist de saída

- [ ] Pasta `documentos/` com 7 documentos da TechStore no Drive
- [ ] Módulo `chunking.py` salvo no Drive
- [ ] Arquivo `chunks.json` gerado e salvo
- [ ] Sei explicar os riscos de chunks muito grandes ou muito pequenos
- [ ] Sei explicar para que servem a sobreposição e os metadados

**Próxima aula:** vamos transformar cada chunk em um vetor (embedding) e construir o mecanismo de busca semântica.
