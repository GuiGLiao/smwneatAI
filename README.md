Implementação do NEAT-Python e Redes neurais para treinamento de uma IA capaz de finalizar a fase Yoshi Island 2 do jogo Super Mario World (SNES).

Arquivos:
  - **config-feedforward:** Configurações do algoritmo genético para treinamento em *train.py*
  - **RandomAgent.py:** Executa um agente aletório sem treinamento
  - **new_train.py:** Uso do NEAT-Python e Redes Neurais para treinamento do agente
  - **play.py:** Programa para execução do arquivo que contêm o melhor agente
  - **winner.pkl:** Arquivo serializado do melhor agente
  - **train.py:** Executa o treinamento a partir do checkpoint-26, no qual foi encontrado o melhor agente
  - **requirements.txt:** Bibliotecas utilizadas para execução dos arquivos

Pastas:
  - **data_files:**
    - data.json: Elementos do jogo com seus endereços de memória, utilizados como parâmetros de recompensa durante o treinamento
    - scenario.json = Parâmetros de finalização e recompensa
    - YoshiIsland2.state = Fase que será jogada
  - **generation_report:**
    - generation(x).png: Relatórios das gerações 12 até 26, contendo informações sobre as gerações
    - bestIndividual.png: Mensagem de melhor agente encontrado e exportado para o *winner.pkl*
  - **marioVideos:** Registros do *train.py* e *play.py*
  - **neat-checkpoints:** Registro dos pontos de parada de cada geração. (Apenas o 26 disponivel, por ser a geração que atinge o objetivo de criar o melhor agente)

Bibliotecas usadas no train.py:
  - import retro  # Biblioteca gym-retro
  - import numpy as np  # Biblioteca para manipulação de arrays e operações matemáticas
  - import neat  # Biblioteca para aprendizado evolutivo e NEAT
  - import cv2  # Biblioteca de processamento de imagens e vídeo
  - import pickle  # Biblioteca para serialização de objetos
  - import imgarray  # Biblioteca para manipulação de arrays de imagens
  - import time  # Biblioteca para manipulação de tempo
  - import os # Biblioteca para funcionalidades do sistema operacional
    
Instalação:

  - É necessário utilizar a versão 3.8.x do Python, então é recomendado usar o Pyenv (tutorial de instalação na seção de "*Referências e Links úteis*")
  - Após configurar o pyenv e ativar o ambiente virtual (é necessário clonar o repositório e baixar as bibliotecas necessárias para executar os arquivos):

```
git clone https://github.com/itseriqq/super_mario_ia.git
```

```
pip install -r requirements.txt
```
  - Baixe a ROM do jogo Super Mario World (SNES) e copie o arquivo .sfc para .venv/lib/python3.8/site-packages/retro/data/stable/SuperMarioWorld-Snes
  - Substitua os arquivos presentes na pasta acima pelos que estão em **data_files** no diretório do projeto
  - Importe o jogo (path pode ser diferente caso tenho usado outro nome para o venv):
```
python -m retro.import .venv/lib/python3.8/site-packages/retro/data/stable/SuperMarioWorld-Snes
```

Exemplos de execução:

```
python fileName.py
```

  - RandomAgent.py
```
python RandomAgent.py
```
  - new_train.py
```
python new_train.py
```
  - train.py
```
python train.py
```
  - play.py
```
python play.py
```

Referências e links úteis:
  - NEAT-Python documentation: https://neat-python.readthedocs.io/en/latest/installation.html
  - OpenAI Retro repository: https://github.com/openai/retro
  - OpenAI Retro documentation: https://retro.readthedocs.io/en/latest/getting_started.html
  - OpenAI game integration tool tutorial (usado para determinar os endereços de memória úteis): https://www.youtube.com/watch?v=lPYWaUAq_dY
  - Pyenv tutorial: https://k0nze.dev/posts/install-pyenv-venv-vscode/
  - Pyenv vídeo: https://www.youtube.com/watch?v=1Zgo8M9yUtM
