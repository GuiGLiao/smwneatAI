Implementacao do NEAT-Python e Redes neurais para treinamento de uma IA capaz de finalizar a fase Yoshi Island 2 do jogo Super Mario World (SNES).

Arquivos:
  - config-feedforward: Configuracoes do algoritmo genetico para treinamento em *train.py*
  - train.py: Uso do NEAT-Python e Redes Neurais para treinamento do agente
  - play.py: Programa para execucao do arquivo que contem o melhor agente
  - winner.pkl: Arquivo serializado do melhor agente

Pastas:
  - data_files:
    - data.json: Elementos do jogo com seus enderecos de memoria, utilizados como parametros de recompensa durante o treinamento
    - scenario.json = Parametros de finalizacao e recompensa
    - YoshiIsland2.state = Fase que sera jogada
  - generation_report:
    - generation(x).png: Relatorios das geracoes 12 ate 26, contendo informacoes sobre as geracoes
    - bestIndividual.png: Mensagem de melhor agente encontrado e exportado para o *winner.pkl*
  - marioVideos: Videos do *train.py* e *play.py*
  - neat-checkpoints: Registro dos pontos de parada de cada geracao, ate a geracao 28. (Melhor agente encontrado do neat-checkpoint-26 para o 27)
  - checkpoints_dir: Registro das geracoes seguintes ate estagnacao e extincao de especies (checkpoints 29-110)

Bibliotecas usadas no treinamento em si:
  - import retro  # Biblioteca gym-retro
  - import numpy as np  # Biblioteca para manipulação de arrays e operações matemáticas
  - import neat  # Biblioteca para aprendizado evolutivo e NEAT
  - import cv2  # Biblioteca de processamento de imagens e vídeo
  - import pickle  # Biblioteca para serialização de objetos
  - import imgarray  # Biblioteca para manipulação de arrays de imagens
  - import time  # Biblioteca para manipulação de tempo
  - import os # Biblioteca para funcionalidades do sistema operacional
    
Instalacao:
  - OpenAI retro: https://github.com/openai/retro | pip3 install gym-retro | https://retro.readthedocs.io/en/latest/getting_started.html
  - numpy: pip install numpy
  - NEAT-Python: pip install neat-python | https://neat-python.readthedocs.io/en/latest/installation.html
  - cv2: pip install opencv-python
  - pickle: pip install pickle5
  - imgarray: pip install imgarray

    
Referencias e links uteis:
  - NEAT-Python documentation: https://neat-python.readthedocs.io/en/latest/
  - OpenAI Retro repository: https://github.com/openai/retro
  - OpenAI game integration tool tutorial (usado para determinar os enderecos de memoria uteis): https://www.youtube.com/watch?v=lPYWaUAq_dY
