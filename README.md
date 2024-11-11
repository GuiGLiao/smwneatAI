Implementação do NEAT-Python e Redes neurais para treinamento de uma IA capaz de finalizar a fase Yoshi Island 2 do jogo Super Mario World (SNES).

Arquivos:
  - config-feedforward: Configurações do algoritmo genético para treinamento em *train.py*
  - RandomAgent.py: Executa um agente aletório sem treinamento
  - train.py: Uso do NEAT-Python e Redes Neurais para treinamento do agente
  - play.py: Programa para execução do arquivo que contêm o melhor agente
  - winner.pkl: Arquivo serializado do melhor agente

Pastas:
  - data_files:
    - data.json: Elementos do jogo com seus endereços de memória, utilizados como parâmetros de recompensa durante o treinamento
    - scenario.json = Parâmetros de finalização e recompensa
    - YoshiIsland2.state = Fase que será jogada
  - generation_report:
    - generation(x).png: Relatórios das gerações 12 até 26, contendo informações sobre as gerações
    - bestIndividual.png: Mensagem de melhor agente encontrado e exportado para o *winner.pkl*
  - marioVideos: Vídeos do *train.py* e *play.py*
  - neat-checkpoints: Registro dos pontos de parada de cada geração, até a geracao 28. (Melhor agente encontrado do neat-checkpoint-26 para o 27)

Bibliotecas usadas no treinamento em si:
  - import retro  # Biblioteca gym-retro
  - import numpy as np  # Biblioteca para manipulação de arrays e operações matemáticas
  - import neat  # Biblioteca para aprendizado evolutivo e NEAT
  - import cv2  # Biblioteca de processamento de imagens e vídeo
  - import pickle  # Biblioteca para serialização de objetos
  - import imgarray  # Biblioteca para manipulação de arrays de imagens
  - import time  # Biblioteca para manipulação de tempo
  - import os # Biblioteca para funcionalidades do sistema operacional
    
Instalação:

  - OpenAI retro: pip3 install gym-retro
  - numpy: pip install numpy
  - NEAT-Python: pip install neat-python
  - cv2: pip install opencv-python
  - pickle: pip install pickle5
  - imgarray: pip install imgarray

    
Referencias e links uteis:
  - NEAT-Python documentation: https://neat-python.readthedocs.io/en/latest/installation.html
  - OpenAI Retro repository: https://github.com/openai/retro
  - OpenAI Retro documentation: https://retro.readthedocs.io/en/latest/getting_started.html
  - OpenAI game integration tool tutorial (usado para determinar os enderecos de memoria uteis): https://www.youtube.com/watch?v=lPYWaUAq_dY
