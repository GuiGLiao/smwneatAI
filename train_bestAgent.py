# Importando bibliotecas necessárias
import retro  # Biblioteca gym-retro
import numpy as np  # Biblioteca para manipulação de arrays e operações matemáticas
import neat  # Biblioteca para aprendizado evolutivo e NEAT
import cv2  # Biblioteca de processamento de imagens e vídeo
import pickle  # Biblioteca para serialização de objetos
import imgarray  # Biblioteca para manipulação de arrays de imagens
import time  # Biblioteca para manipulação de tempo
import os # Biblioteca para funcionalidades do sistema operacional
from time import sleep

# Inicializando o ambiente de jogo, especificando o jogo Super Mario World (SNES) e o estado 'YoshiIsland2'
env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)


# Função para garantir que a pasta de checkpoints existe
def create_checkpoint_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# Diretório para salvar os checkpoints
checkpoint_dir = 'checkpoints_dir'  

# Crie o diretório, caso não exista
create_checkpoint_directory(checkpoint_dir)

# Função de avaliação de cada genoma
def eval_genomes(genomes, config):
    
    # Iterando sobre os genomas e seus identificadores
    for genome_id, genome in genomes:
        
        # Inicializando o ambiente de jogo
        ob = env.reset()  # Reseta o jogo e recebe o primeiro quadro da tela
        ac = env.action_space.sample()  # Ação inicial

        # Obtendo o formato da imagem de observação do ambiente
        inx, iny, inc = env.observation_space.shape
        
        # Reduzindo a resolução da imagem para acelerar o treinamento
        inx = int(inx / 8)
        iny = int(iny / 8)
        
        # Criando a rede neural a partir do genoma e da configuração
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        # Inicializando variáveis para o acompanhamento do desempenho do agente
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        endOfLevel = 0
        endOfLevel_x = 0
        end = 0
        time_ini = time.time()  # Inicializando o tempo de início da execução
        
        done = False  # Variável para indicar quando o episódio termina
        
        # Loop principal de simulação, onde o agente interage com o ambiente
        while not done:
            
            # Renderizando o ambiente para visualização (se necessário)
            env.render()
            frame += 1  # Contador de quadros

            # Processando a imagem recebida do ambiente
            ob = cv2.resize(ob, (inx, iny))  # Redimensionando a imagem
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)  # Convertendo a imagem para escala de cinza
            ob = np.reshape(ob, (inx, iny))  # Remodelando a imagem para ser compatível com a entrada da rede neural

            # Achando a saída da rede neural dada a imagem processada
            imgarray = ob.flatten()  # Achata a imagem para um vetor unidimensional
            nnOutput = net.activate(imgarray)  # Passando o vetor pela rede neural

            # Realizando a ação no ambiente com a saída da rede neural
            ob, rew, done, info = env.step(nnOutput)
            
            # Usa os parâmetros definidos no arquivo data.json
            xpos = info['xpos']
            end = info['end']

            # Atualizando a pontuação com base na posição do jogador
            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            # Caso o nível termine, adicionar uma grande recompensa e encerrar a run
            if end == 1:
                fitness_current += 100000
                done = True
                time_end = str(time.time() - time_ini)
                print("Tempo da run: ", time_end)
                fitness_current -= float(time_end)  # Penalizando o tempo
                time_ini = time.time()  # Resetando o tempo

            # Comparando o desempenho atual com o melhor desempenho até agora
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                fitness_current -= 0.2  # Penalizando por estagnar

            # Se o agente ficou parado por um certo tempo, finaliza o loop
            if done or counter == 250:
                done = True
                time_end = str(time.time() - time_ini)
                print("Tempo da run: ", time_end)
                print("Score Total: ", fitness_current)
                fitness_current -= float(time_end)  # Penalizando o tempo
                time_ini = time.time()  # Resetando o tempo
                print("ID:", genome_id, "///", "Score - Tempo", fitness_current)
                print("//////////////////////")
            
            # Atribuindo a pontuação ao genoma atual
            genome.fitness = fitness_current


# Configuração do NEAT, com parâmetros para a rede neural e evolução
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')



# Carregando o checkpoint de um treinamento anterior
checkpoint_file = 'neat_checkpoints/neat-checkpoint-26'  # Substitua pelo nome do arquivo de checkpoint
p = neat.Checkpointer.restore_checkpoint(checkpoint_file)  # Restaurando o estado do treinamento

#p = neat.Population(config)

# Adicionando relatórios para acompanhar o progresso do treinamento
p.add_reporter(neat.StdOutReporter(True))  # Relatório no console
stats = neat.StatisticsReporter()  # Relatório estatístico sobre o desempenho
p.add_reporter(stats)  # Adicionando o relatório estatístico
p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-')))

# Rodando o treinamento, passando a função de avaliação dos genomas
winner = p.run(eval_genomes)

# Salvando o melhor genoma após o treinamento
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

