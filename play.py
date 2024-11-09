import retro
import numpy as np
import neat
import cv2
import pickle
import time


# Criar o ambiente
env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)

# Carregar o arquivo 'winner.pkl'
with open('winner.pkl', 'rb') as f:
    winner = pickle.load(f)

# Criar a rede neural a partir do melhor genoma
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')
net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)

# Função para jogar o jogo com a rede neural carregada
def play_game():
    ob = env.reset()
    inx, iny, inc = env.observation_space.shape
    
    # Redimensionar a entrada da imagem
    inx = int(inx / 8)
    iny = int(iny / 8)
    
    done = False
    fitness_current = 0
    xpos_max = 0
    frame = 0
    time_ini = time.time()

    while not done:
        env.render()
        frame += 1
        
        # Processamento da imagem (como no código original)
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))

        # A rede neural gera uma ação com base na imagem
        imgarray = ob.flatten()
        nnOutput = net.activate(imgarray)
        
        # Passa a ação para o ambiente
        ob, rew, done, info = env.step(nnOutput)
        
        xpos = info['xpos']
        end = info['end']
        
        if xpos > xpos_max:
            fitness_current += 1
            xpos_max = xpos

        if end == 1:
            fitness_current += 100000
            done = True
            time_end = str(time.time() - time_ini)
            print("Tempo da run: ", time_end)
            fitness_current -= float(time_end)
        
        # Exibir o score atual
        print("Score: ", fitness_current)

    print("Fim do jogo!")
    env.close()

# Chamar a função para jogar o jogo com o melhor genoma encontrado
play_game()

