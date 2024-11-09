import retro
import numpy as np
import neat
import cv2
import pickle
import imgarray
import time


env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players = 1)


def eval_genomes(genomes, config):
	
	for genome_id, genome in genomes:
		
		ob = env.reset()
		ac = env.action_space.sample()

		inx, iny, inc = env.observation_space.shape
		
		inx = int(inx/8)
		iny = int(iny/8)
		
		net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
		
		current_max_fitness = 0
		fitness_current = 0
		frame = 0
		counter = 0
		xpos = 0
		xpos_max = 0
		endOfLevel = 0
		endOfLevel_x = 0
		end = 0
		time_ini = time.time()
		
		done = False
		#cv2.namedWindow("main", cv2.WINDOW_NORMAL)
		
		while not done:
		
			env.render()
			frame += 1
			#scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
			#scaledimg = cv2.resize(scaledimg, (iny, inx))
			ob = cv2.resize(ob, (inx, iny))
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))
			#cv2.imshow('main', scaledimg)
			#cv2.waitKey(1)
			
			imgarray = ob.flatten()
			
			nnOutput = net.activate(imgarray)
			
			
			ob, rew, done, info = env.step(nnOutput)
			
			xpos = info['xpos']
			end = info['end']
			#time_ini = time.time()
			
			if xpos > xpos_max:
				fitness_current += 1
				xpos_max = xpos
				
				
			if end == 1:
				fitness_current += 100000
				done = True
				time_end = str(time.time()-time_ini)
				print("Tempo da run: ", time_end)
				fitness_current -= float(time_end)
				time_ini = time.time()
			
			if fitness_current > current_max_fitness:
				current_max_fitness = fitness_current
				counter = 0
			else:
				counter += 1
				fitness_current -= 0.2
			if done or counter == 250:
				done = True
				time_end = str(time.time()-time_ini)
				print("Tempo da run: ", time_end)
				print("Score Total: ", fitness_current)
				fitness_current -= float(time_end)
				time_ini = time.time()
				print("ID:", genome_id, "///" ,"Score - Tempo" ,fitness_current)
				print("//////////////////////")
			genome.fitness = fitness_current
			

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')


#p = neat.Population(config)


# Carregue o checkpoint
checkpoint_file = 'neat-checkpoint-26'  # Substitua pelo nome do seu arquivo de checkpoint

p = neat.Checkpointer.restore_checkpoint(checkpoint_file)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
	pickle.dump(winner, output, 1)








