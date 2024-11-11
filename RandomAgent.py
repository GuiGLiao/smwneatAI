import retro  # Importa a biblioteca 'retro'

def main():
    # Cria o ambiente do jogo utilizando a função 'retro.make()'.
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    
    # Inicializa o ambiente e obtém a primeira observação (imagem) do jogo.
    obs = env.reset()
    
    # Inicia um loop infinito para jogar continuamente até que o jogo seja finalizado.
    while True:
        env.render()  # Exibe o ambiente do jogo em uma janela.
        
        # Gera uma ação aleatória dentro do espaço de ações permitido pelo jogo.
        action = env.action_space.sample()
        
        # Executa a ação no ambiente e retorna:
        # obs: Nova observação (imagem) após a ação.
        # rew: Recompensa recebida após executar a ação.
        # done: Booleano indicando se o jogo terminou.
        # info: Informações adicionais (como pontuação, tempo, etc).
        obs, rew, done, info = env.step(action)
        
        # Imprime os valores retornados (estado atual, recompensa, se o jogo terminou e informações).
        print(obs, rew, done, info)
        
        # Se o jogo terminar (done == True), o ambiente é reiniciado para começar novamente.
        if done:
            obs = env.reset()
    
    # Fecha o ambiente
    env.close()


if __name__ == "__main__":
    main()

