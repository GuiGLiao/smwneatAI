import retro

def main():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players = 1)
    obs = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        print(obs, rew, done, info)
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()

