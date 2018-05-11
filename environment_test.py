import gym
env = gym.make('Pong-v0')
env.reset()

while True:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())