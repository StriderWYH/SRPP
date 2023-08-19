import gymnasium as gym

from stable_baselines3 import SAC

def main():
    env = gym.make("Pendulum-v1", render_mode="human")

    model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./sac_pendulum-v1_tensorboard/",
                _init_setup_model=True)
    model.learn(total_timesteps=10000, log_interval=4,tb_log_name="first_run")
    # model.save("sac_pendulum")

    # del model # remove to demonstrate saving and loading

    model = SAC.load("sac_pendulum")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated,_,info = env.step(action)
        if terminated :
            obs, info = env.reset()

if __name__ == '__main__':
    main()