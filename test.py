import argparse
import numpy as np
from environment import Environment

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="PG & Actor-Critic")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--test_pg_model_path', type=str, default = "pretrained_model/pg_model.h5", help='')
    parser.add_argument('--save_summary_path', type=str, default = "pg_summary/", help='')
    parser.add_argument('--save_network_path', type=str, default = "saved_pg_networks/", help='')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_pg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)


if __name__ == '__main__':
    args = parse()
    run(args)
