import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--train_ac', action='store_true', help='wheher train Actor Critic')
    parser.add_argument('--train_pgc', action='store_true', help='wheher train PG on cart')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--save_summary_path', type=str, default = "pg_summary/", help='')
    parser.add_argument('--save_network_path', type=str, default = "saved_pg_networks/", help='')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        agent.train()

    if args.test_pg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)

    # Experiment on Cartpole only, test unsupported
    if args.train_ac:
        env_name = args.env_name or 'CartPole-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_actorcritic import Agent_ActorCritic
        agent = Agent_ActorCritic(env, args)
        agent.train()
    if args.train_pgc:
        env_name = args.env_name or 'CartPole-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_pg_cart import Agent_PGC
        agent = Agent_PGC(env, args)
        agent.train()


if __name__ == '__main__':
    args = parse()
    run(args)
