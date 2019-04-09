from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from PIL import Image
import time
import logging

from A3C_Cnt.env import create_env
from A3C_Cnt.utils import setup_logger, mkdir
from A3C_Cnt.model import A3C_CONV, A3C_MLP
from A3C_Cnt.player_util import Agent



def test(args, shared_model):

    ptitle('Test Agent')
    gpu_id = args['gpu_ids'][-1]
    reward_history = []
    log = {}

    setup_logger('{}_log'.format(args['env']),
                 r'{0}{1}_log'.format(args['log_dir'], args['env']))
    log['{}_log'.format(args['env'])] = logging.getLogger(
        '{}_log'.format(args['env']))

    for k in args.keys():
        log['{}_log'.format(args['env'])].info('{0}: {1}'.format(k, args[k]))

    torch.manual_seed(args['seed'])
    if gpu_id >= 0:
        torch.cuda.manual_seed(args['seed'])
    env = create_env(args['env'], args['stack_frames'])
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id

    if args['model'] == 'MLP':
        player.model = A3C_MLP(
            player.env.observation_space.shape[0], player.env.action_space, args['stack_frames'])
    if args['model'] == 'CONV':
        player.model = A3C_CONV(args['stack_frames'], player.env.action_space)

    player.state = player.env.reset()
    if args['render'] == True:
        player.env.render()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    player.model.eval()
    max_score = 0
    t = 0
    testing = True

    while testing:
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())

        player.action_test()
        if args['render'] == True:
            player.env.render()

        if args['save_gif'] == True:
            mkdir('gif', '1')
            img = player.env.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.save('gif/1/{}.jpg'.format(t))
            t += 1

        reward_sum += player.reward
        reward_history.append(reward_sum)

        if len(reward_history) > 2000:
            reward_history.pop(0)

        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = np.mean(reward_history[-500:])
            log['{}_log'.format(args['env'])].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if args['save_max'] and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(args['save_model_dir'], args['env']))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(args['save_model_dir'], args['env']))

            if reward_mean >= args['threshold']:
                testing = False
            else:
                reward_sum = 0
                player.eps_len = 0
                state = player.env.reset()
                time.sleep(60)
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
