from __future__ import division
from setproctitle import setproctitle as ptitle
import time
import logging
import torch
from PIL import Image

from A3C_CNNLSTM.environment import atari_env
from A3C_CNNLSTM.utils import setup_logger
from A3C_CNNLSTM.model import A3Clstm
from A3C_CNNLSTM.player_util import Agent




def test(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args['gpu_ids'][-1]
    log = {}
    setup_logger('{}_log'.format(args['env']), r'{0}{1}_log'.format(
        args['log_dir'], args['env']))
    log['{}_log'.format(args['env'])] = logging.getLogger('{}_log'.format(
        args['env']))
    for k in args.keys():
        log['{}_log'.format(args['env'])].info('{0}: {1}'.format(k, args[k]))

    torch.manual_seed(args['seed'])
    if gpu_id >= 0:
        torch.cuda.manual_seed(args['seed'])
    env = atari_env(args['env'], env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    t = 0

    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

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

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
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
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args['save_model_dir'], args['env']))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        args['save_model_dir'], args['env']))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
