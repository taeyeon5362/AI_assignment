import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import copy
import utils
import time
from plot_result import plot_result

args = utils.get_config()
save_dir = "Result/{}".format(args.RESULT_SAVE)
utils.remove_reuslt_files("{}/".format(save_dir))

# ## DQN 에이전트
class DQNAgent:
    def __init__(self, datas):
        self.model = nn.Sequential(
            nn.Linear(4, datas[0]),
            nn.ReLU(),
            nn.Linear( datas[0], 2)
        )

        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), args.LR)
        self.steps_done = 0
        self.memory = deque(maxlen=args.BUFFER_SIZE)
        self.TARGET_UPDATE = datas[2]
        self.BATCH_SIZE = datas[1]

    # 1, 이 메소드의 역할은 ? ->
    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))

    # 2, 이 메소드의 역할은 ? ->
    def act(self, state):
        eps_threshold = args.EPS_END + (args.EPS_START - args.EPS_END) * math.exp(-1. * self.steps_done / args.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    # 3, 이 메소드의 역할은 ? ->
    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)

        if args.USE_TARGET_NET:
            max_next_q = self.target_model(next_states).detach().max(1)[0]
        else:
            max_next_q = self.model(next_states).detach().max(1)[0]

        expected_q = rewards + (args.GAMMA * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 4, 이 메소드의 역할은 ? ->
    def update_target(self, total_steps):
        if args.USE_TARGET_NET is not True:
            return

        if total_steps % self.TARGET_UPDATE == 0 and total_steps is not 0:
            self.target_model.load_state_dict(self.model.state_dict())


env = gym.make('CartPole-v0')
total_episode = 0
time_list = []

for index in range(3):
    start = time.time()

    new_args = utils.get_refine_args(args, index)
    test_object = utils.get_test_object(args)

    for index2 in range(3):
        agent = DQNAgent(new_args)
        score_history = []
        total_steps = 0

        for e in range(1, args.EPISODES+1):
            state = env.reset()
            steps = 0

            while True:
                if args.RENDER:
                    env.render()
                state = torch.FloatTensor([state])
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action.item())

                # 게임이 끝났을 경우 마이너스 보상주기
                if done:
                    reward = -1

                agent.memorize(state, action, reward, next_state)
                agent.learn()
                agent.update_target(total_steps)

                state = next_state
                steps += 1
                total_steps += 1

                if done:
                    total_episode += 1
                    progress_ratio = (total_episode / (args.EPISODES * 3 * 3)) * 100

                    print("{0}-{1} 에피소드:{2} 점수: {3} .... 실험 진행도 : {4}%".format(index + 1,index2 + 1, e, steps, round(progress_ratio, 2)))
                    score_history.append(steps)

                    with open("{0}/{1}_{2}_{3}.txt".format(save_dir, test_object[index], args.RESULT_SAVE, index2 + 1), "a") as f:
                        f.write(str(steps) + "\n")
                    break

    time_list.append(utils.calc_exp_times(time.time() - start))


print("실험 종료!!!")
utils.print_exp_times(time_list, args)

plot_result()