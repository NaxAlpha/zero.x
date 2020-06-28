import math
import random

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from rl import RLAI
from maths import Vec3
from introp import game_state_to_torch_state, torch_action_to_game_action
from rlbot.setup_manager import SetupManager


class Timer:

    def __init__(self):
        self.old = -1
        self.time = 0.0

    def __call__(self, current_time):
        if self.old == -1:
            self.old = current_time
        dt = current_time - self.old 
        self.old = current_time
        self.time += dt
        return self.time

    def reset(self):
        self.old = -1
        self.time = 0.0


class TrainController:
    
    def __init__(self, my_team):
        self.my_team = my_team
        self.reward = 0
        self.goals = {0: 0, 1: 0}
        self.done = True
        self.timer = Timer()

    def update(self, state: GameTickPacket):
        self.done = False
        self.reward = 0
        
        for t in range(2):
            if self.goals[t] != state.teams[t].score:
                self.reward = 1
                if t != self.my_team:
                    self.reward *= -10
                self.done = True
                self.timer.reset()
            self.goals[t] = state.teams[t].score
            
        if self.timer(state.game_info.seconds_elapsed) > 5:
            self.done = True
            self.reward = 10
            self.timer.reset()
            

class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.ai = RLAI()
        self.action = None
        self.controller = TrainController(team)
        self.wait = False
        self.timer = Timer()
        self.reward = 0
        self.speed = 5

    def reset(self):
        ball_start = Vec3(
            y=random.randint(-1365, 1365),
            x=random.randint(-2560, -1280),
            z=random.randint(50, 200),
        )
        ball_end = Vec3(
            x=random.randint(-880, 880),
            y=random.randint(-5125, -5120),
            z=random.randint(50, 200),
        )
        ball_speed = (ball_end-ball_start).normalized() * random.randint(2000, 2500)
        car_start = Vec3(
            x=random.randint(-440, 440),
            y=random.randint(-5560, -5120),
            z=50,
        )
        car_rot = Rotator(
            pitch=0,
            yaw=math.pi/2 + (random.random()-0.5)*math.pi/6,
            roll=0,
        )
        self.set_game_state(GameState(
            ball=BallState(physics=Physics(
                location=ball_start.to_vector3(),
                velocity=ball_speed.to_vector3(),
            )),
            cars={0: CarState(physics=Physics(
                location=car_start.to_vector3(),
                rotation=car_rot,
                velocity=Vector3(0, 0, 0),
                angular_velocity=Vector3(0, 0, 0)
            ), 
                boost_amount=random.randint(60, 90),
            )},
            game_info=GameInfoState(
                game_speed=self.speed
            )
        ))

    def get_reward(self):
        return self.controller.done, self.controller.reward

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.wait:
            if self.timer(packet.game_info.seconds_elapsed) < 3.1:
                self.set_game_state(GameState(game_info=GameInfoState(game_speed=2*self.speed)))
                return SimpleControllerState()
            self.reset()
            self.wait = False
            self.timer.reset()
            self.controller.timer.reset()
            return SimpleControllerState()

        self.controller.update(packet)

        self.state = game_state_to_torch_state(packet, self.index)

        done, reward = self.get_reward()
        if self.action is not None:
            self.action = self.ai.run_step(self.state, reward, done)
        else:
            self.action = self.ai.init_run(self.state)

        self.reward += reward
        if done:
            if reward < 0:
                self.send_quick_chat(False, QuickChatSelection.Apologies_Whoops)
                self.wait = True
            else:
                self.send_quick_chat(False, QuickChatSelection.Compliments_WhatAPlay)
                self.reset()
            
            print('Total Reward:', self.reward, 'Replay Size:', len(self.ai.replay))
            self.ai.save('model.pt')
            self.reward = 0
        
        return torch_action_to_game_action(self.action)

