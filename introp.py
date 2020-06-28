import math
import torch
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState


def game_state_to_torch_state(game_state: GameTickPacket, self_car_idx):
    car = game_state.game_cars[self_car_idx].physics
    ball = game_state.game_ball.physics
    state = torch.tensor([
        ball.location.x/4096,
        ball.location.y/5120,
        ball.location.z/2044,
        ball.velocity.x/6000,
        ball.velocity.y/6000,
        ball.velocity.z/6000,
        car.location.x/4096,
        car.location.y/5120,
        car.location.z/2044,
        car.velocity.x/2300,
        car.velocity.y/2300,
        car.velocity.z/2300,
        ball.angular_velocity.x/6,
        ball.angular_velocity.y/6,
        ball.angular_velocity.z/6,
        car.rotation.pitch/math.pi/2,
        car.rotation.yaw/math.pi/2,
        car.rotation.roll/math.pi/2,
        car.angular_velocity.x/5.5,
        car.angular_velocity.y/5.5,
        car.angular_velocity.z/5.5,
    ])
    return state.unsqueeze(0)


def torch_action_to_game_action(torch_action):
    torch_action = torch_action[0]
    action = [
        *(torch_action[:-3]*2-1).float().tolist(),
        *(torch_action[-3:]>0.5).float().tolist(),
    ]
    return SimpleControllerState(*action)

