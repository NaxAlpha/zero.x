import math
import torch
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState


def game_state_to_torch_state(game_state: GameTickPacket, self_car_idx):
    car = game_state.game_cars[self_car_idx].physics
    ball = game_state.game_ball.physics
    state = torch.tensor([
        ball.location.x,
        ball.location.y,
        ball.location.z,
        ball.velocity.x,
        ball.velocity.y,
        ball.velocity.z,
        car.location.x,
        car.location.y,
        car.location.z,
        car.velocity.x,
        car.velocity.y,
        car.velocity.z,
        ball.rotation.pitch,
        ball.rotation.yaw,
        ball.rotation.roll,
        ball.angular_velocity.x,
        ball.angular_velocity.y,
        ball.angular_velocity.z,
        car.rotation.pitch,
        car.rotation.yaw,
        car.rotation.roll,
        car.angular_velocity.x,
        car.angular_velocity.y,
        car.angular_velocity.z,
    ])
    state[0:12:3] /= 4096
    state[1:12:3] /= 5120
    state[2:12:3] /= 2044
    state[20:] /= math.pi/2
    return state.unsqueeze(0)


def torch_action_to_game_action(torch_action):
    torch_action = torch_action[0]
    action = [
        *(torch_action[:-3]*2-1).float().tolist(),
        *(torch_action[-3:]>0.5).float().tolist(),
    ]
    return SimpleControllerState(*action)

