#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import thriftpy

rl_thrift = thriftpy.load("RL.thrift", module_name="rl_thrift")

from thriftpy.rpc import make_server

import numpy as np

from use_actor import load_model
from metadata import metadata

class Dispatcher(object):

    def __init__(self):
        # self.model = load_model({'layers': '1024, 500, 300, 100'})#TODO better system
        # self.model = load_model({'layers': '2048, 1024, 512, 256'})#TODO better system
        # self.model = load_model({'layers': '2048, 2048, 1024, 512, 256'})#TODO better system
        self.model = load_model({'layers': '2048, 2048, 1024, 1024, 1024'})#TODO better system




    def act(self, state):
        # TODO better system
        stateSerialized = [
            int(state.is_kickable),
            state.ball_pos_x,
            state.ball_pos_y,
            state.ball_vel_x,
            state.ball_vel_y,
            state.ball_pos_count,
            state.self_stamina,
            state.goalie_pos_x,
            state.goalie_pos_y,
            state.goalie_vel_x,
            state.goalie_vel_y,
            state.goalie_body,
            state.self_pos_x,
            state.self_pos_y,
            state.self_vel_x,
            state.self_vel_y,
            state.self_body,
            state.self_dash_rate,
            state.time_spent, # TODO keep ?
            state.self_unum
        ]
        action = rl_thrift.Action()
        action_choice_ev, action_parameters_ev = self.model(np.matrix(stateSerialized))
        action_choice_ev, action_parameters_ev = action_choice_ev.T, action_parameters_ev.T
        print("action_choice_ev.shape", action_choice_ev.shape)
        print("action_parameters_ev.shape", action_parameters_ev.shape)

        actions_name = ['dash', 'kick', 'move', 'turn']
        action.name = actions_name[np.argmax(action_choice_ev)]

        if action.name == 'dash':
            action.dash_power = action_parameters_ev[metadata.action_id_for_param('dash_power')]
            action.dash_dir = action_parameters_ev[metadata.action_id_for_param('dash_dir')]
        elif action.name == 'kick':
            action.kick_power = action_parameters_ev[metadata.action_id_for_param('kick_power')]
            action.kick_direction = action_parameters_ev[metadata.action_id_for_param('kick_direction')]
        elif action.name == 'move':
            action.move_x = action_parameters_ev[metadata.action_id_for_param('move_x')]
            action.move_y = action_parameters_ev[metadata.action_id_for_param('move_y')]
        elif action.name == 'turn':
            action.turn_moment = action_parameters_ev[metadata.action_id_for_param('turn_moment')]

        action.turn_neck_angle = action_parameters_ev[metadata.action_id_for_param('turn_neck_angle')]
        print(action)
        return action

    def send_state(self, state):
        pass

    def send_action(self, action):
        # print(action)
        pass

    def send_start_signal(self, cycle, gamemode, side, unum):
        pass

    def send_terminal_signal(self, cycle, gamemode, side):
        pass

dispatcher = Dispatcher()
server = make_server(rl_thrift.RL, dispatcher, unix_socket='/robocup/RL/socket')

server.serve()
