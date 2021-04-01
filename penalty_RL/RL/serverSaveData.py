#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import thriftpy

rl_thrift = thriftpy.load("RL.thrift", module_name="rl_thrift")

from thriftpy.rpc import make_server

from M import *

import numpy as np

class Dispatcher(object):
    def __init__(self):
        self.M = M()

        self.our_score = 0
        self.n_penalty = 0

        self.episode_ongoing = False

        self.our_side = ''
        self.current_cycle = -1
        self.total_cycle = 0


    def send_state(self, state):
        if state.cycle < self.current_cycle:
            self.M.process(flush=True)
            self.total_cycle += self.current_cycle
            print("flush", self.current_cycle, self.total_cycle)
        # if state.self_unum == self.M.get_taker(state.cycle):
        #     print("State: {} by {}".format(state.cycle, state.self_unum))
        self.M.add_state(state)
        self.M.process()
        self.current_cycle = state.cycle
        # print("self.current_cycle", self.current_cycle)


    def send_action(self, action):
        self.M.add_action(action)
        # if action.name == 'dash' and action.unum == self.M.get_taker(action.cycle):
        #     print("{} by {}: DASH {} {}".format(action.cycle, action.unum, action.dash_power, action.dash_dir))
        # elif action.name == 'kick' and action.unum == self.M.get_taker(action.cycle):
        #     print("{} by {}: KICK {} {}".format(action.cycle, action.unum, action.kick_power, action.kick_direction))
        # elif action.name == 'move' and action.unum == self.M.get_taker(action.cycle):
        #     print("{} by {}: MOVE {} {}".format(action.cycle, action.unum, action.move_x, action.move_y))
        # elif action.name == 'turn_neck' and action.unum == self.M.get_taker(action.cycle):
        #     print("{} by {}: TURNNECK {}".format(action.cycle, action.unum, action.turn_neck_angle))
        # elif action.name == 'turn' and action.unum == self.M.get_taker(action.cycle):
        #     print("{} by {}: TURN {}".format(action.cycle, action.unum, action.turn_moment))

    def send_start_signal(self, cycle, gamemode, side, unum):
        self.our_side = 'r' if side == -1 else 'l'

        if not self.episode_ongoing and gamemode == "penalty_setup_{}".format(self.our_side):
            self.episode_ongoing = True
            self.M.signal_start(cycle, unum)
            print("----------------------------------------------------------Start", cycle, unum)


    def send_terminal_signal(self, cycle, gamemode, side):
        if self.episode_ongoing:
            self.episode_ongoing = False
            if 'score' in gamemode:
                self.our_score += 1
                reward = 1
            else:
                reward = 0

            self.M.signal_end(cycle, reward)

            self.n_penalty += 1
            print("@@@@@@@@@@@@@@Score: {}/{}={}".format(self.our_score, self.n_penalty, self.our_score/self.n_penalty))
            print("self.total_cycle:", self.total_cycle)


    def act(self, state):
        action = rl_thrift.Action()

        if np.random.rand() < 0.2:
            action.name = 'dash'
            action.dash_power = np.random.rand()*100
            action.dash_dir = np.random.rand()*360 - 180
            print("Random dash", action.dash_power, action.dash_dir)
            self.M.set_drop(state.cycle)
        else:
            action.name = 'IA'
            print("IA")

        # print(action)
        return action



dispatcher = Dispatcher()
server = make_server(rl_thrift.RL, dispatcher, unix_socket='/robocup/RL/socket')
try:
    server.serve()
except KeyboardInterrupt:
    dispatcher.M.save('penalty_with_random.data')
