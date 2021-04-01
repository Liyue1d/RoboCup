#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from threading import Thread, Lock

from rl_memory import *
# from use_actor import load_model
class PlayerActionState:
    def __init__(self):
        self.state = None
        self.action = None
        self.actionNeck = None

class Cycle:
    def __init__(self, taker):
        self.taker = taker
        self.terminal = False
        self.reward = 0
        self.players = [PlayerActionState() for i in range(12)]
        self.drop = False

    def is_complete(self):
        p = self.players[self.taker]
        if self.terminal:
            if p.state is None:
                return False
        else:
            if p.state is None:
                return False
            if p.action is None:
                return False
            if p.actionNeck is None:
                return False
        return True

    def display(self):
        p = self.players[self.taker]
        return [p.state, p.action, p.actionNeck, self.reward, self.terminal]

    def serialize(self):
        p = self.players[self.taker]
        state = [
            int(p.state.is_kickable),
            p.state.ball_pos_x,
            p.state.ball_pos_y,
            p.state.ball_vel_x,
            p.state.ball_vel_y,
            p.state.ball_pos_count,
            p.state.self_stamina,
            p.state.goalie_pos_x,
            p.state.goalie_pos_y,
            p.state.goalie_vel_x,
            p.state.goalie_vel_y,
            p.state.goalie_body,
            p.state.self_pos_x,
            p.state.self_pos_y,
            p.state.self_vel_x,
            p.state.self_vel_y,
            p.state.self_body,
            p.state.self_dash_rate,
            p.state.time_spent, # TODO keep ?
            p.state.self_unum
        ]
        action = [
            1 if p.action.name == 'dash' else 0, # DASH
            1 if p.action.name == 'kick' else 0, # KICK
            1 if p.action.name == 'move' else 0, # MOVE
            1 if p.action.name == 'turn' else 0, # TURN
            # 1 if p.actionNeck.name == 'turn_neck' else 0, # ACTION NECK TODO always there?
            p.action.dash_power,
            p.action.dash_dir,
            p.action.kick_power,
            p.action.kick_direction,
            p.action.move_x,
            p.action.move_y,
            p.action.turn_moment,
            p.actionNeck.turn_neck_angle
        ]
        return state, action, self.reward, self.terminal

class M:
    def __init__(self):
        self.cycles = {}
        self.current_taker = None
        self.mutex = Lock()
        self.memory = RLMemory()
        # self.model = load_model({'layers': '1024, 500, 300, 100'})

    def get_cycle(self, cycle):
        if cycle not in self.cycles:
            self.cycles[cycle] = Cycle(self.current_taker)
        return self.cycles[cycle]

    def get_action_state(self, cycle, unum):
        return self.get_cycle(cycle).players[unum]

    def get_taker(self, cycle):
        return self.get_cycle(cycle).taker

    def add_state(self, state):
        self.get_action_state(state.cycle, state.self_unum).state = state

    def add_action(self, action):
        if action.name == 'turn_neck':
            self.get_action_state(action.cycle, action.unum).actionNeck = action
        else:
            self.get_action_state(action.cycle, action.unum).action = action

    def signal_start(self, cycle, taker):
        self.current_taker = taker
        c = self.get_cycle(cycle)
        c.taker = self.current_taker # Necessary in case state or action before signal start

    def signal_end(self, cycle, reward):
        c = self.get_cycle(cycle)
        c.terminal = True
        c.reward = 0

        self.current_taker = None # Starting at next cycle, no takers

    def set_drop(self, cycle):
        self.get_cycle(cycle).drop = True

    def process(self, threshold=2, flush=False):
        self.mutex.acquire()
        cycles = list(self.cycles.keys())
        most_recent_cycle = max(cycles)

        for cycle in cycles:
            c = self.cycles[cycle]
            if most_recent_cycle - cycle > threshold or flush:#and most_recent_cycle - cycle < 100:
                if c.taker is None:
                    pass
                    # DROP
                elif c.drop:
                    print("Set dropped")
                else:
                    if c.is_complete():
                        self.memory.add(*c.serialize())
                        # state, _, _, _ = c.serialize()
                        # e =self.model(np.matrix(state))
                        # print('e', e)
                    else:
                        print(c.display())
                        self.memory.add_crash()
                        msg = "Missed cycle: {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(cycle)
                        print(msg, most_recent_cycle, cycle)
                        # raise Exception(msg)

                del self.cycles[cycle]

        self.mutex.release()

    def save(self, filename):
        self.memory.save_to_file(filename)
