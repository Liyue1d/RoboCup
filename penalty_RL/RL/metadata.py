#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class MetaData:

    def __init__(self):

        self.state = VectorMetaData('State')
        self.actions = []
        self.other_parameters = VectorMetaData('OtherParameters')


    def state_size(self):
        return self.state.size()

    def action_size(self):
        return self.N_actions_choice() + self.action_parameters_size()

    def N_actions_choice(self):
        return len(self.actions)

    def action_parameters_size(self):
        n = 0
        for action in self.actions:
            n += action.size()
        n += self.other_parameters.size()
        return n

    def action_parameters_min(self):
        pass # TODO
        # Useful for bounding

    def action_parameters_max(self):
        pass # TODO

    def get_parameters_for_action(self, action):
        pass # TODO
        # returns indices of relevant parameters (including other parameters)
        # Or maybe a mask

    def action_id_for_param(self, param_name):
        id_ = None

        for action in self.actions:
            for i in range(len(action.names)):
                if action.names[i] == param_name:
                    return i

        for i in range(len(self.other_parameters.names)):
            for i in range(len(self.other_parameters.names)):
                if self.other_parameters.names[i] == param_name:
                    return i

        raise Exception('action_id_for_param: unknow param: {}'.format(param_name))

class VectorMetaData:

    def __init__(self, name):
        self.name = name
        self.names = []
        self.min = []
        self.max = []
        self.bool = []

    def add(self, name, min_, max_, bool_):
        self.names.append(name)
        self.min.append(min_)
        self.max.append(max_)
        self.bool.append(bool_)

    def size(self):
        return len(self.min)


metadata = MetaData()

metadata.state.add('is_kickable', 0, 1, True)
metadata.state.add('ball_pos_x', -57, 57, False)# Sometime more than 52.5
metadata.state.add('ball_pos_y', -34, 34, False)
metadata.state.add('ball_vel_x', -3, 3, False)# Not sure
metadata.state.add('ball_vel_y', -3, 3, False)# Not sure
metadata.state.add('ball_pos_count', 0, 1000, False)# Not sure
metadata.state.add('self_stamina', 0, 8000, False)# Not sure
metadata.state.add('goalie_pos_x', -57, 57, False)
metadata.state.add('goalie_pos_y', -34, 34, False)
metadata.state.add('goalie_vel_x', -3, 3, False)# Not sure
metadata.state.add('goalie_vel_y', -3, 3, False)# Not sure
metadata.state.add('goalie_body', -180, 180, False)
metadata.state.add('self_pos_x', -57, 57, False)# Not sure
metadata.state.add('self_pos_y', -34, 34, False)# Not sure
metadata.state.add('self_vel_x', -3, 3, False)# Not sure
metadata.state.add('self_vel_y', -3, 3, False)# Not sure
metadata.state.add('self_body', -180, 180, False)
metadata.state.add('self_dash_rate', 0.006, 0.006, False)# Constant
metadata.state.add('time_spent', 0, 300, False)# to adapt
metadata.state.add('self_unum', 0, 12, False) #TODO as separate booleans?


dash = VectorMetaData('Dash')
dash.add('dash_power', -100, 100, False)
dash.add('dash_dir', -180, 180, False)
metadata.actions.append(dash)

kick = VectorMetaData('Kick')
kick.add('kick_power', -100, 100, False)
kick.add('kick_direction', -180, 180, False)
metadata.actions.append(kick)

move = VectorMetaData('Move') #TODO remove because not used?
move.add('move_x', -52.5, 52.5, False)
move.add('move_y', -34.0, 34.0, False)
metadata.actions.append(move)

turn = VectorMetaData('Turn')
turn.add('turn_moment', -180, 180, False)
metadata.actions.append(turn)

# TODO Is this the best method?
metadata.other_parameters.add('turn_neck_angle', -180, 180, False)
