#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py

from metadata import metadata

class RLMemory:

    def __init__(self, filename=None):
        self.memory = []

        if filename is not None:
            self._load_from_file(filename)

    def add(self, s, a, r, terminal=False):
        if not terminal:
            m = self._to_internal_form(False, r, s, a)
        else:
            m = self._to_internal_form(True, r, s, None)
        self.memory.append(m)

    def add_crash(self):
        # TODO: remove all the episode data?
        # TODO: non 0 reward?
        # Register as terminal, with state equal to previous state
        _, _, previous_s, _ = self._to_rl_form(self.memory[-1])
        m = self._to_internal_form(True, 0, previous_s, None)
        self.memory.append(m)

    def sample_batch(self, batch_size):
        # print(len(self.memory), int(len(self.memory)/2), batch_size)
        selected = np.random.choice(int(len(self.memory)/2), batch_size, replace=False)
        batch = []
        for i in range(selected.shape[0]):
            j = 2*selected[i]
            if self.memory[j][0] == 1:
                #Picked a terminal state -> pick the previous state
                j -= 1

            #     s, a, r, s'
            terminal, r, s, a = self._to_rl_form(self.memory[j])
            terminal_prime, r_prime, s_prime, a_prime = self._to_rl_form(self.memory[j+1])
            m = (s, a, r_prime, s_prime)# TODO check proper order

            # TODO temp
            # if j > 1:
            #     terminal_past, r_past, s_past, a_past = self._to_rl_form(self.memory[j-1])
            #     old_a_c = a_past[0:12]
            #     terminal_past2, r_past, s_past, a_past = self._to_rl_form(self.memory[j-2])
            #     old2_a_c = a_past[0:12]
            #     # print("a_past", old_a_c)
            #     if terminal_past:
            #         old_a_c = [0 for _ in range(12)]
            #     if terminal_past2:
            #         old2_a_c = [0 for _ in range(12)]
            # else:
            #     old_a_c = [0 for _ in range(12)]
            #     old2_a_c = [0 for _ in range(12)]

            # print("old_a_c", old_a_c)
            # print("s", s)
            # m = (np.concatenate((s, old_a_c, old2_a_c), axis=0), a, r_prime, s_prime)# TODO check proper order
            # print(m[0])

            batch.append(m)
        return batch

    def get_memory_size(self):
        return len(self.memory)

    def save_to_file(self, filename):
        print("Saving data to {}".format(filename))

        f = h5py.File(filename, "w")
        print("aazeazeea", self.memory[0])
        memory = np.matrix(self.memory)
        shape = memory.shape
        print("mem", memory.shape)
        d = f.create_dataset("penalty_data", shape, maxshape=shape, dtype='d')

        d[0: shape[0]] = memory

        f.close()

    def _load_from_file(self, filename):
        f = h5py.File(filename, "r")
        d = f['penalty_data']
        self.memory = d[0:d.shape[0]] # Load data in RAM, might not be necessary
        # terminal, r ,s ,a = self._to_rl_form(np.mean(self.memory, axis=0))
        # print("MEAN", a)
        f.close()

    def _to_internal_form(self, terminal, r, s, a):
        data = -42*np.ones(2+metadata.state_size()+metadata.action_size())
        empty_a = np.zeros((1, metadata.action_size()))
        # [Terminal, r, s ,a]

        state_end = 2+metadata.state_size()
        action_end = state_end + metadata.action_size()

        data[0] = int(terminal)
        data[1] = r
        data[2:state_end] = s
        data[state_end:action_end] = a if not terminal else empty_a

        return data

    def _to_rl_form(self, data):
        state_end = 2+metadata.state_size()
        action_end = state_end + metadata.action_size()
        terminal, r, s, a = bool(data[0]), data[1], data[2:state_end], data[state_end:action_end]
        return terminal, r, s, a
