#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import threading

import thriftpy
robocup_thrift = thriftpy.load("robocup.thrift", module_name="robocup_thrift")

from thriftpy.rpc import make_server

import time
import h5py

class myThread(threading.Thread):
    def __init__(self, workQueue):
        threading.Thread.__init__(self)
        self.q = workQueue
        self.exit_flag = False
        self.f = h5py.File("data.hdf5", "a")

        if 'field_evaluator' not in self.f:
            self.f.create_dataset('field_evaluator', (0, 88), dtype='d', maxshape=(None, 88))

    def run(self):
        while not self.exit_flag:
            if not self.q.empty():
                data = self.q.get()
                if data[0] == 'sync':
                    print("SYNC", len(data[1]))
                    self.extend_dataset(data[1])
                elif data[0] == 'exit':
                    print('EXIT')
                    self.exit()
            time.sleep(1)

    def extend_dataset(self, buffer_data):
        current_size = self.f['field_evaluator'].shape[0]
        self.f['field_evaluator'].resize((current_size+len(buffer_data), 88))
        for i in range(0, len(buffer_data)):
            self.f['field_evaluator'][current_size+i] = buffer_data[i]
        del buffer_data

    def exit(self):
        self.f.close()
        self.exit_flag = True



class Dispatcher(object):

    def __init__(self):
        self.buffer_data = []
        self.workQueue = queue.Queue(1)
        self.thread = myThread(self.workQueue)
        self.thread.start()

    def save_field_evaluations(self, list_):
        for l in list_:
            ll = [l.cycle, l.res, l.holder_unum, l.wm_self_unum, l.state_self_unum,
                 l.wm_ball_x, l.wm_ball_y, l.state_ball_x, l.state_ball_y,
                 l.theirTeamGoalPos_x, l.theirTeamGoalPos_x, l.wm_self_x, l.wm_self_y,
                 l.holder_x, l.holder_y, l.action_type]

            opps = [[0, 0, -1] for i in range(0, 12)]
            for opp in l.opps:
                if opp.unum >=0:#If identified player
                    # print("opp.unum", opp.unum)
                    opps[opp.unum][0] = opp.x
                    opps[opp.unum][1] = opp.y
                    opps[opp.unum][2] = 1#1 for identified, -1 for null data
            opps = [item for sublist in opps for item in sublist]
            ll.extend(opps)

            mates = [[0, 0, -1] for i in range(0, 12)]
            for mate in l.mates:
                if mate.unum >=0:#If identified player
                    mates[mate.unum][0] = mate.x
                    mates[mate.unum][1] = mate.y
                    mates[mate.unum][2] = 1#1 for identified, -1 for null data
            mates = [item for sublist in mates for item in sublist]
            ll.extend(mates)
            self.buffer_data.append(ll)

        print(list_[0].cycle, len(list_))
        self.sync()

    def sync(self, blocking=False):
        res = self.send_thread('sync', self.buffer_data, blocking)
        if res:
            self.buffer_data = []
    def send_thread(self, msg, data, blocking):

        try:
            self.workQueue.put((msg, data), block=blocking)
            return True
        except:
            return False

    def exit(self):
        self.sync(True)
        self.send_thread('exit', None, True)
        self.thread.join()

try:
    dispatcher = Dispatcher()
    server = make_server(robocup_thrift.RoboCup, dispatcher, unix_socket='/robocup/src/socket')
    server.serve()
except KeyboardInterrupt:
    dispatcher.exit()
