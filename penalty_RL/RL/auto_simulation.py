#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import subprocess
import time
from multiprocessing import Process
import shutil
import os
import stat
import datetime
from collections import defaultdict
import shlex
from os.path import isfile, join

import settings

def launch_match(processID, TEAM_L, TEAM_R, log_directory, port, coach_port, olcoach_port, fast_mode):
    timeout_simulation = settings.client_task_timeout

    l = Process(target=launch_team, args=(TEAM_L, port, coach_port, olcoach_port,2,))
    r = Process(target=launch_team, args=(TEAM_R, port, coach_port, olcoach_port,4,))
    l.start()
    r.start()
    server = Process(target=launch_server, args=(processID, log_directory, port, coach_port, olcoach_port, fast_mode,))


    server.start()
    begin = time.time()
    while server.is_alive():
        time.sleep(10)
        print("simulation still alive")
        if time.time() - begin > timeout_simulation:
            print("Timeout: Killing simulation.")
            l.terminate()
            r.terminate()
            server.terminate()
            break



def launch_team(team, port, coach_port, olcoach_port, wait):

    os.chdir(os.path.dirname(team.split(" ")[0]))
    time.sleep(wait)

    args = shlex.split(team.format_map(defaultdict(str, port=port, coach_port=coach_port, olcoach_port=olcoach_port)))
    print("args:", args)
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    output, err = p.communicate()


def launch_server(processID, log_directory, port, coach_port, olcoach_port, fast_mode):

    os.chdir(settings.rcssser_path)
    synch_mode = "true" if fast_mode else "false"

    p = subprocess.Popen([
                "/robocup/serverPenalty/src/rcssserver", \
                "server::auto_mode=1", \
                "server::synch_mode={}".format(synch_mode), \
                "server::port={}".format(port),\
                "server::coach_port={}".format(coach_port),\
                "server::olcoach_port={}".format(olcoach_port),\
                "server::kick_off_wait=20", \
                "server::coach_w_referee=true", \
                "server::game_logging=1", \
                "server::text_logging=1", \
                # "server::pen_max_extra_kicks=0", \
                # "server::pen_nr_kicks=0", \
                # "server::nr_extra_halfs=0", \
                "server::log_date_format=%Y%m%d%H%M%S-{}-".format(processID),\
                "server::game_log_dir={}".format(log_directory), \
                "server::text_log_dir={}".format(log_directory), \
                "server::penalty_shoot_outs=true", \
                "server::nr_normal_halfs=0", \
                "server::nr_extra_halfs=0", \
                "server::pen_nr_kicks=50"

                ])
    p.wait()
    print("Returncode of P: ", p.returncode)
