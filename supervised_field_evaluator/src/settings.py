#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# rcssserver settings ----------------------------------------------------------

# auto_simulation settings -----------------------------------------------------
rcssser_path = "/robocup/" #Used by launch_server to prevent rcssserver first launch's bug
server_task_timeout = 60*10 #Server side timeout for waiting for a simulation result
client_task_timeout = 60*10

# Experiment settings ----------------------------------------------------------------------------------------------------

# ip = "157.16.52.60"
# port = 4242
# path_to_data_file = "/robocup/robocup42/"
# data_file = "fireflyTestEpsilon.data"
# formation_conf_templates = "../robocup42-payload/formationsPayloadopuSCOM/"
#
# team_to_optimize_name = "opuSCOM"
TEAM_L = "/robocup/opuSCOM2D/src/start.sh -p {port} -P {olcoach_port}"

# opponent_team_name = "agent2d"
TEAM_R = "/robocup/rctools-agent2d/src/start.sh -p {port} -P {olcoach_port}"
