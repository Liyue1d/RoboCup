#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import settings
from auto_simulation import launch_match

for i in range(0, 2):
    print("Launching match {}!!!!!!!!!!!!!!!!!!!!!".format(i))
    launch_match(i, settings.TEAM_L, settings.TEAM_R, "/robocup/", 6000, 6001, 6002, True)
