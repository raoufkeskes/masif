# -*- coding: utf-8 -*-
# @Time    : 17/06/2022 11:03
# @Author  : Raouf KESKES
# @Email   : raouf.keskes@mabsilico.com
# @File    : tmp.py

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



get_available_gpus()

