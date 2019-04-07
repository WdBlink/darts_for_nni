"""
File: test_nni_tuner.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    This file is just test for custom tuner
"""
from nni.tuner import Tuner


class test_tuner(Tuner):
    """
    test_tuner
    """

    def __init__(self, arg1):
        print(arg1)

    def receive_trial_result(self, parameter_id, parameters, value):
        '''
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including
        default metric
        '''
        pass

    def generate_parameters(self, parameter_id):
        # implementation
        '''
        The return_new_parameter should be a dict and it
        should be like follow:
        {"dropout": 0.3, "learning_rate":
        0.4}
        '''
        pass
