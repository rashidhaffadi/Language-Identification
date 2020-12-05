# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:51:14 2020

@author: Rashid Haffadi
"""

class ArgumentError(Exception):
    def __init__(self):
        self.__init__()
    def __call__(self, arg_name, func_name, arg_actual_value, arg_expected_value):
        self.e = """Argument \{{}\} in function \{{}\}, expected value is \{{}\}, 
        but got  value \{{}\} istead.""".format(arg_name, func_name, 
                                        arg_expected_value, arg_actual_value)
        print(self.e)
        