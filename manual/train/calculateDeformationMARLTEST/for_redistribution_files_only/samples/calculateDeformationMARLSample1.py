#!/usr/bin/env python
"""
Sample script that uses the calculateDeformationMARLTEST module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

import calculateDeformationMARLTEST
# Import the matlab module only after you have imported
# MATLAB Compiler SDK generated Python modules.
import matlab

my_calculateDeformationMARLTEST = calculateDeformationMARLTEST.initialize()

fixturePos_nIn = matlab.double([350.0, 360.0, 370.0, 800.0, 820.0, 840.0, 0.0, 0.0, 0.0], size=(3, 3))
drillPosIn = matlab.double([505.283, 487.164, 10.0], size=(1, 3))
maxValuesOut = my_calculateDeformationMARLTEST.calculateDeformationMARL(fixturePos_nIn, drillPosIn)
print(maxValuesOut, sep='\n')

my_calculateDeformationMARLTEST.terminate()
