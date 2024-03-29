#!/usr/bin/env python
#-*- coding: utf-8 -*-
#*****************************************************************************
#*
#* ALPS MPS DMRG Project
#*
#* Copyright (C) 2014 Laboratory for Physical Chemistry, ETH Zurich
#*               2014-2014 by Sebastian Keller <sebkelle@phys.ethz.ch>
#*
#* 
#* This software is part of the ALPS Applications, published under the ALPS
#* Application License; you can use, redistribute it and/or modify it under
#* the terms of the license, either version 1 or (at your option) any later
#* version.
#* 
#* You should have received a copy of the ALPS Application License along with
#* the ALPS Applications; see the file LICENSE.txt. If not, the license is also
#* available from http://alps.comp-phys.org/.
#*
#* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
#* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
#* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
#* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
#* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
#* DEALINGS IN THE SOFTWARE.
#*
#*****************************************************************************

import sys
import numpy as np
from maquis.fileio import loadEigenstateMeasurements

from maquis.utils.corrutils import pretty_print

#import numpy as np
def load_1rdm(inputfile):
    # load data from the HDF5 result file
    rdm =  loadEigenstateMeasurements([inputfile], what='oneptdm')[0][0]
    return rdm

def print_1rdm(rdm):
    #fmt = '% -016.10E'
    fmt = '%e'

    L = int(rdm.props["L"])
    mat = np.zeros((L,L))

    for lab, val in zip(rdm.x, rdm.y[0]):
        i = lab[0]
        j = lab[1]

        mat[i,j] = val;
        mat[j,i] = val;

    pretty_print(mat)

if __name__ == '__main__':
    inputfile = sys.argv[1]

    rdm = load_1rdm(inputfile)
    print_1rdm(rdm)
