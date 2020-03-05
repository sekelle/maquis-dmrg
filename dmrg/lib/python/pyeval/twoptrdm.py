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

import maquisFile

#import numpy as np
def load_2rdm(inputfile):
    # load data from the HDF5 result file
    rdm =  maquisFile.loadEigenstateMeasurements([inputfile], what='twoptdm')[0][0]
    rdm.y[0] = 0.5 * rdm.y[0]
    return rdm

def load_2rdm_matrix(rdm):
    L = int(rdm.props['L'])
    odm = np.zeros([L,L,L,L])

    for lab, val in zip(rdm.x, rdm.y[0]):
        i = lab[0]
        j = lab[1]
        k = lab[2]
        l = lab[3]

        odm[i,j,k,l] = val

        if l != k or i != j:
            odm[j,i,l,k] = val

        if min(i,j) != min(l,k) or max(i,j) != max(l,k):
            odm[k,l,i,j] = val
            if l != k or i != j:
                odm[l,k,j,i] = val

    return odm

def print_2rdm_matrix(rdm):
    fmt = '%e'

    assert (rdm.shape[0] == rdm.shape[1] == rdm.shape[2] == rdm.shape[3])
    L = rdm.shape[0]

    irange = np.arange(L)
    idx = [ (i,j,k,l) for i in irange for j in irange for k in irange for l in irange ]
    for (i,j,k,l) in idx:
        print(i,j,k,l, fmt%rdm[i,j,k,l])

if __name__ == '__main__':
    inputfile = sys.argv[1]

    rdm_dataset = load_2rdm(inputfile)
    rdm = load_2rdm_matrix(rdm_dataset)
    print_2rdm_matrix(rdm)
