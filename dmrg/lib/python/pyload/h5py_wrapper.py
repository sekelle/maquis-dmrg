#!/usr/bin/env python
# -*- coding: utf-8 -*-
#*****************************************************************************
#*
#* ALPS MPS DMRG Project
#*
#* Copyright (C) 2020 CSCS, ETH Zurich
#*               2020-2020 by Sebastian Keller <sebkelle@phys.ethz.ch>
#*
#*
#*****************************************************************************


import h5py

class archive:

    def __init__(self, filename, fileAttr = "r"):
       self.impl = h5py.File(filename, fileAttr) 

    def list_children(self, path):
        return list(self.impl[path])

    def __getitem__(self, key):
        sk = self.impl[key]
        if len(sk.shape) > 0:
            if sk.shape[0] > 0:
                if (isinstance(sk.value[0], bytes)):
                    return [ x.decode() for x in sk.value ]

        if isinstance(sk.value, bytes):
            return sk.value.decode()

        return self.impl[key].value

    def is_group(self, path):
        return isinstance(self.impl[path], h5py.Group)


class pt:

    def hdf5_name_decode(x):
        if isinstance(x, bytes):
            return x.decode()
        else:
            return x

    def hdf5_name_encode(x):
        if isinstance(x, str) == True:
            return x.encode('ascii')
        else:
            return x
