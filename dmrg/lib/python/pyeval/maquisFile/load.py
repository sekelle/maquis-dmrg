#!/usr/bin/env python
# -*- coding: utf-8 -*-
#*****************************************************************************
#*
#* ALPS MPS DMRG Project
#*
#* Copyright (C) 2020 CSCS, ETH Zurich
#*               2020-2020 by Sebastian Keller <sebkelle@phys.ethz.ch>
#*
#* adapted from pyalps/load.py to python 3 and h5py
#*
#* Copyright (C) 2009-2010 by Bela Bauer <bauerb@phys.ethz.ch>
#*                            Brigitte Surer <surerb@phys.ethz.ch> 
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

import numpy as np
import os.path
from copy import deepcopy

from . import h5pyWrapper as h5
from . dataset import *

##############################################################################################
# these functions correspond to the original pyalps code for loading final measurement results
##############################################################################################

def log(m):
    """ print a log message"""
    print(m)


def parse_label(label):
    if '--' in label:
      vals = label.rsplit('--')
      ret = ()
      for val in vals:
          ret = ret + (eval(val),)
      return ret
    else:
      return eval(str(label))


def parse_labels(labels):
    if type(labels)==int:
      return np.array([labels])
    larr=[]
    allsame = True
    first = None
    for x in labels:
      v = parse_label(x)
      larr.append(v)
      if '--' in x:
        if first==None:
          first = v[0]
        else:
          if first != v[0] or len(v) != 2:
            allsame = False
      else:
        allsame = False
    if allsame:
      larr = [x[1] for x in larr]
    return np.array(larr)


class Hdf5Loader:

    def GetFileNames(self, flist):
        files = []
        for f in flist:
          if f[-4:]=='.xml':
            f = f[:-3]+'h5'
          else:
            if f[-3:]!='.h5':
              f += '.h5'
          if os.path.exists(f):
            files.append(f)
          else:
            log( "FILE "+ f+ "DOES NOT EXIST!")
        return files

    def ReadParameters(self, proppath):
        ret = {'filename' : self.h5fname}
        ret.update(self.h5f.readParameters(proppath))
        return ret
        
    def GetProperties(self, flist, proppath='/parameters', respath='/simulation/results', verbose=False):
        fs = self.GetFileNames(flist)
        resultfiles = []
        for f in fs:
            try:
                self.h5f = h5.archive(f, 'r')
                self.h5fname = f
                if verbose: log( "Loading from file" + f)
                rfile = ResultFile(f)
                rfile.props = self.ReadParameters(proppath)
                try:
                    obs = self.GetObservableList(respath)
                    rfile.props["ObservableList"] = [h5.pt.hdf5_name_decode(x) for x in obs]
                except: pass
                resultfiles.append(rfile)
            except Exception as e:
                log(e)
                log(traceback.format_exc())
        return resultfiles

    def GetObservableList(self,respath):
        if self.h5f.is_group(respath):
            olist = self.h5f.list_children(respath)
        else:
            olist = []
        return olist


    def GetIterations(self, current_path, params={}, measurements=None, index=None, verbose=False):
        iterationset=[]
        #iteration_grp = self.h5f.require_group(respath+'/iteration')
        for it in self.h5f.list_children(current_path+'/iteration'):
            obsset=[]
            iteration_props = {}
            if 'parameters' in self.h5f.list_children(current_path+'/iteration/'+it):
                iteration_props = self.ReadParameters(current_path+'/iteration/'+it+'/parameters')
            iteration_props['iteration'] = it

            respath = current_path+'/iteration/'+it+'/results'
            list_ = self.GetObservableList(respath)
            if measurements == None:
                obslist = list_
            else:
                obslist = [h5.pt.hdf5_name_encode(obs) for obs in measurements if h5.pt.hdf5_name_encode(obs) in list_]
            for m in obslist:
                if m in self.h5f.list_children(respath):
                    if "mean" in self.h5f.list_children(respath+'/'+m):
                        try:
                            d = DataSet()
                            itresultspath = respath+'/'+m
                            if verbose: log("Loading "+ m)
                            measurements_props = {}
                            measurements_props['hdf5_path'] = itresultspath
                            measurements_props['observable'] = h5.pt.hdf5_name_decode(m)
                            if index == None:
                                d.y = self.h5f[itresultspath+'/mean/value']
                                d.x = np.arange(0,len(d.y))
                            else:
                                try:
                                    d.y = self.h5f[itresultspath+'/mean/value'][index]
                                except:
                                    pass
                            if "labels" in self.h5f.list_children(itresultspath):
                                d.x = parse_labels(self.h5f[itresultspath+'/labels'])
                            else:
                                d.x = np.arange(0,len(d.y))
                            d.props.update(params)
                            d.props.update(iteration_props)
                            d.props.update(measurements_props)
                        except AttributeError:
                            log( "Could not create DataSet")
                    obsset.append(d)
            iterationset.append(obsset)
        return iterationset


    def ReadDiagDataFromFile(self,flist,proppath='/parameters',respath='/spectrum', measurements=None, index=None, loadIterations=False,verbose=False):
        fs = self.GetFileNames(flist)
        sets = []
        for f in fs:
            try:
                fileset=[]
                self.h5f = h5.archive(f, 'r')
                self.h5fname = f
                if verbose: log("Loading from file"+ f)
                params = self.ReadParameters(proppath)
                if 'results' in self.h5f.list_children(respath):
                    list_ = self.GetObservableList(respath+'/results')
                    if measurements == None:
                        obslist = list_
                    else:
                        #obslist = [h5.pt.hdf5_name_encode(obs) for obs in measurements if h5.pt.hdf5_name_encode(obs) in list_]
                        obslist = [obs for obs in measurements if obs in list_]
                    if loadIterations==True:
                        if "iteration" in self.h5f.list_children(respath+'/results'):
                            fileset.append(self.GetIterations(respath+'/results', params, measurements, index, verbose))
                    else:
                        for m in obslist:
                            if "mean" in self.h5f.list_children(respath+'/results/'+m):
                                try:
                                    if verbose: log("Loading " + m)
                                    d = DataSet()
                                    secresultspath = respath+'/results/'+m
                                    d.props['hdf5_path'] = secresultspath
                                    d.props['observable'] = h5.pt.hdf5_name_decode(m)
                                    if index == None:
                                        d.y = self.h5f[secresultspath+'/mean/value']
                                        d.x = np.arange(0,len(d.y))
                                    else:
                                        try:
                                            d.y = self.h5f[secresultspath+'/mean/value'][index]
                                        except:
                                            pass
                                    if "labels" in self.h5f.list_children(secresultspath):
                                        d.x = parse_labels(self.h5f[secresultspath+'/labels'])
                                    else:
                                        d.x = np.arange(0,len(d.y))
                                    d.props.update(params)

                                    fileset.append(d)
                                except AttributeError:
                                    log("Could not create DataSet")
                if loadIterations==True:
                    if "iteration" in self.h5f.list_children(respath):
                        fileset.append(self.GetIterations(respath, params, measurements, index, verbose))
                if 'sectors' in self.h5f.list_children(respath):
                    list_ = self.GetObservableList(respath+'/sectors/0/results')
                    if measurements == None:
                        obslist = list_
                    else:
                        obslist = [h5.pt.hdf5_name_encode(obs) for obs in measurements if h5.pt.hdf5_name_encode(obs) in list_]
                    for secnum in self.h5f.list_children(respath+'/sectors'):
                        sector_sets=[]
                        for m in obslist:
                            if "mean" in self.h5f.list_children(respath+'/sectors/'+secnum+'/results/'+m):
                                try:
                                    if verbose: log("Loading" + m)
                                    d = DataSet()
                                    secpath = respath+'/sectors/'+secnum
                                    secresultspath = respath+'/sectors/'+secnum+'/results/'+m
                                    d.props['hdf5_path'] = secresultspath
                                    d.props['observable'] = h5.pt.hdf5_name_decode(m)
                                    if index == None:
                                        d.y = self.h5f[secresultspath+'/mean/value']
                                        d.x = np.arange(0,len(d.y))
                                    else:
                                        try:
                                            d.y = self.h5f[secresultspath+'/mean/value'][index]
                                        except:
                                            pass
                                    if "labels" in self.h5f.list_children(secresultspath):
                                        d.x = parse_labels(self.h5f[secresultspath+'/labels'])
                                    else:
                                        d.x = np.arange(0,len(d.y))
                                    d.props.update(params)
                                    try:
                                        d.props.update(self.ReadParameters(secpath+'/quantumnumbers'))
                                    except:
                                        if verbose: log("no quantumnumbers stored ")
                                        pass
                                    sector_sets.append(d)

                                except AttributeError:
                                    log( "Could not create DataSet")
                                    pass
                        fileset.append(sector_sets)
                sets.append(fileset)
            except RuntimeError:
                raise
            except Exception as e:
                log(e)
        return sets


def loadEigenstateMeasurements(files, what=None, verbose=False):
    """ loads ALPS eigenstate measurements from ALPS HDF5 result files
    
        this function loads results of ALPS diagonalization or DMRG simulations from an HDF5 file
        
        Parameters:
            files (list): ALPS result files which can be either XML or HDF5 files. XML file names will be changed to the corresponding HDF5 names.
            what (list): an optional argument that is either a string or list of strings, specifying the names of the observables which should be loaded
            verbose (bool): an optional argument that if set to True causes more output to be printed as the data is loaded
        
        Returns:
            list of list of (lists of) DataSet objects: loaded measurements.
            The elements of the outer list each correspond to the file names specified as input
            The elements of the next level are different quantum number sectors, if any exists
            The elements of the inner-most list are each for a different observable
            The y-values of the DataSet objects is an array of the measurements in all eigenstates calculated in this sector, and the x-values optionally the labels (indices) of array-valued measurements
    """
    ll = Hdf5Loader()
    if isinstance(what,str):
      what = [what]
    return ll.ReadDiagDataFromFile(files,measurements=what,verbose=verbose)


#############################################################################################
# the functions below arw used for loading results recorded for every sweep
#############################################################################################


def ReadMeasurements(ar, measurements, path, props):
    ret = []
    if measurements is None:
        measurements = ar.list_children(path)

    for meas in measurements:
        if meas in ar.list_children(path):
            try:
                d = DataSet()
                d.props = deepcopy(props)
                d.props['observable'] = meas
                d.props['hdf5_path'] = path+'/'+meas
                d.y = ar[ d.props['hdf5_path']+'/mean/value' ]
                if 'labels' in ar.list_children(d.props['hdf5_path']):
                    d.x = parse_labels(ar[ d.props['hdf5_path']+'/labels' ])
                else:
                    d.x = range(len(d.y))
                ret.append(d)
            except Exception as e:
                print("Could not load " + meas)
                print(e)
                pass
    return ret

def ReadDMRGSweeps(ar, measurements, props, path='/simulation', selector=None):
    ret = []
    for sweep_name in ar.list_children(path):
        if sweep_name == 'iteration':  ## new iteration format
            # add sweeps in order
            sweeps = sorted(map(int, ar.list_children(path+'/iteration')))
            for sweep in sweeps:
                sweep_num = str(sweep)
                ret_sweep = []
                p_sweep = path+'/iteration/'+sweep_num
                props_sweep = deepcopy(props)
                props_sweep['sweep'] = sweep
                if 'parameters' in ar.list_children(p_sweep):
                    props_sweep.update( ar.readParameters(p_sweep+'/parameters') )
                ret_sweep.extend(ReadMeasurements(ar, measurements, p_sweep+'/results', props_sweep))
                if len(ret_sweep) > 0:
                    ret.append(ret_sweep)

        elif 'sweep' in sweep_name:  ## old iterarion format
            sweep = eval(sweep_name.strip('sweep'))
            ret_sweep = []
            p_sweep = path+'/'+sweep_name
            props_sweep = deepcopy(props)
            props_sweep['sweep'] = sweep
            if 'parameters' in ar.list_children(p_sweep):
                props_sweep.update( ar.readParameters(p_sweep+'/parameters') )

            ret_sweep.extend(ReadMeasurements(ar, measurements, p_sweep+'/results', props_sweep))
            if len(ret_sweep) > 0:
                ret.append(ret_sweep)

        elif sweep_name == 'results' and 'simulation' in path:
            props_sweep = deepcopy(props)
            props_sweep['sweep'] = -1
            if 'parameters' in ar.list_children(path):
                props_sweep.update( ar.readParameters(path+'/parameters') )

            tmp = ReadMeasurements(ar, measurements, path+'/results', props_sweep)
            if len(tmp) > 0:
                ret.append(tmp)
    return ret

def LoadDMRGSweeps(files, what=None, selector=None):
    """ loads measurements in each DMRG iteration from HDF5 result files
        
        The parameters are:
        
        files:    a list of DMRG result files in HDF5 format.
        what:     an optional argument that is either a string or list of
                  strings, specifying the names of the observables which
                  should be loaded
        selector: (if specified) function taking a dict of properties and
                  returning a boolean of the dataset has to be loaded
        
        The function returns a list of list of lists of DataSet objects. 
        The elements of the outer list each correspond to the file names
        specified as input.
        The elements of the next level are different sweeps
        The elements of the inner-most list are each for a different observable.
        The y-values of the DataSet objects is an array of the measurements
        in all eigenstates calculated in this sector, and the x-values
        optionally the labels (indices) of array-valued measurements
    """
    
    if isinstance(what, str):
      what = [what]
    
    ret = []
    for fname in files:
        ar = h5.archive(fname)
        if 'simulation' in ar.list_children('/'):
            props = ar.readParameters('/parameters')
            props['filename'] = fname
            tmp = ReadDMRGSweeps(ar, what, props, path='/simulation', selector=selector)
            if len(tmp) > 0:
                ret.append(tmp)
        if 'spectrum' in ar.list_children('/') and 'iteration' in  ar.list_children('/spectrum'):
            props = ar.readParameters('/parameters')
            props['filename'] = fname
            tmp = ReadDMRGSweeps(ar, what, props, path='/spectrum', selector=selector)
            if len(tmp) > 0:
                ret.append(tmp)
    return ret
