#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ****************************************************************************
# 
# MAQUIS DMRG Project 
# 
# Copyright (C) 2009 by Bela Bauer
# Copyright (C) 2013 by Michele Dolfi
# Copyright (C) 2020 by Sebastian Keller
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
# FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# ****************************************************************************

import numpy as np
import matplotlib.pyplot as plt

from maquis.fileio import flatten, DataSet

def on_xlim_changed(ax):
    xlim = ax.get_xlim()
    for a in ax.figure.axes:
        # shortcuts: last avoids n**2 behavior when each axis fires event
        if a is ax or len(a.lines) == 0 or getattr(a, 'xlim', None) == xlim:
            continue

        ylim = np.inf, -np.inf
        for l in a.lines:
            x, y = l.get_data()
            # faster, but assumes that x is sorted
            start, stop = np.searchsorted(x, xlim)
            yc = y[max(start-1,0):(stop+1)]
            ylim = min(ylim[0], np.nanmin(yc)), max(ylim[1], np.nanmax(yc))

        # TODO: update limits from Patches, Texts, Collections, ...

        # x axis: emit=False avoids infinite loop
        a.set_xlim(xlim, emit=False)

        # y axis: set dataLim, make sure that autoscale in 'y' is on 
        corners = (xlim[0], ylim[0]), (xlim[1], ylim[1])
        a.dataLim.update_from_data_xy(corners, ignore=True, updatex=False)
        a.autoscale(enable=True, axis='y')
        # cache xlim to mark 'a' as treated
        a.xlim = xlim


colors = ['k','b','g','m','c','y']
markers = ['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8', '+', 'x']


def plot(data):
    """ plots a list of datasets
    
        This function takes a DataSet or a list of DataSets and creates a matplotlib plot.
        
        It creates a new plot set for each dataset, using the x and y members of the DataSet.
        It also inspects the props dictionary and uses the following key-value pairs in that dict:
        
         title
         xlabel
         ylabel
         label
         filename (used as alternative label of the set if no label is specified)
         color (can be 'k','b','g','m','c', or 'y'
         line (can be 'line' or 'scatter')
      
    """
    lines = []
    icolor = 0
    imarker = 0
    if isinstance(data, DataSet):
      s = [data]
    else:
      s = data
    for q in flatten(s):
        try:
            xmeans = np.array([xx.mean for xx in q.x])
            xerrors = np.array([xx.error for xx in q.x])
        except AttributeError:
            xmeans = [float(vvv) for vvv in q.x]
            xerrors = None
        except TypeError:
            xmeans = [q.x]
            xerrors = None

        try:
            ymeans = np.array([xx.mean for xx in q.y])
            yerrors = np.array([xx.error for xx in q.y])
        except AttributeError:
            ymeans = [float(vvv) for vvv in q.y]
            yerrors = None
        except TypeError: # this usually means that it's scalar
            ymeans = [q.y]
            yerrors = None

        if 'label' in q.props and q.props['label'] != 'none':
            lab = q.props['label']
        elif 'filename' in q.props:
            lab = q.props['filename']
        else:
            lab = None

        if 'xlabel' in q.props:
            plt.xlabel(q.props['xlabel'])

        if 'ylabel' in q.props:
            plt.ylabel(q.props['ylabel'])

        if 'title' in q.props:
            plt.title(q.props['title'])

        thiscolor = colors[icolor]
        icolor = (icolor+1)%len(colors)
        if 'color' in q.props:
            thiscolor = q.props['color']

        thismarker = markers[imarker]
        if 'marker' in q.props:
            thismarker = q.props['marker']
        imarker = (imarker+1)%len(markers)

        fillmarkers = True
        if 'fillmarkers' in q.props and q.props['fillmarkers'] == False:
            fillmarkers = False

        linewidth = plt.rcParams['lines.linewidth']
        if 'line.linewidth' in q.props:
            linewidth = q.props['line.linewidth']

        if 'line' in q.props and q.props['line'] == 'scatter':
            if fillmarkers:
                plt.scatter(xmeans, ymeans, color=thiscolor, marker=thismarker, label=lab)
            else:
                plt.scatter(xmeans, ymeans, label=lab,
                            marker=thismarker, color=thiscolor, facecolors='none')
            imarker = (imarker+1)%len(markers)
        else:
            line_props = '-'
            if 'line' in q.props:
                line_props = q.props['line']

            if fillmarkers:
                plt.errorbar(xmeans,ymeans,yerr=yerrors,xerr=xerrors,fmt=line_props,linewidth=linewidth,color=thiscolor,label=lab)
            else:
                plt.errorbar(xmeans,ymeans,yerr=yerrors,xerr=xerrors,label=lab,
                            fmt=line_props,linewidth=linewidth,color=thiscolor, mfc='none', mec=thiscolor)
