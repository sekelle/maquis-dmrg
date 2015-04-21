#!/usr/bin/env python
# -*- coding: utf-8 -*-
#*****************************************************************************
#*
#* ALPS MPS DMRG Project
#*
#* Copyright (C) 2013 Laboratory for Physical Chemistry, ETH Zurich
#*               2012-2014 by Sebastian Keller <sebkelle@phys.ethz.ch>
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
import s2
if __name__ == '__main__':
    inputfile = sys.argv[1]

    guinea_pig = s2.MaquisMeasurement(inputfile)

    print "s1 matrix"
    print guinea_pig.s1()

    print "s2 matrix"
    s2m = guinea_pig.s2()
    s2.pretty_print(s2m)

    print "I (mutual information)"
    s2.pretty_print(guinea_pig.I())

import matplotlib.pyplot as plt
from matplotlib import lines
from pylab import *

#MUTUALI INFORMATION PLOT:

#mutual information matrix:
mat_I=guinea_pig.I()
#single entropy vector:
vec_s1=guinea_pig.s1()

N= len(guinea_pig.I())
theta = np.zeros(N)
r=np.zeros(N)
labels=np.zeros(N)
area=np.zeros(N)

for i in range(N):
  theta[i]=-2*pi/N*i+pi/2
  r[i]=1.0
  labels[i]=i+1
  area[i]=vec_s1[i]*500


ax = plt.subplot(111,polar=True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(b=False)
c = plt.scatter(theta,r,c="Red",s=area)

plt.title('N = '+str(N/2)+', Results file: '+sys.argv[1])

#this is dummy:
c1 = plt.scatter(theta,(r+0.1),c="red",s=0)

# generation of orbital images. If "-i" switch is passed to the script, the script will incorporate orbital pictures into the image. Pictures must be present in the current directory with names #.png where # is the number of each site. Such images can be conveniently generated with gabedit, vmd or any other orbital plotting program you desire.

# Generation of pictures requires new python and matplotlib versions
pics=False
if (len(sys.argv) > 2):
  if (sys.argv[2] == '-i'):
    pics=True

legendlines = {}

for i in range(N):
#  plt.annotate(int(labels[i]),xy=(theta[i],(r[i]+0.2)),size='xx-large',)
  plt.text(theta[i],(r[i]+0.18),int(labels[i]),size='xx-large',ha='center',va='center')
  
  if(pics): # generate pictures.
    import matplotlib.pyplot as plt
    from matplotlib import lines
    from pylab import *
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.cbook import get_sample_data
    from matplotlib._png import read_png
    
    img=OffsetImage(read_png(str(i+1)+".png"),zoom=0.4) # The zoom factor should be ideally adjusted to the size of the images
    ab=AnnotationBbox(img,[theta[i],r[i]+0.57]) # pass Frameon=False to disable the frames around the images
    ax.add_artist(ab)
    
  for j in range(i,N):
    x =[theta[i],theta[j]]
    y =[1,1]
    if mat_I[i,j] >= 0.1:
      line = lines.Line2D(x, y, linewidth=2*10*mat_I[i,j], color='black',linestyle='-', alpha=1,label='0.1')
      legendlines['0.1'] = line
      ax.add_line(line)
    elif mat_I[i,j] >=0.01:
      line = lines.Line2D(x, y, linewidth=2*30*mat_I[i,j], color='gray',linestyle='--', alpha=1,label='0.01')
      legendlines['0.01'] = line
      ax.add_line(line)
    elif mat_I[i,j] >=0.001:
      line = lines.Line2D(x, y, linewidth=1.5, color='lime',linestyle=':', alpha=1,label='0.001')
      legendlines['0.001'] = line
      ax.add_line(line)


ax.legend(legendlines.values(),[l.get_label() for l in legendlines.values()],bbox_to_anchor=(0.00,1.0),fancybox=True,shadow=True)
plt.show()
