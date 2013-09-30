#!/usr/bin/python
# Must be used from build/tests/util/create_files
# Makefile should place it there.  (does not at this time.)

import os

nb_sten = [16,31,32,33,64]
nb_pts = [8,16,32,64,128]
nb_pts = [48]
nb_pts = [256]
nb_pts = [128]
nb_pts = [32,48,64,128,256]
nb_sten = [32,64]
# compact, random, kd-tree
sten = ["compact"]
sten = ["kd-tree"]
sym = 0

for ns in nb_sten:
       for np in nb_pts:
               for s in sten:
                       file_content="""
                         DIMENSION = 2
                         STENCIL_SIZE = %d
                         NB_X =  %d
                         NB_Y =  %d
                         NB_Z = 1
                         NODE_DIST = %s
                         SYM_ADJ = %d
                         """ % (ns,np,np,s, sym)
                       print(file_content)
                       out_file = "%s_nb_%d_sten_%d_sym_%d_2d.out" % (s,np,ns,sym)
                       fd = open('create.conf', 'w')
                       fd.write(file_content)
                       fd.close()
                       CMD = "./create_files.x > %s" % out_file
                       os.system(CMD)

