#!/usr/bin/python
# Must be used from build/tests/util/create_files
# Makefile should place it there.  (does not at this time.)

import os

nb_sten = [16,31,32,33,64]
nb_pts = [8,16,32,64,128]
nb_sten = [64]
nb_sten = [32]
nb_pts = [32,48]
nb_pts = [64,96]
nb_pts = [96,128]
nb_pts = [8]
nb_pts = [128]
nb_pts = [48]
nb_pts = [96]
nb_pts = [32]

nb_sten = [32,64]
nb_pts = [32,64,96]
nb_sten = [32]
nb_pts = [32,64]
nb_pts = [128]
nb_pts = [128, 192]
nb_pts = [96]
nb_pts = [64]
sten = ["compact", "random", "kd-tree"]
sten = ["kd-tree"]
#sten = ["compact"]
# better results with sym=1 (symmetrize adjacency matrix)
sym = 1

for ns in nb_sten:
       for np in nb_pts:
               for s in sten:
                       file_content="""
                         DIMENSION = 3
                         STENCIL_SIZE = %d
                         NB_X = %d
                         NB_Y = %d
                         NB_Z = %d
                         NODE_DIST = %s
                         SYM_ADJ = %d
                         """ % (ns,np,np,np,s,sym)
                       print(file_content)
                       out_file = "%s_nb_%d_sten_%d_sym_%d_3d.out" % (s,np,ns,sym)
                       fd = open('create.conf', 'w')
                       fd.write(file_content)
                       fd.close()
                       CMD = "./create_files.x > %s" % out_file
                       os.system(CMD)

