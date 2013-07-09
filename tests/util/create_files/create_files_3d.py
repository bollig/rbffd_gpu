# Must be used from build/tests/util/create_files
# Makefile should place it there.  (does not at this time.)

import os

nb_sten = [16,31,32,33,64]
nb_pts = [8,16,32,64,128]
nb_sten = [32]
nb_pts = [64]
sten = ["compact", "random"]
sten = ["random"]

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
                         """ % (ns,np,np,np,s)
                       print(file_content)
                       fd = open('create.conf', 'w')
                       fd.write(file_content)
                       fd.close()
                       CMD = "./create_files.x"
                       os.system(CMD)

