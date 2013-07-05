# Must be used from build/tests/util/create_files
# Makefile should place it there.  (does not at this time.)

import os

coprocessor = "cascade_phi"
notes = ""   # for output file name
output_dir = "output_%s_%s" % (coprocessor, notes)

#nb_sten = [16,31,32,33,64]
#nb_pts = [8,16,32,64,128]
nb_sten = [32]
nb_pts = [64]
sten = ["compact", "random"]

#nb_sten = [32]
#nb_pts = [128]
#sten = ["compact"]

func_kernel = ["FUN_KERNEL", 
               "FUN_INV_KERNEL",
               "FUN4_DERIV4_WEIGHT4",
               "FUN4_DERIV4_WEIGHT4_INV",
               "FUN1_DERIV4_WEIGHT4"]

func_kernel = ["FUN1_DERIV4_WEIGHT4"]

os.system("mkdir %s" % output_dir )

for ns in nb_sten:
       for np in nb_pts:
               for s in sten:
                   for kernel in func_kernel:
                       file_content="""
 #  ===========================================================
                         DIMENSION = 2
                         STENCIL_SIZE = %d
                         NB_X = %d
                         NB_Y = %d
                         NB_Z = 1
                         NODE_DIST = %s
                         FUN_KERNEL = %s
 #  ===========================================================
                         """ % (ns,np,np,s,kernel)
                       print(file_content)
                       fd = open('data.conf', 'w')
                       fd.write(file_content)
                       fd.close()
                       out_file = "%s_x_weights_direct__no_hv_stsize_%d_3d_%dx_%dy_1z.mtxb_%s" % (s,ns,np,np,kernel)
                       out_full_path = "%s/%s" % (output_dir, out_file)

                       CMD = "cat data.conf > %s" % out_full_path
                       os.system(CMD)

                       CMD = "./gpu_one_deriv_one_fun.x >> %s" % out_full_path
                       os.system(CMD)

                       CMD = "cat data.conf >> %s" % out_full_path
                       os.system(CMD)
                       
                       CMD = "lscpu >> %s" % out_full_path 
                       os.system(CMD)

                       CMD = "/opt/intel/mic/bin/micinfo >> %s" % out_full_path
                       os.system(CMD)

                       CMD = "hostname >> %s" % out_full_path
                       os.system(CMD)

                       CMD = "uname -a  >> %s" % out_full_path
                       os.system(CMD)

