# Must be used from build/tests/util/create_files
# Makefile should place it there.  (does not at this time.)

import os

coprocessor = "cascade_phi"
notes = "study_of_attributes"   # for output file name
output_dir = "d_output_%s_%s" % (coprocessor, notes)

#nb_sten = [16,31,32,33,64]
#nb_pts = [8,16,32,64,128]
nb_sten = [16,32,64]
nb_pts = [32,64,128]
sten = ["compact", "random"]

func_kernel = ["FUN_KERNEL", 
               "FUN_INV_KERNEL",
               "FUN4_DERIV4_WEIGHT4",
               "FUN4_DERIV4_WEIGHT4_INV",
               "FUN1_DERIV4_WEIGHT4"]

kernel_attributes = ["double4", 
                     "double",
                     "float4",
                     "float",
                     ""]

nb_sten = [16,32,64]
nb_pts = [128]
nb_pts = [64]
nb_sten = [16,32]
nb_sten = [32]
#kernel_attributes = ["double4"]
#func_kernel = ["FUN4_DERIV4_WEIGHT4_INV"]

sten = ["random", "compact"]
sten = ["compact"]

os.system("mkdir %s" % output_dir )

for ns in nb_sten:
       for np in nb_pts:
               for s in sten:
                 for kernel in func_kernel:
                   for kern_attr in kernel_attributes:
                       file_content="""
 #  ===========================================================
                         DIMENSION = 3
                         STENCIL_SIZE = %d
                         NB_X = %d
                         NB_Y = %d
                         NB_Z = %d
                         NODE_DIST = %s
                         FUN_KERNEL = %s
                         KERNEL_ATTRIBUTES = %s 
 #  ===========================================================
                         """ % (ns,np,np,np,s,kernel,kern_attr)
                       print(file_content)
                       fd = open('data.conf', 'w')
                       fd.write(file_content)
                       fd.close()

                       attrib_content= """
                  // =================================================
                  #define ATTRIBUTES __attribute__((vec_type_hint(%s)))
                  // =================================================
                       """ %  kern_attr
                       if kern_attr == "":
                               attrib_content= """
                  // =================================================
                  #define ATTRIBUTES 
                  // =================================================
                       """

                       print(attrib_content);
                       fd = open("cl_kernels/cl_attributes.h", "w")
                       fd.write(attrib_content)
                       fd.close()

                       if kern_attr == "":
                             kern_attr = "none"
                       out_file = "%s_x_weights_direct__no_hv_stsize_%d_3d_%dx_%dy_%dz.mtxb_%s_attr_%s" % (s,ns,np,np,np,kernel,kern_attr)
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


