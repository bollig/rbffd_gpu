
###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
#		helloworld      # WORKS
#		message_test	# WORKS
#		density1	# WORKS
#		density2	# WORKS
#		density3	# WORKS
#		bollig_main	# INCOMPLETE
		3D_test
#		original	# WORKS
#		rbf_message	# INCOMPLETE
		template	# WORKS
		surface_integral
		surface_projection
#		3d_ccvt
		3d_octree
#		2d_regular_square
 #               3d_regular_cube
#                3d_cube_heat
		regulargrid3D_heat
		regular_square2D_heat
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
