
###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
#		density1            # WORKS
#		density2            # WORKS
#		density3            # WORKS
#		bollig_main         # INCOMPLETE
		3D_test
		original            # WORKS (base case without parallel support. 2D ellipse)
		template            # WORKS (empty test)
		surface_integral
		surface_projection
		3d_ccvt
		3d_octree
		regulargrid3D_heat
		regular_square2D_heat
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
