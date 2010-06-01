
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
		surface_integral    # WORKS (compute a simple surf integral on parametric geometry)
		surface_projection  # WORKS (compute a simple projection on parametric geometry)
		3d_ccvt             # INCOMPLETE (compute a 3D ccvt inside ellipsoid)
		3d_octree           # WORKS (build and test a 3D octree)
		regulargrid3D_heat  # WORKS (Compute Diffusion equation inside 3D cube volume)
		regular_square2D_heat # WORKS (compute Diffusion equation inside 2D Square)
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
