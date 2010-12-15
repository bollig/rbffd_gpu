
###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
		projectSettings	    # WORKS (demonstrates using ProjectSettings class)
		template            # WORKS (empty test)
		surface_integral    # WORKS (compute a simple surf integral on parametric geometry)
		surface_projection  # WORKS (compute a simple projection on parametric geometry)
		3d_octree           # WORKS (build and test a 3D octree)
		regulargrid3D_heat  # WORKS (Compute Diffusion equation inside 3D cube volume)
		regular_square2D_heat # WORKS (compute Diffusion equation inside 2D Square)
		binary_cvt_io       # COMPLETE (test if reading and writing a binary CVT file functions the same as ascii data)
		nested_spheres_ccvt # COMPLETE (nested spheres with ccvt of the boundary and cvt of interior.)
		original            # WORKS (base case without parallel support. 2D ellipse)
		#cusp_version_test   # COMPLETE (verifies build with CUSP and THRUST libraries).  
		#cusp_simple_example   # COMPLETE (form sparse system and solve)


		ncar_poisson1       # INCOMPLETE (poisson equation on nested spheres)
		new_ncar_poisson1       # INCOMPLETE (poisson equation on nested spheres)
		ncar_poisson2       # INCOMPLETE (poisson equation on nested spheres)


		3D_test             # INCOMPLETE (3D ellipsoid test which fails to generate points correctly resulting in singular systems)
                3d_ccvt             # INCOMPLETE (compute a 3D ccvt inside ellipsoid)
                cubeHeat3D_benchmarks # INCOMPLETE (regulargrid3D_heat plus benchmarking)

                square_and_cube_ccvt # INCOMPLETE (square and cube cvt with points projected to boundary. NOTE: boundary projection is incomplete (not perfect for distribution; code projects points as they near boundary but the distance is not guesswork)).

		gl_stencil_visualizer
		vtk_stencil_visualizer

		viennacl_simple_example

	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
