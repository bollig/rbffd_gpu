
###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
		# find * |grep -v "\." | grep -v "incomplete\/" | grep -v "cut\/"| grep "\/" | grep -v "NOTES" |grep -v "TODO"
		#P2		experiments/ncar_poisson1
		#P2		experiments/ncar_poisson2
		experiments/ncar_poisson1_nonuniform_diffusivity
		experiments/original_heat
		#P1		experiments/ncar_poisson1_regular_grid
		#P2		experiments/regular_square2D_heat
		#P2		experiments/regulargrid3D_heat
		#P2		experiments/heat_regulargrid_2d

		gridgen/nested_spheres_ccvt
		gridgen/regular_grid
		
		util/3d_octree
		util/binary_cvt_io
		util/cusp_simple_example	# needs macro for add_cu_test
		util/cusp_version_test		# Needs macro for add_cu_test
		util/project_settings
		util/surface_integral
		util/surface_projection
		util/viennacl_simple_example	# Needs macro for add_cl_test
		#P3		util/vtk_stencil_visualizer
		util/test_complex
		util/test_template_macros

		template
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
