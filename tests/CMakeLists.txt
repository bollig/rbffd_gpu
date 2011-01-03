
###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
		# find * |grep -v "\." | grep -v "incomplete\/" | grep -v "cut\/"| grep "\/" | grep -v "NOTES" |grep -v "TODO"
		experiments/ncar_poisson1
		experiments/ncar_poisson2
		experiments/ncar_poisson1_nonuniform_diffusivity
		experiments/original_heat
		experiments/regular_square2D_heat
		experiments/regulargrid3D_heat

		gridgen/nested_spheres_ccvt
		
		util/3d_octree
		util/binary_cvt_io
		util/cusp_simple_example
		util/cusp_version_test
		util/project_settings
		util/surface_integral
		util/surface_projection
		util/viennacl_simple_example
		util/vtk_stencil_visualizer
		util/test_complex
		util/test_template_macros

		template
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
