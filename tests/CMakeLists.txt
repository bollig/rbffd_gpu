
###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
		# find * |grep -v "\." | grep -v "incomplete\/" | grep -v "cut\/"| grep "\/" | grep -v "NOTES" |grep -v "TODO"
		#P2		experiments/ncar_poisson1
		#P2		experiments/ncar_poisson1
		#P2		experiments/ncar_poisson2
		#*		experiments/ncar_poisson1_nonuniform_diffusivity
		#*		experiments/original_heat
		#P1		experiments/ncar_poisson1_regular_grid
		#P2		experiments/regular_square2D_heat
		#P2		experiments/regulargrid3D_heat
		#TODO		experiments/convection_regulargrid
		experiments/heat_regulargrid_2d
		experiments/heat_1d_verification
		experiments/heat_1d_gpu
		experiments/heat_1d_cusp
		experiments/heat_1d_nonuniform_diffusion
		#TODO		experiments/sarler_vertnick_2006
		experiments/heat_2d_verification
		experiments/heat_ellipse_2d
		experiments/vortex_rollup_on_sphere
		experiments/cosine_bell_on_sphere
		experiments/cosine_bell_vcl
		experiments/stokes_equation
		experiments/ncar_poisson1_vcl

		#****		gridgen/nested_spheres_ccvt
		gridgen/regular_grid
		gridgen/square_cvt
		gridgen/stencil_gen
		gridgen/subdomain_gen
		# HIGH PRIORITY: Test Heat on unit disk, then warp into Ellipse
		
		util/3d_octree
		#		util/binary_cvt_io
		util/cusp_simple_example	# needs macro for add_cu_test
		util/cusp_version_test		# Needs macro for add_cu_test
		util/project_settings
		util/surface_integral
		util/surface_projection
		util/viennacl_simple_example	# Needs macro for add_cl_test
		util/viennacl_spmv_benchmark
		util/viennacl_gmres_poisson
		util/viennacl_1d_pgmres
		util/viennacl_stokes_pgmres
		util/cusp_1d_pgmres
		#util/vtk_stencil_visualizer
		util/test_complex
		util/test_template_macros
		util/gpu_compute_derivs
		util/gpu_compute_multiple_derivs
		util/gpu_compute_multiple_derivs_weight
		util/gpu_compute_multi_derivs_weight_fun
		util/gpu_compute_multi_derivs_weight_fun4
		util/gpu_deriv_fun
		util/matrix_market_io


		util/cusp_spmv_benchmark
		util/cusp_gmres_poisson
		util/cusp_stokes_steady

		unified

		template
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
