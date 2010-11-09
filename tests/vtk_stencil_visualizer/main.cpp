#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCellArray.h>
#include <vtkMath.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRectilinearGridSource.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkVoxelContoursToSurfaceFilter.h>
#include <vtkPolyDataWriter.h>
#include <vtkDelaunay2D.h>
#include <vtkElevationFilter.h>
#include <vtkDataSetMapper.h>

// INTERESTING: the poisson include must come first. Otherwise I get an
// error in the constant definitions for MPI. I wonder if its because
// nested_sphere_cvt.h accidentally overrides one of the defines for MPI
//#include "ncar_poisson1.h"
//#include "ncar_poisson1_cusp.h"
//#include "ncar_poisson1_cl.h"
#include "nonuniform_poisson1_cl.h"
#include "grid.h"
#include "nested_sphere_cvt.h"
#include "cvt.h"
#include "gpu.h"
#include "derivative.h"
#include "derivative_tests.h"
#include "exact_solution.h"
#include "exact_ncar_poisson1.h"
#include "exact_ncar_poisson2.h"
#include "communicator.h"
#include "projectsettings.h"


int main(int argc, char ** argv)
{
    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

    // Discrete energy divided by number of sample pts
    double energy;

    // L2 norm of difference between iteration output
    double it_diff;

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num =0;      // Total number of iterations taken.

    int dim = settings->GetSettingAs<int>("DIMENSION");

    // Generate the CVT if the file doesnt already exist
    CVT* cvt = new NestedSphereCVT(settings);
    int load_errors = cvt->cvt_load(-1);
    if (load_errors) { // File does not exist
        cvt->cvt(&it_num, &it_diff, &energy);
    }

    // TODO: run this in parallel:
    double* generators = cvt->getGenerators();
    Grid* grid = new Grid(settings);
    // Compute stencils given a set of generators
    grid->computeStencils(generators, cvt->getKDTree());

    GPU* subdomain = new GPU(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());      // TODO: get these extents from the cvt class (add constructor to GPU)

    // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
    // Remove need for extents in constructor.
    subdomain->fillLocalData(grid->getRbfCenters(), grid->getStencil(), grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
    subdomain->fillVarData(grid->getRbfCenters()); // Sets function values in U

    // Verbosely print the memberships of all nodes within the subdomain
    //subdomain->printCenterMemberships(subdomain->G, "G");

    // 0: 2D problem; 1: 3D problem
    ExactSolution* exact_poisson;
    if (dim == 3) {
        exact_poisson = new ExactNCARPoisson1();        // 3D problem is not verified yet
    } else {
        exact_poisson = new ExactNCARPoisson2();        // 2D problem works with uniform diffusion
    }

    // Clean this up. Have the Poisson class construct Derivative internally.
    Derivative* der = new Derivative(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size());
    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        DerivativeTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
    }

    for (int i = 0; i < subdomain->Q_stencils.size(); i++) {
        //subdomain->printStencil(subdomain->Q_stencils[i], "Q[i]");
        // Compute all derivatives for our centers and return the number of
        // weights that will be available
        der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[i], i, dim);
    }

    // Generate a grid of samples for the interpolant
    // This would be equivalent to converting an unstructured grid
    // to a regular grid.

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    StencilType& stencil = subdomain->Q_stencils[0];
    double* x_weights = der->getXWeights(0);
    for (int j = 0; j < stencil.size(); j++) {
        points->InsertNextPoint(subdomain->G_centers[stencil[j]].x()*25, subdomain->G_centers[stencil[j]].y()*25, x_weights[j]);
    }

    // Store grid and polys as poly data

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName("stencil_points_x.vtk");
    writer->SetInput(polydata);
    writer->Write();

    vtkSmartPointer<vtkDelaunay2D> stencil_delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
    stencil_delaunay->SetInput(polydata);
    stencil_delaunay->SetTolerance(0.001);
    stencil_delaunay->SetAlpha(18.0);
    stencil_delaunay->Update();

#if 1
    vtkSmartPointer<vtkElevationFilter> elevation_filter = vtkSmartPointer<vtkElevationFilter>::New();
    elevation_filter->SetInput(stencil_delaunay->GetOutput());
    elevation_filter->SetLowPoint(0., 0., -6.);
    elevation_filter->SetHighPoint(0,0,13.);
#endif
    vtkSmartPointer<vtkDataSetMapper> stencil_mapper3D = vtkSmartPointer<vtkDataSetMapper>::New();
    stencil_mapper3D->SetInput(elevation_filter->GetOutput());

    vtkSmartPointer<vtkActor> stencil_actor = vtkSmartPointer<vtkActor>::New();
    stencil_actor->SetMapper(stencil_mapper3D);




    // Compute interpolated value as Z component
#if 0

    vtkSmartPointer<vtkPolyDataMapper> surfaceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    surfaceMapper->SetInputConnection( poly->GetOutputPort() );
    surfaceMapper->ScalarVisibilityOff();
    surfaceMapper->ImmediateModeRenderingOn();

    vtkSmartPointer<vtkActor> surfaceActor = vtkSmartPointer<vtkActor>::New();
    surfaceActor->SetMapper( surfaceMapper );
    surfaceActor->GetProperty()->SetRepresentationToWireframe();
    surfaceActor->GetProperty()->ShadingOff();
#endif
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize( 800, 400 );

    renderWindow->AddRenderer( renderer );

    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow( renderWindow );

    renderer->AddViewProp( stencil_actor );
    renderWindow->Render();

    interactor->Start();

    return EXIT_SUCCESS;
}