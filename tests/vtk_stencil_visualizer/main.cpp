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
#include <vtkLookupTable.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAnnotatedCubeActor.h>
#include <vtkAssembly.h>
#include <vtkInteractorStyleSwitch.h>
#include <vtkAppendFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>
#include <vtkCamera.h>
#include <vtkLightCollection.h>
#include <vtkLight.h>
#include <vtkLightActor.h>
#include <vtkLightKit.h>

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
#include <vtkLookupTable.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAnnotatedCubeActor.h>
#include <vtkAssembly.h>
#include <vtkInteractorStyleSwitch.h>
#include <vtkAppendFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>
#include <vtkCamera.h>
#include <vtkLightCollection.h>
#include <vtkLight.h>
#include <vtkLightActor.h>
#include <vtkLightKit.h>

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

void GetYData(int sindx, GPU* subdomain, Derivative* der, vtkPoints* cpoints, vtkPoints* points) {
    // Generate a grid of samples for the interpolant
    // This would be equivalent to converting an unstructured grid
    // to a regular grid.

    vtkSmartPointer<vtkPoints> ipoints = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
   // vtkSmartPointer<vtkPoints> cpoints = vtkSmartPointer<vtkPoints>::New(); // Collocation points
    //int sindx = 299;        // NOTE: 299 is interior with 10 neighbors. 0 is boundary with 6 neighbors (due to symmetry forcing)
    StencilType& stencil = subdomain->Q_stencils[sindx];
#if 1
    double* y_weights = der->getYWeights(sindx);
#else
    double *y_weights;
    y_weights = new double[13];
    int iindx = 2;
    y_weights[iindx] = 1.;
    for (int i = 0; i < iindx; i++) {
        y_weights[i] = (double)0;
    }
    for (int i = iindx+1; i < 13; i++) {
        y_weights[i] = (double)0;
    }
#endif
    BasesType& phi = der->getRBFList(sindx);

    int M = 20;
    int N = 20;

    double x0 = -0.25;
    double x1 = 0.25;
    double y0 = -0.25;
    double y1 = 0.25;
    double dx = (x1 - x0)/(M-1);
    double dy = (y1 - y0)/(N-1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int indx = i*N + j;
            Vec3& st_center = subdomain->G_centers[stencil[0]];    // stencil center
            Vec3 disp_point((double)(x0 + i*dx), (double)(y0 + j*dy), 0.);            // Displacement from center
            Vec3 interp_point = st_center + disp_point;    // Interpolation sample points
            double z_val = 0.;
            // Calculate interpolated value of surface
            for (int k = 0; k < stencil.size(); k++) {
                Vec3& center = subdomain->G_centers[stencil[k]];
                // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
                z_val += y_weights[k] * phi[k]->yderiv(center, interp_point);
            }
#if 1
            // Analytic derivatives of monomials
            z_val += y_weights[stencil.size()+0] * 0.;
            z_val += y_weights[stencil.size()+1] * 0.;
            z_val += y_weights[stencil.size()+2] * 1.;
#endif
            ipoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val);
        }
    }

    for (int i = 0; i < stencil.size(); i++) {
        Vec3& interp_point = subdomain->G_centers[stencil[i]];  // interpolate to known stencil nodes
        double z_val = 0.;
        // Calculate interpolated value of surface
        for (int k = 0; k < stencil.size(); k++) {
            Vec3& center = subdomain->G_centers[stencil[k]];
            // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
            z_val += y_weights[k] * phi[k]->yderiv(center, interp_point);
        }
#if 1
            // Analytic derivatives of monomials
            z_val += y_weights[stencil.size()+0] * 0.;
            z_val += y_weights[stencil.size()+1] * 0.;
            z_val += y_weights[stencil.size()+2] * 1.;
#endif
        cpoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val);
    }

    //vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
    points->DeepCopy(ipoints);
    for (int i = 0; i < stencil.size(); i++) {
        points->InsertNextPoint(cpoints->GetPoint(i));
    }
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    char filename[80];
    sprintf(filename, "interpolant_yderiv_%d.vtk", sindx);
    writer->SetFileName(filename);
    writer->SetInput(polydata);
    writer->Write();

}

void GetXData(int sindx, GPU* subdomain, Derivative* der, vtkPoints* cpoints, vtkPoints* points) {
    // Generate a grid of samples for the interpolant
    // This would be equivalent to converting an unstructured grid
    // to a regular grid.

    vtkSmartPointer<vtkPoints> ipoints = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
   // vtkSmartPointer<vtkPoints> cpoints = vtkSmartPointer<vtkPoints>::New(); // Collocation points
    //int sindx = 299;        // NOTE: 299 is interior with 10 neighbors. 0 is boundary with 6 neighbors (due to symmetry forcing)
    StencilType& stencil = subdomain->Q_stencils[sindx];
#if 1
    double* x_weights = der->getXWeights(sindx);
#else
    double *x_weights;
    x_weights = new double[13];
    int iindx = 2;
    x_weights[iindx] = 1.;
    for (int i = 0; i < iindx; i++) {
        x_weights[i] = (double)0;
    }
    for (int i = iindx+1; i < 13; i++) {
        x_weights[i] = (double)0;
    }
#endif
    BasesType& phi = der->getRBFList(sindx);

    int M = 20;
    int N = 20;

    double x0 = -0.25;
    double x1 = 0.25;
    double y0 = -0.25;
    double y1 = 0.25;
    double dx = (x1 - x0)/(M-1);
    double dy = (y1 - y0)/(N-1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int indx = i*N + j;
            Vec3& st_center = subdomain->G_centers[stencil[0]];    // stencil center
            Vec3 disp_point((double)(x0 + i*dx), (double)(y0 + j*dy), 0.);            // Displacement from center
            Vec3 interp_point = st_center + disp_point;    // Interpolation sample points
            double z_val = 0.;
            // Calculate interpolated value of surface
            for (int k = 0; k < stencil.size(); k++) {
                Vec3& center = subdomain->G_centers[stencil[k]];
                // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
                z_val += x_weights[k] * phi[k]->xderiv(center, interp_point);
            }
#if 1
            // Analytic derivatives of monomials
            z_val += x_weights[stencil.size()+0] * 0.;
            z_val += x_weights[stencil.size()+1] * 1.;
            z_val += x_weights[stencil.size()+2] * 0.;
#endif
            ipoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val);
        }
    }

    for (int i = 0; i < stencil.size(); i++) {
        Vec3& interp_point = subdomain->G_centers[stencil[i]];  // interpolate to known stencil nodes
        double z_val = 0.;
        // Calculate interpolated value of surface
        for (int k = 0; k < stencil.size(); k++) {
            Vec3& center = subdomain->G_centers[stencil[k]];
            // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
            z_val += x_weights[k] * phi[k]->xderiv(center, interp_point);
        }
#if 1
     // Analytic derivatives of monomials
        z_val += x_weights[stencil.size()+0] * 0.;
        z_val += x_weights[stencil.size()+1] * 1.;
        z_val += x_weights[stencil.size()+2] * 0.;
#endif
        cpoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val);
    }

    //vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
    points->DeepCopy(ipoints);
    for (int i = 0; i < stencil.size(); i++) {
        points->InsertNextPoint(cpoints->GetPoint(i));
    }
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    char filename[80];
    sprintf(filename, "interpolant_x_%d.vtk", sindx);
    writer->SetFileName(filename);
    writer->SetInput(polydata);
    writer->Write();

}


void GetLaplData(int sindx, GPU* subdomain, Derivative* der, vtkPoints* cpoints, vtkPoints* points) {
    // Generate a grid of samples for the interpolant
    // This would be equivalent to converting an unstructured grid
    // to a regular grid.

    vtkSmartPointer<vtkPoints> ipoints = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
   // vtkSmartPointer<vtkPoints> cpoints = vtkSmartPointer<vtkPoints>::New(); // Collocation points
    //int sindx = 299;        // NOTE: 299 is interior with 10 neighbors. 0 is boundary with 6 neighbors (due to symmetry forcing)
    StencilType& stencil = subdomain->Q_stencils[sindx];
#if 1
    double* l_weights = der->getLaplWeights(sindx);
#else
    double *l_weights;
    l_weights = new double[13];
    int iindx = 2;
    l_weights[iindx] = 1.;
    for (int i = 0; i < iindx; i++) {
        l_weights[i] = (double)0;
    }
    for (int i = iindx+1; i < 13; i++) {
        l_weights[i] = (double)0;
    }
#endif
    BasesType& phi = der->getRBFList(sindx);

    int M = 20;
    int N = 20;

    double x0 = -0.25;
    double x1 = 0.25;
    double y0 = -0.25;
    double y1 = 0.25;
    double dx = (x1 - x0)/(M-1);
    double dy = (y1 - y0)/(N-1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int indx = i*N + j;
            Vec3& st_center = subdomain->G_centers[stencil[0]];    // stencil center
            Vec3 disp_point((double)(x0 + i*dx), (double)(y0 + j*dy), 0.);            // Displacement from center
            Vec3 interp_point = st_center + disp_point;    // Interpolation sample points
            double z_val = 0.;
            // Calculate interpolated value of surface
            for (int k = 0; k < stencil.size(); k++) {
                Vec3& center = subdomain->G_centers[stencil[k]];
                // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
                z_val += l_weights[k] * phi[k]->lapl_deriv(center, interp_point);
            }
#if 0
            // These are disabled because the laplacian of monomials are 0
            z_val += l_weights[stencil.size()+0] * 1.;
            z_val += l_weights[stencil.size()+1] * interp_point.x();
            z_val += l_weights[stencil.size()+2] * interp_point.y();
#endif
            ipoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val);
        }
    }

    for (int i = 0; i < stencil.size(); i++) {
        Vec3& interp_point = subdomain->G_centers[stencil[i]];  // interpolate to known stencil nodes
        double z_val = 0.;
        // Calculate interpolated value of surface
        for (int k = 0; k < stencil.size(); k++) {
            Vec3& center = subdomain->G_centers[stencil[k]];
            // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
            z_val += l_weights[k] * phi[k]->lapl_deriv(center, interp_point);
        }
#if 0
        // These are disabled because the laplacian of monomials are 0
        z_val += l_weights[stencil.size()+0] * 1.;
        z_val += l_weights[stencil.size()+1] * interp_point.x();
        z_val += l_weights[stencil.size()+2] * interp_point.y();
#endif
        cpoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val);
    }

    //vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
    points->DeepCopy(ipoints);
    for (int i = 0; i < stencil.size(); i++) {
        points->InsertNextPoint(cpoints->GetPoint(i));
    }
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    char filename[80];
    sprintf(filename, "interpolant_lapl_%d.vtk", sindx);
    writer->SetFileName(filename);
    writer->SetInput(polydata);
    writer->Write();

}

// Add a visualizer for the sindx'th stencil to the renderer
void AddWeightVisualizer(int sindx, int type, GPU* subdomain, Derivative* der, vtkRenderer* renderer, double* minZ, double* maxZ) {
    vtkSmartPointer<vtkPoints> cpoints = vtkSmartPointer<vtkPoints>::New(); // Collocation points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); // Interpolation points

    // Get the laplacian interpolant
    switch (type) {
    case 0:
        GetXData(sindx, subdomain, der, cpoints, points);
        break;
    case 1:
        GetYData(sindx, subdomain, der, cpoints, points);
        break;
    default:
        GetLaplData(sindx, subdomain, der, cpoints, points);
        break;
    }

    // Now draw it:
    vtkSmartPointer<vtkPolyData> cpolydata = vtkSmartPointer<vtkPolyData>::New();
    cpolydata->SetPoints(cpoints);

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    vtkSmartPointer<vtkDelaunay2D> stencil_delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
    stencil_delaunay->SetInput(polydata);
    //stencil_delaunay->SetTolerance(0.001);
    //stencil_delaunay->SetAlpha(18.0);
    stencil_delaunay->Update();

    vtkSmartPointer<vtkPolyData> outputPolyData = stencil_delaunay->GetOutput();

    double bounds[6];
    outputPolyData->GetBounds(bounds);

    // Find min and max z
    double minz = bounds[4];
    double maxz = bounds[5];

    //std::cout << "minz: " << minz << std::endl;
    //std::cout << "maxz: " << maxz << std::endl;

#if 1
    vtkSmartPointer<vtkElevationFilter> elevation_filter = vtkSmartPointer<vtkElevationFilter>::New();
    elevation_filter->SetInput(stencil_delaunay->GetOutput());
    elevation_filter->SetLowPoint(0., 0., minz);
    elevation_filter->SetHighPoint(0,0, maxz);
#endif
    vtkSmartPointer<vtkDataSetMapper> stencil_mapper3D = vtkSmartPointer<vtkDataSetMapper>::New();
    stencil_mapper3D->SetInput(elevation_filter->GetOutput());
    //stencil_mapper3D->SetInput(outputPolyData->GetOutput());

    vtkSmartPointer<vtkActor> stencil_actor = vtkSmartPointer<vtkActor>::New();
    stencil_actor->SetMapper(stencil_mapper3D);
    //stencil_actor->GetProperty()->SetRepresentationToWireframe();
    stencil_actor->GetProperty()->SetRepresentationToSurface();
    //stencil_actor->GetProperty()->SetRepresentationToPoints();
    stencil_actor->GetProperty()->BackfaceCullingOff();
    // stencil_actor->GetProperty()->LightingOff();
    stencil_actor->GetProperty()->ShadingOn();

    vtkSmartPointer<vtkSphereSource> balls = vtkSmartPointer<vtkSphereSource>::New();
    balls->SetRadius(0.01);
    balls->SetPhiResolution(10);
    balls->SetThetaResolution(10);

    vtkSmartPointer<vtkGlyph3D> glyphPoints = vtkSmartPointer<vtkGlyph3D>::New();
    glyphPoints->SetInput(cpolydata);
    glyphPoints->SetSource(balls->GetOutput());

    vtkSmartPointer<vtkPolyDataMapper> glyphMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    glyphMapper->SetInputConnection(glyphPoints->GetOutputPort());

    vtkSmartPointer<vtkActor> cpoint_actor = vtkSmartPointer<vtkActor>::New();
    cpoint_actor->SetMapper(glyphMapper);
    cpoint_actor->GetProperty()->SetColor(1., 0., 1.);
    cpoint_actor->GetProperty()->LightingOn();

    renderer->AddViewProp( stencil_actor );
    renderer->AddViewProp( cpoint_actor );

    *minZ = minz;
    *maxZ = maxz;
}




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
#if 1
    double epsilon = settings->GetSettingAs<double>("EPSILON");
#else
    double epsilon = 1.85;
#endif
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

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

    double minZ;
    double maxZ;
#if 0
    for (int i = 0; i < subdomain->Q_stencils.size(); i++) {
#else
        for (int i = 0; i < subdomain->Q_stencils.size(); i+=100) {
#endif
            AddWeightVisualizer(i, 2, subdomain, der, renderer, &minZ, &maxZ);
        }
        renderer->GetActiveCamera()->SetFocalPoint(0.,0.,(maxZ + minZ) / 2.);
        renderer->GetActiveCamera()->SetPosition(0.,0.,maxZ+2);
        renderer->GetActiveCamera()->SetParallelProjection(1);

        vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
        renderWindow->SetSize( 800, 800 );

        renderWindow->AddRenderer( renderer );

        vtkSmartPointer<vtkInteractorStyleSwitch> interStyle = vtkSmartPointer<vtkInteractorStyleSwitch>::New();
        interStyle->SetCurrentStyleToTrackballCamera();
        //interStyle->SetCurrentStyleToTrackballActor();

        vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        interactor->SetRenderWindow( renderWindow );
        interactor->SetInteractorStyle(interStyle);
#if 0
        renderer->AutomaticLightCreationOff();

        // lighting the box.
        vtkSmartPointer<vtkLight> l1 = vtkSmartPointer<vtkLight>::New();
        l1->SetPosition(0.0,0.0,-minZ*2.);
        l1->SetFocalPoint(0., 0., 0.);
        l1->SetColor(1.0,1.0,1.0);
        l1->SetPositional(1);
        renderer->AddLight(l1);
        l1->SetSwitch(1);
#endif

#if 0
        // lighting the box.
        vtkSmartPointer<vtkLight> l2 = vtkSmartPointer<vtkLight>::New();
        l2->SetPosition(0.0,0.0,maxZ*2.);
        l2->SetFocalPoint(0., 0., 0.);
        l2->SetColor(1.0,1.0,1.0);
        l2->SetPositional(1);
        renderer->AddLight(l2);
        l2->SetSwitch(1);
#endif

#if 0
        vtkSmartPointer<vtkLightCollection> lights = renderer->GetLights();
        lights->InitTraversal();
        vtkSmartPointer<vtkLight> l = lights->GetNextItem( );
        while(l!=0)
        {
            double angle=l->GetConeAngle();
            if(l->LightTypeIsSceneLight() && l->GetPositional()
                && angle<180.0) // spotlight
                {
                vtkLightActor *la=vtkLightActor::New();
                la->SetLight(l);
                renderer->AddViewProp(la);
                la->Delete();
            }
            l->SetIntensity(l->GetIntensity()*2.);
            l=lights->GetNextItem();
        }
#endif

        vtkSmartPointer<vtkLightKit> lightKit = vtkSmartPointer<vtkLightKit>::New();
        lightKit->AddLightsToRenderer(renderer);

        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
        vtkSmartPointer<vtkProperty> property = axes->GetXAxisTipProperty();
        property->SetRepresentationToSurface();
        property->SetDiffuse(0);
        property->SetAmbient(1);
        property->SetColor( 1, 0, 0 );

        property = axes->GetYAxisTipProperty();
        property->SetRepresentationToSurface();
        property->SetDiffuse(0);
        property->SetAmbient(1);
        property->SetColor(0 , 1, 0 );

        property = axes->GetZAxisTipProperty();
        property->SetRepresentationToSurface();
        property->SetDiffuse(0);
        property->SetAmbient(1);
        property->SetColor( 0, 0, 1 );

        vtkSmartPointer<vtkOrientationMarkerWidget> widget  = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        vtkSmartPointer<vtkAnnotatedCubeActor> cube = vtkSmartPointer<vtkAnnotatedCubeActor>::New();
        cube->SetFaceTextScale(0.5);
        property = cube->GetCubeProperty();
        property->SetColor( 0.5, 1, 1 );
        property = cube->GetTextEdgesProperty();
        property->SetLineWidth( 1 );
        property->SetDiffuse( 0 );
        property->SetAmbient( 1 );
        property->SetColor( 0.1800, 0.2800, 0.2300 );
        property = cube->GetXPlusFaceProperty();
        property->SetColor(0, 0, 1);
        property->SetInterpolationToFlat();
        property = cube->GetXMinusFaceProperty();
        property->SetColor(0, 0, 1);
        property->SetInterpolationToFlat();
        property = cube->GetYPlusFaceProperty();
        property->SetColor(0, 1, 0);
        property->SetInterpolationToFlat();
        property = cube->GetYMinusFaceProperty();
        property->SetColor(0, 1, 0);
        property->SetInterpolationToFlat();
        property = cube->GetZPlusFaceProperty();
        property->SetColor(1, 0, 0);
        property->SetInterpolationToFlat();
        property = cube->GetZMinusFaceProperty();
        property->SetColor(1, 0, 0);
        property->SetInterpolationToFlat();

        vtkSmartPointer<vtkAssembly> assemble = vtkSmartPointer<vtkAssembly>::New();
        assemble->AddPart(cube);
        assemble->AddPart(axes);
        widget->SetOrientationMarker(assemble);
        widget->SetCurrentRenderer(renderer);
        widget->SetInteractor(interactor);
        widget->SetEnabled(1);
        widget->SetInteractive(1);

        //renderer->AddViewProp( assemble );

        renderWindow->Render();

        interactor->Initialize();
        interactor->Start();

        return EXIT_SUCCESS;
    }
