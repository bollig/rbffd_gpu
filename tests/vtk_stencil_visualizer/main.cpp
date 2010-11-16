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
    der->setEpsilon(2.5);

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

    vtkSmartPointer<vtkPoints> ipoints = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
    vtkSmartPointer<vtkPoints> cpoints = vtkSmartPointer<vtkPoints>::New(); // Collocation points
    int sindx = 299;
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
    double scale = 1.;
    double min_weight = 100000.;
    double max_weight = 0.;

    for (int k = 0; k < stencil.size()+3; k++) {
        if (min_weight > fabs(l_weights[k])) { min_weight = fabs(l_weights[k]); }
        if (max_weight < fabs(l_weights[k])) { max_weight = fabs(l_weights[k]); }
    }
    cout << "Min Weight: " << min_weight << endl;
    cout << "Max weight: " << max_weight << endl;
#if 0
    if (min_weight > 0) {
        scale = (max_weight / min_weight) * 0.1;
    }
#endif
    cout << "Scale: " << scale << endl;

    double x0 = -0.25;
    double x1 = 0.25;
    double y0 = -0.25;
    double y1 = 0.25;
    double dx = (x1 - x0)/(M-1);
    double dy = (y1 - y0)/(N-1);
    double *z_val = new double[M*N];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int indx = i*N + j;
            Vec3& st_center = subdomain->G_centers[stencil[0]];    // stencil center
            Vec3 disp_point((double)(x0 + i*dx), (double)(y0 + j*dy), 0.);            // Displacement from center
            Vec3 interp_point = st_center + disp_point;    // Interpolation sample points
            z_val[indx] = 0.;
            // Calculate interpolated value of surface
            for (int k = 0; k < stencil.size(); k++) {
                Vec3& center = subdomain->G_centers[stencil[k]];
                // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
                z_val[indx] += l_weights[k] * phi[k]->eval(center, interp_point);
            }
            z_val[indx] += l_weights[stencil.size()+0] * 1.;
            z_val[indx] += l_weights[stencil.size()+1] * interp_point.x();
            z_val[indx] += l_weights[stencil.size()+2] * interp_point.y();

            //            cout << "Z_VAL = " << z_val[indx] << endl;
            //points->InsertNextPoint(interp_point.x(), interp_point.y(), phi[0]->lapl_deriv(center, interp_point));
            //points->InsertNextPoint(interp_point.x(), interp_point.y(), (center-interp_point).magnitude())   ;
            ipoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val[indx]*scale);
        }
    }

    cout << "Interpolant for Stencil nodes"<<endl;
    for (int i = 0; i < stencil.size(); i++) {
        Vec3& interp_point = subdomain->G_centers[stencil[i]];  // interpolate to known stencil nodes
        double z_val2 = 0.;
        // Calculate interpolated value of surface
        cout << "StencilWeights: " << endl;
        for (int k = 0; k < stencil.size(); k++) {
            Vec3& center = subdomain->G_centers[stencil[k]];
            // z_val = Sum_{k=0}^{n} phi_k(x,y) * w(k)
            z_val2 += l_weights[k] * phi[k]->eval(center, interp_point);
            cout << l_weights[k] << endl;
        }
        z_val2 += l_weights[stencil.size()+0] * 1.;
        cout << l_weights[stencil.size()+0] << endl;
        z_val2 += l_weights[stencil.size()+1] * interp_point.x();
        cout << l_weights[stencil.size()+1] << endl;
        z_val2 += l_weights[stencil.size()+2] * interp_point.y();
        cout << l_weights[stencil.size()+2] << endl;

        cout << "Z_VAL = " << z_val2*scale; interp_point.print("\tNode: ");
        //points->InsertNextPoint(interp_point.x(), interp_point.y(), phi[0]->lapl_deriv(center, interp_point));
        //points->InsertNextPoint(interp_point.x(), interp_point.y(), (center-interp_point).magnitude())   ;
        cpoints->InsertNextPoint(interp_point.x(), interp_point.y(), z_val2*scale);
    }

    // Store grid and polys as poly data
    vtkSmartPointer<vtkPolyData> ipolydata = vtkSmartPointer<vtkPolyData>::New();
    ipolydata->SetPoints(ipoints);

    vtkSmartPointer<vtkPolyData> cpolydata = vtkSmartPointer<vtkPolyData>::New();
    cpolydata->SetPoints(cpoints);

#if 0
    vtkSmartPointer<vtkAppendFilter> appender = vtkSmartPointer<vtkAppendFilter>::New();
    appender->AddInput(cpolydata);
    appender->AddInput(ipolydata);

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(appender->GetOutput()->GetPoints());
#else
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); // Interpolation points
    points->DeepCopy(ipoints);
    for (int i = 0; i < stencil.size(); i++) {
        points->InsertNextPoint(cpoints->GetPoint(i));
    }
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
#endif


    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName("stencil_points_x.vtk");
    writer->SetInput(polydata);
    writer->Write();

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

    std::cout << "minz: " << minz << std::endl;
    std::cout << "maxz: " << maxz << std::endl;

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
    stencil_actor->GetProperty()->LightingOff();
    //stencil_actor->GetProperty()->ShadingOff();


    vtkSmartPointer<vtkSphereSource> balls = vtkSmartPointer<vtkSphereSource>::New();
    balls->SetRadius(0.02);
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

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize( 800, 400 );

    renderWindow->AddRenderer( renderer );

    vtkSmartPointer<vtkInteractorStyleSwitch> interStyle = vtkSmartPointer<vtkInteractorStyleSwitch>::New();
    interStyle->SetCurrentStyleToTrackballCamera();
    //interStyle->SetCurrentStyleToTrackballActor();

    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow( renderWindow );
    interactor->LightFollowCameraOn();
    interactor->SetInteractorStyle(interStyle);

    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

    vtkSmartPointer<vtkOrientationMarkerWidget> widget  = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    vtkSmartPointer<vtkAnnotatedCubeActor> cube = vtkSmartPointer<vtkAnnotatedCubeActor>::New();
    cube->SetFaceTextScale(0.5);
    vtkSmartPointer<vtkAssembly> assemble = vtkSmartPointer<vtkAssembly>::New();
    assemble->AddPart(cube);
    assemble->AddPart(axes);
    widget->SetOrientationMarker(assemble);
    widget->SetCurrentRenderer(renderer);
    widget->SetInteractor(interactor);
    widget->SetEnabled(1);
    widget->SetInteractive(0);

    renderer->AddViewProp( stencil_actor );
    renderer->AddViewProp( cpoint_actor );
    renderer->AddViewProp( assemble );

    renderWindow->Render();

    interactor->Initialize();
    interactor->Start();

    return EXIT_SUCCESS;
}
