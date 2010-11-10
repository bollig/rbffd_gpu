#include <vector>
#include <cmath>

#if 0
#include "stencil_visualizer.h"


// Startup
int main( int argc, char *argv[] )
{
    // Load data into:
    //      1) vector<vector<Vec3> > Stencil_nodes
    //      2) vector<vector<double> > Stencil_weights
    //      3) vector<vector<RBF> > Stencil_bases
    // NOTE: 3 includes details of the RBF support parameter internally

    NodeListType* stencil_nodes;
    WeightListType* stencil_weights;
    BasesListType* stencil_bases;



    StencilVisualizer* visualizer = new StencilVisualizer(argc, argv);

    // Load data into visualizer. Visualizer will construct a vector of VisualStencils
    visualizer->setNodes(stencil_nodes);
    visualizer->setWeights(stencil_weights);
    visualizer->setBases(stencil_bases);
    visualizer->setupVisualStencils();

    // Setup our rendering callbacks
    visualizer->setupCallbacks();

    // Start the viewer window and run the callbacks
    visualizer->run();

    return 0;
}
#endif

#include <QtGui/QApplication>

#include "stencil_visualizer.h"

int main( int argc, char **argv )
{
    //QApplication::setColorSpec( QApplication::CustomColor );
    QApplication::setColorSpec( QApplication::ManyColor );
    QApplication a( argc, argv );

    if ( !QGLFormat::hasOpenGL() ) {
        qWarning( "This system has no OpenGL support. Exiting." );
        return -1;
    }

    StencilVisualizer w1;
    w1.setWindowTitle("W1");

//    StencilVisualizer w2;
//    w2.setWindowTitle("W2");

    w1.show();
//    w2.show();

    // NOTE: if we raise in reverse order of how they are shown, QT
    // will display them all above our terminal and with the last shown
    // on the top of the stack. If we go in order then the last one
    // will be shown above and the rest are below the terminal.
//    w2.raise();
    w1.raise();

    // We resize the window to what we want here. Inside the visualizer
    // we have to specify a different dimension so the window shape changes
    // when this is called. this forces a resizeGL call and updates the view
    // of our data so the perspective is orthogonal.
    w1.resize(800, 600);

    return a.exec();
}


