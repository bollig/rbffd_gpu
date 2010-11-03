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

/*
 * Copyright (c) 1993-1997, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(R) is a registered trademark of Silicon Graphics, Inc.
 */

// File:
//  robot.cpp
//
// Description:
//
// This program shows how to composite modeling transformations
// to draw translated and rotated hierarchical models.
// Interaction:  pressing the s and e keys (shoulder and elbow)
// alters the rotation of the robot arm.
// Author:
//
//   Travis Astle   Modified use Qt instead of glut.
//   Oct 19, 2001
//

#include <QtGui/QApplication>

#include "stencil_visualizer.h"

/*  Main Loop
 *  Open window with initial window size, title bar,
 *  color index display mode, and handle input events.
 */
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

    StencilVisualizer w2;
    w2.setWindowTitle("W2");

    w1.show();
    w2.show();

    // NOTE: if we raise in reverse order of how they are shown, QT
    // will display them all above our terminal and with the last shown
    // on the top of the stack. If we go in order then the last one
    // will be shown above and the rest are below the terminal.
    w2.raise();
    w1.raise();

    return a.exec();
}


