#include "stencil_visualizer.h"

VisualStencil::VisualStencil(std::vector<Vec3> nodes, std::vector<double> weights) {

}


VisualStencil::~VisualStencil() {

}


void VisualStencil::Draw() {

}



void VisualStencil::Select() {

}

void VisualStencil::Deselect() {

}


void VisualStencil::GetInterpolationPoints() {
#if 0

    // Evan: HERE we interpolate grid to stencil using a cartesian overlay, but we want to do it in radial fashion (see bottom of this file).
    // Interpolate grid heights
    for ( int x=-GRID_W/2; x<GRID_W/2; ++x )
    {
        for ( int z=-GRID_H/2; z<GRID_H/2; ++z )
        {
            double h = mtx_v(p+0, 0) + mtx_v(p+1, 0)*x + mtx_v(p+2, 0)*z;
            Vec3 pt_i, pt_cur(x,0,z);
            for ( unsigned i=0; i<p; ++i )
            {
                pt_i = control_points[i];
                pt_i.y = 0;
                h += mtx_v(i,0) * tps_base_func( ( pt_i - pt_cur ).len());
            }
            grid[x+GRID_W/2][z+GRID_H/2] = h;
        }
    }
#endif
}

void VisualStencil::DrawInterpolation() {

}


void VisualStencil::GetCartesianInterpolationPoints() {

}


#if 0
// From: http://www.cocos2d-iphone.org/forum/topic/2207
// Calculates vertices of triangle strip for a disk (behaves like gluDisk)
// All we need to do is add a Z component and get the interpolation points.
(GLfloat *) calculateSegmentPoints {

    float step = (2*M_PI) / NR_OF_SEGMENTS;

    float *vertices = malloc( sizeof(float)*4*(NR_OF_SEGMENTS+1));
    if( ! vertices )
        return 0;

    memset( vertices,0, sizeof(float)*4*(NR_OF_SEGMENTS+1));

    int count = 0;
    for( int i = 0; i <= NR_OF_SEGMENTS; i++ ) {
        // calculating the current vertice on the outer side of the segment
        float outerRads = i*step;
        float outerX = OUTER_RADIUS * cos( outerRads );
        float outerY = OUTER_RADIUS * sin( outerRads );
        vertices[count++]	= outerX;
        vertices[count++]	= outerY;

        // calculating the current vertice on the inner side of the segment
        float innerRads = i*step;
        float innerX = INNER_RADIUS * cos( innerRads );
        float innerY = INNER_RADIUS * sin( innerRads );
        vertices[count++]		= innerX;
        vertices[count++]		= innerY;
    }
    return vertices;
}
#endif
