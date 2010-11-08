#include <QtOpenGL/qgl.h>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtGui/qmatrix4x4.h>
#include <QtGui/qvector3d.h>
#include <cmath>

#include "stencil_visualizer.h"
#include "visual_stencil.h"
#include "trackball.h"


static void multMatrix(const QMatrix4x4& m)
{
    // static to prevent glMultMatrixf to fail on certain drivers
    static GLfloat mat[16];
    const qreal *data = m.constData();
    for (int index = 0; index < 16; ++index)
        mat[index] = data[index];
    glMultMatrixf(mat);
}


StencilVisualizer::StencilVisualizer( QWidget *parent,  QGLWidget *shareWidget) :
        QGLWidget(parent, shareWidget)
{
    clearColor = Qt::red;

    // Create a trackball sphere for our viewer
    trackball = TrackBall(TrackBall::Sphere);

    // Setup a timer to autorefresh the window
    m_timer = new QTimer(this);
    m_timer->setInterval(20);
    connect(m_timer, SIGNAL(timeout()), this, SLOT(update()));
    m_timer->start();
};


QSize StencilVisualizer::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize StencilVisualizer::sizeHint() const
{
    // We want the hint to be an odd size so when we resize
    // the window to be something straightforward like 800x600
    // the window shape will change and a call to resizeGL will
    // execute.
    return QSize(801, 600);
}



// Initialize function adapted from existing code

void StencilVisualizer::initializeGL(void)
{
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glShadeModel (GL_FLAT);
}

// Wire cube program to replace glutWireCube();

void StencilVisualizer::wireCube(float length)
{
    glBegin(GL_LINE_LOOP);
    glVertex3f(-length/2, -length/2, length/2);
    glVertex3f(length/2, -length/2, length/2);
    glVertex3f(length/2, length/2, length/2);
    glVertex3f(-length/2, length/2, length/2);
    glEnd();

    glBegin(GL_LINE_LOOP);
    glVertex3f(-length/2, -length/2, -length/2);
    glVertex3f(length/2, -length/2, -length/2);
    glVertex3f(length/2, length/2, -length/2);
    glVertex3f(-length/2, length/2, -length/2);
    glEnd();

    glBegin(GL_LINES);
    glVertex3f(-length/2, -length/2, length/2);
    glVertex3f(-length/2, -length/2, -length/2);

    glVertex3f(length/2, -length/2, length/2);
    glVertex3f(length/2, -length/2, -length/2);

    glVertex3f(length/2, length/2, length/2);
    glVertex3f(length/2, length/2, -length/2);

    glVertex3f(-length/2, length/2, length/2);
    glVertex3f(-length/2, length/2, -length/2);
    glEnd();
}

void StencilVisualizer::setClearColor(const QColor &color)
{
    clearColor = color;
    updateGL();
}

// paint function adapted from original display function

void StencilVisualizer::paintGL(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();


    // Draw in Center of screen


    // Draw in Bottom corner
    glPushMatrix();
    glTranslatef(-4.0f, -4.0f, 0.0f);
    drawAxes();
    glPopMatrix();
}

void StencilVisualizer::drawAxes()
{
    // Setup the rotation of the axis to match the camera orbit
    glPushMatrix();
    QMatrix4x4 m;
    m.rotate(trackball.rotation());
    multMatrix(m);

    // Axes: (x,y,z)=(r,g,b)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPushMatrix();
    glRotatef(90.0, 0.0, 1.0, 0.0);
    glColor3ub( 255, 0, 0 );
    GLUquadric* quad = gluNewQuadric();
    gluCylinder(quad, 0.02, 0.02, 1, 5, 1);
    glPopMatrix();
    glPushMatrix();
    glRotatef(-90.0, 1.0, 0.0, 0.0);
    glColor3ub( 0, 255,  0 );
    gluCylinder(quad, 0.02, 0.02, 1, 5, 1);
    glPopMatrix();
    glPushMatrix();
    glRotatef(0.0, 1.0, 0.0, 0.0);
    glColor3ub( 0, 0, 255 );
    gluCylinder(quad, 0.02, 0.02, 1, 5, 1);
    glPopMatrix();

    glPopMatrix();
}

// Resize function adapted from reshape function
void StencilVisualizer::resizeGL(int width, int height)
{
    int side = qMin(width, height);

    // Clip window so we have a square viewing area
    //glViewport((width - side) / 2, (height - side) / 2, side, side);

    // 1:1 Proportional view that fills the window
    glViewport(0, 0, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-5, +5, -5, +5, -4.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
}


// Key press event function to replace keyboard function.  This
// function is modified to only take one argument (the keyboard input)
// instead of receiving three, as in the keyboard function.

void StencilVisualizer::keyPressEvent(QKeyEvent *e)
{
    switch (e->key()) {
    case 'q':
        this->close();
        break;
    case 'Q':
        this->close();
        break;
    case 's':
        shoulder = (shoulder + 5) % 360;
        updateGL();
        break;
    case 'S':
        shoulder = (shoulder - 5) % 360;
        updateGL();
        break;
    case 'e':
        elbow = (elbow + 5) % 360;
        updateGL();
        break;
    case 'E':
        elbow = (elbow - 5) % 360;
        updateGL();
        break;
    case 27:    // Not sure what this key is. Is it ESC? Doesnt work on OSX, thats for sure.
        exit(0);
        break;
    default:
        break;
    }
}

/** Trackball details borrowed from the boxes demo froom Qt4 */

QPointF StencilVisualizer::pixelPosToViewPos(const QPointF& p)
{
    return QPointF(2.0 * float(p.x()) / width() - 1.0,
                   1.0 - 2.0 * float(p.y()) / height());
}

void StencilVisualizer::mouseMoveEvent(QMouseEvent *event)
{
    QGLWidget::mouseMoveEvent(event);
    if (event->isAccepted())
        return;

    if (event->buttons() & Qt::LeftButton) {
        trackball.move(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
        event->accept();
    } else {
        trackball.release(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
    }

    // No need to support different buttons right now
#if 0
    if (event->buttons() & Qt::RightButton) {
        trackball.move(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
        event->accept();
    } else {
        trackball.release(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
    }

    if (event->buttons() & Qt::MidButton) {
        trackball.move(pixelPosToViewPos(event->pos()), QQuaternion());
        event->accept();
    } else {
        trackball.release(pixelPosToViewPos(event->pos()), QQuaternion());
    }
#endif
    // No need to call this because we use an QTimer to auto update the window
    // updateGL();
}

void StencilVisualizer::mousePressEvent(QMouseEvent *event)
{
    QGLWidget::mousePressEvent(event);
    if (event->isAccepted())
        return;

    if (event->buttons() & Qt::LeftButton) {
        trackball.push(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
        event->accept();
    }

    // No need to support different buttons right now.
#if 0
    if (event->buttons() & Qt::RightButton) {
        trackball.push(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
        event->accept();
    }

    if (event->buttons() & Qt::MidButton) {
        trackball.push(pixelPosToViewPos(event->pos()), QQuaternion());
        event->accept();
    }
#endif
    // No need to call this because we use an QTimer to auto update the window
    // updateGL();
}

void StencilVisualizer::mouseReleaseEvent(QMouseEvent *event)
{
    QGLWidget::mouseReleaseEvent(event);
    if (event->isAccepted())
        return;

    if (event->button() == Qt::LeftButton) {
        trackball.release(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
        event->accept();
    }

    // No need to support different buttons right now
#if 0
    if (event->button() == Qt::RightButton) {
        trackball.release(pixelPosToViewPos(event->pos()), trackball.rotation().conjugate());
        event->accept();
    }

    if (event->button() == Qt::MidButton) {
        trackball.release(pixelPosToViewPos(event->pos()), QQuaternion());
        event->accept();
    }
#endif
    // No need to call this because we use an QTimer to auto update the window
    // updateGL();
}

void StencilVisualizer::wheelEvent(QWheelEvent * event)
{
    QGLWidget::wheelEvent(event);
    if (!event->isAccepted()) {
        m_distExp += event->delta();
        if (m_distExp < -8 * 120)
            m_distExp = -8 * 120;
        if (m_distExp > 10 * 120)
            m_distExp = 10 * 120;
        event->accept();
    }
    // No need to call this because we use an QTimer to auto update the window
    // updateGL();
}


#if 0

StencilVisualizer::StencilVisualizer(int argc, char** argv)
{
    winW = 800;
    winH = 600;
    screen_dirty = true;

    initializeWindow(argc, argv);
    glCheckErrors();


    glCheckErrors();

    setupLights();
    glCheckErrors();

    create_menu() ;
    glCheckErrors();

}

StencilVisualizer::~StencilVisualizer()
{

}


void StencilVisualizer::run()
{
    glutMainLoop();
}

void GLApp::Callbacks::display()
{
#if 0
    unsigned i;

    // Uploads a Vec to OGL
#define UPLOADVEC(v) glVertex3f( v.x, v.y, v.z )

    static GLfloat color1[] = {0.8, 0.8, 0.8, 1.0};
    static GLfloat color2[] = {1.0, 1.0, 1.0, 1.0};

    static GLfloat red[] = {1.0, 0.0, 0.0, 1.0};
    static GLfloat green[] = {0.0, 1.0, 0.0, 1.0};
    static GLfloat blue[] = {0.0, 0.0, 1.0, 1.0};

    // Make a rotation matrix out of mouse point

    const Mtx& rot = rotateY( camBeta ) * rotateX( camAlpha );

    // Rotate camera
    Vec3 cam_loc(0,0,-150), cam_up(0,1,0);
    cam_loc = cam_loc * rot;
    cam_up = cam_up * rot;

    // Clear the screen
    glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );

    // Prepare zoom by changiqng FOV
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float fov = 45 + camZoom;
    if ( fov < 5 ) fov = 5;
    if ( fov > 160 ) fov = 160;
    gluPerspective( fov, (float)winW/(float)winH, 1.0, 500.0 );

    gluLookAt( cam_loc.x, cam_loc.y, cam_loc.z,  // eye
               0, 0, 0, // target
               cam_up.x, cam_up.y, cam_up.z ); // up

    // Curve surface
    glEnable(GL_LIGHTING) ;
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBegin( GL_QUADS );

    static GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    static GLfloat mat_shininess[] = { 100.0 };

    for ( int x=-GRID_W/2; x<GRID_W/2-1; ++x )
    {
        for ( int z=-GRID_H/2; z<GRID_H/2-1; ++z )
        {
            if ( (x&8)^(z&8) )
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color1);
            else
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color2);

            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
            glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

            // Evan: HERE we draw the cartesian grid overlay.
            float a[] = { x+0, grid[x+0+GRID_W/2][z+0+GRID_H/2], z+0 };
            float b[] = { x+0, grid[x+0+GRID_W/2][z+1+GRID_H/2], z+1 };
            float c[] = { x+1, grid[x+1+GRID_W/2][z+1+GRID_H/2], z+1 };
            float d[] = { x+1, grid[x+1+GRID_W/2][z+0+GRID_H/2], z+0 };

#define V_MINUS(A,B) {A[0]-B[0], A[1]-B[1], A[2]-B[2]}
#define V_CROSS(A,B) \
            {A[1]*B[2]-A[2]*B[1], \
             A[2]*B[0]-A[0]*B[2], \
             A[0]*B[1]-A[1]*B[0]}
            float ab[] = V_MINUS(a,b);
            float cb[] = V_MINUS(c,b);
            float n[] = V_CROSS( cb, ab );

            glNormal3f( n[0],n[1],n[2] );
            glVertex3f( a[0],a[1],a[2] );
            glVertex3f( b[0],b[1],b[2] );
            glVertex3f( c[0],c[1],c[2] );
            glVertex3f( d[0],d[1],d[2] );
        }
    }
    glEnd();

#if 0 // visual helpers

    glDisable(GL_LIGHTING) ;

    // Flat grid
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBegin( GL_QUADS );
    glColor3ub( 128, 128, 128 );

    for ( int x=-GRID_W/2; x<GRID_W/2-4; x+=5 )
    {
        for ( int z=-GRID_H/2; z<GRID_H/2-4; z+=5 )
        {
            glVertex3f( x-0.5, -0.5f, z-0.5 );
            glVertex3f( x-0.5, -0.5f, z+4.5 );
            glVertex3f( x+4.5, -0.5f, z+4.5 );
            glVertex3f( x+4.5, -0.5f, z-0.5 );
        }
    }
    glEnd();

    // Axes: (x,y,z)=(r,g,b)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPushMatrix();
    glRotatef(90.0, 0.0, 1.0, 0.0);
    glColor3ub( 255, 0, 0 );
    glutSolidCone(0.5, 80.0, 5, 1);
    glPopMatrix();
    glPushMatrix();
    glRotatef(-90.0, 1.0, 0.0, 0.0);
    glColor3ub( 0, 255,  0 );
    glutSolidCone(0.5, 80.0, 5, 1);
    glPopMatrix();
    glPushMatrix();
    glRotatef(0.0, 1.0, 0.0, 0.0);
    glColor3ub( 0, 0, 255 );
    glutSolidCone(0.5, 80.0, 5, 1);
    glPopMatrix();

#endif

    // Control points
    int old_sel = selected_cp;
    if ( mouseState[0] == mouseState[1] == mouseState[2] == 0 )
        selected_cp = -1;

    for ( int i=0; i < control_points.size(); ++i )
    {
        const Vec& cp = control_points[i];
        if ( ( cp - cursor_loc ).len() < 2.0 )
        {
            selected_cp = i;
            glutSetCursor( GLUT_CURSOR_UP_DOWN );
        }

        glPushMatrix();
        glTranslatef(cp.x, cp.y, cp.z);
        if ( selected_cp == i )
            glColor3ub( 0, 255, 255 );
        else
            glColor3ub( 255, 255, 0 );
        glutSolidSphere(1.0,12,12);
        glPopMatrix();

        glBegin( GL_LINES );
        glVertex3f( cp.x, 0, cp.z );
        glVertex3f( cp.x, cp.y, cp.z );
        glEnd();
    }

    if ( selected_cp < 0 && old_sel != selected_cp )
        glutSetCursor( GLUT_CURSOR_CROSSHAIR );

    // Find out the world coordinates of mouse pointer
    // to locate the cursor
    if ( mouseState[0] == mouseState[1] == mouseState[2] == 0 )
    {
        GLdouble model[16], proj[16];
        GLint view[4];
        GLfloat z;
        GLdouble ox, oy, oz;

        glGetDoublev(GL_MODELVIEW_MATRIX, model);
        glGetDoublev(GL_PROJECTION_MATRIX, proj);
        glGetIntegerv(GL_VIEWPORT, view);

        glReadPixels(mouseX, view[3]-mouseY-1, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
        gluUnProject(mouseX, view[3]-mouseY-1, z, model, proj, view, &ox, &oy, &oz);

        cursor_loc = Vec(ox, oy, oz);

        // Draw the cursor
        glPushMatrix();
        glDisable(GL_LIGHTING) ;
        glTranslatef(ox, oy, oz);
        glColor3ub( 255, 0, 255 );
        glutSolidSphere(0.5,12,12);
        glPopMatrix();
    }

    static char tmp_str[255];
    glDisable( GL_DEPTH_TEST );
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glColor3ub( 255, 255, 0 );
    glRasterPos2f (-0.95, -0.95);
    sprintf( tmp_str, "control points: %d, reqularization: %2.3f, bending energy: %4.3f",
             control_points.size(), regularization, bending_energy );
    draw_string( tmp_str );
    glEnable( GL_DEPTH_TEST );

    glFlush();
    glutSwapBuffers();
    screen_dirty = false;
#endif
}


void GLApp::Callbacks::reshape( int w, int h )
{
    winW = w;
    winH = h;
    glViewport( 0, 0, winW, winH );
    glEnable( GL_DEPTH_TEST );
    glDisable( GL_CULL_FACE );
    glEnable( GL_NORMALIZE );
    glDepthFunc( GL_LESS );
}


void GLApp::Callbacks::idlefunc()
{
    glCheckErrors();
    if (screen_dirty)
        glutPostRedisplay();
}



void GLApp::Callbacks::keyboard( unsigned char key, int, int )
{
#if 0
    switch (key)
    {
    case 'a':
        control_points.push_back( cursor_loc );
        calc_tps();
        break;
    case 'd':
        if ( selected_cp >= 0 )
        {
            control_points.erase( control_points.begin() + selected_cp );
            selected_cp = -1;
            calc_tps();
        }
        break;
        case 'c':
        control_points.clear();
        clear_grid();
        break;
        case '+':
        regularization += 0.025;
        calc_tps();
        break;
        case '-':
        regularization -= 0.025;
        if (regularization < 0) regularization = 0;
        calc_tps();
        break;
        case '/': camZoom -= 1; break;
        case '*': camZoom += 1; break;
        case 'q': exit( 0 ); break;
        }
    screen_dirty=true;
#endif
}


void GLApp::Callbacks::mouse( int button, int state, int, int )
{
#if 0
    mouseState[ button ] = (state==GLUT_DOWN);
    modifiers = glutGetModifiers();

    glutSetCursor( GLUT_CURSOR_CROSSHAIR );

    if ( button == 1 && state==GLUT_DOWN )
        glutSetCursor( GLUT_CURSOR_CYCLE );

    if ( button == 0 )
    {
        if ( state==GLUT_UP )
        {
            calc_tps();
            screen_dirty=true;
        }
        else if ( state==GLUT_DOWN && selected_cp<0 )
            keyboard( 'a', 0,0 );
    }
#endif
}



void GLApp::Callbacks::mouseMotion( int x, int y )
{
#if 0
    if ( mouseState[0] && mouseX != -999 )
        if ( selected_cp >= 0 )
            control_points[selected_cp].y += -(y - mouseY)/3;

    if ( mouseState[1] && mouseX != -999 )
    {
        camAlpha += -(y - mouseY);
        camBeta += (x - mouseX);

        screen_dirty=true;
    }

    if ( mouseX != x || mouseY != y )
    {
        mouseX = x;
        mouseY = y;
        screen_dirty=true;
    }
#endif
}



void StencilVisualizer::keyboard_special( int key, int, int )
{
    switch (key)
    {
    case GLUT_KEY_UP: camAlpha += 5 ; break;
    case GLUT_KEY_DOWN: camAlpha += -5 ; break;
    case GLUT_KEY_RIGHT: camBeta += -5 ; break;
    case GLUT_KEY_LEFT: camBeta += 5 ; break;
    }
    screen_dirty=true;
}



void StencilVisualizer::menu_select(int mode)
{
    StencilVisualizer::keyboard( (unsigned char)mode, 0,0 );
}


void StencilVisualizer::draw_string (const char* str)
{
    for (unsigned i=0; i<strlen(str); i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str[i]);
}



void StencilVisualizer::create_menu(void)
{
    glutCreateMenu(StencilVisualizer::menu_select);
    glutAddMenuEntry(" d     Delete control point",'d');
    glutAddMenuEntry(" a     Add control point",'a');
    glutAddMenuEntry(" c     Clear all",'c');
    glutAddMenuEntry(" +     Relax more",'+');
    glutAddMenuEntry(" -     Rela less",'-');
    glutAddMenuEntry(" /     Zoom in",'/');
    glutAddMenuEntry(" *     Zoom out",'*');
    glutAddMenuEntry(" q     Exit",'q');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


void StencilVisualizer::glCheckErrors()
{
    GLenum errCode = glGetError();
    if ( errCode != GL_NO_ERROR )
    {
        const GLubyte *errString = gluErrorString( errCode );
        fprintf(stderr, "OpenGL error: %s\n", errString);
    }
}




void StencilVisualizer::initializeWindow(int argc, char** argv) {
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutInitWindowSize( winW, winH );
    glutInitWindowPosition( 0, 0 );
    if ( !glutCreateWindow( "Stencil Visualizer ALPHA" ) )
    {
        printf( "Couldn't open window.\n" );
        exit(EXIT_FAILURE);
    }

    glutSetCursor( GLUT_CURSOR_CROSSHAIR );
}

void StencilVisualizer::setupCallbacks() {

    glutDisplayFunc( StencilVisualizer::display );
    glutIdleFunc( StencilVisualizer::idlefunc );
    glutMouseFunc( StencilVisualizer::mouse );
    glutMotionFunc( StencilVisualizer::mouseMotion );
    glutPassiveMotionFunc( StencilVisualizer::mouseMotion );
    glutKeyboardFunc( StencilVisualizer::keyboard );
    glutSpecialFunc( StencilVisualizer::keyboard_special );
    glutReshapeFunc( StencilVisualizer::reshape );
}

void StencilVisualizer::setupLights() {
    glEnable(GL_LIGHTING);
    GLfloat lightAmbient[] = {0.5f, 0.5f, 0.5f, 1.0f} ;
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lightAmbient);

    glEnable(GL_LIGHT0);
    GLfloat light0Position[] = {0.7f, 0.5f, 0.9f, 0.0f} ;
    GLfloat light0Ambient[]  = {0.0f, 0.5f, 1.0f, 0.8f} ;
    GLfloat light0Diffuse[]  = {0.0f, 0.5f, 1.0f, 0.8f} ;
    GLfloat light0Specular[] = {0.0f, 0.5f, 1.0f, 0.8f} ;
    glLightfv(GL_LIGHT0, GL_POSITION,light0Position) ;
    glLightfv(GL_LIGHT0, GL_AMBIENT,light0Ambient) ;
    glLightfv(GL_LIGHT0, GL_DIFFUSE,light0Diffuse) ;
    glLightfv(GL_LIGHT0, GL_SPECULAR,light0Specular) ;

    glEnable(GL_LIGHT1);
    GLfloat light1Position[] = {0.5f, 0.7f, 0.2f, 0.0f} ;
    GLfloat light1Ambient[]  = {1.0f, 0.5f, 0.0f, 0.8f} ;
    GLfloat light1Diffuse[]  = {1.0f, 0.5f, 0.0f, 0.8f} ;
    GLfloat light1Specular[] = {1.0f, 0.5f, 0.0f, 0.8f} ;
    glLightfv(GL_LIGHT1, GL_POSITION,light1Position) ;
    glLightfv(GL_LIGHT1, GL_AMBIENT,light1Ambient) ;
    glLightfv(GL_LIGHT1, GL_DIFFUSE,light1Diffuse) ;
    glLightfv(GL_LIGHT1, GL_SPECULAR,light1Specular) ;
}
#endif
