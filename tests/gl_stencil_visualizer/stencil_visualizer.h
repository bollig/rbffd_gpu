#ifndef __STENCIL_VISUALIZER_CALLBACKS_H__
#define __STENCIL_VISUALIZER_CALLBACKS_H__

#include <vector>
#include <QtOpenGL/qgl.h>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include "trackball.h"

#include "Vec3.h"
#include "rbf.h"

#include "visual_stencil.h"

/**
  * This class creates a glut window, loads a collection of stencils and weights as a
  * vector of VisualStencils, and then renders these stencils using OpenGL.
  *
  * We have this as a separate class and not inside the main because we intend to provide
  * this as part of the RBF library and we'd like to be able to launch multiple of instances
  * to view a variety of test cases.
  */

typedef std::vector<std::vector< Vec3 > > NodeListType;
typedef std::vector<std::vector< double > > WeightListType;
typedef std::vector<std::vector< RBF > > BasesListType;



class StencilVisualizer : public QGLWidget
{

public:
    StencilVisualizer( QWidget *parent=0, QGLWidget *shareWidget=0);

/**
  * Routines related to custom behavior
  */
protected:
    void wireCube(float);
    void drawAxes(void);

private:
    int shoulder, elbow;


/**
  * The following set of member functions are required for a QGLWidget
  * to run. They are equivalent to the callbacks we would setup for
  * GLUT.
  */
protected:
    virtual void initializeGL(void);        // Init window
    virtual void resizeGL( int w, int h );  // Resize window
    virtual void paintGL();                 // DisplayFunc
    virtual void keyPressEvent( QKeyEvent *e);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent * event);
    QSize sizeHint() const;
    QSize minimumSizeHint() const;
    void setClearColor(const QColor &color);


 /*****  END REQUIRED ******/

private:
    // From boxes demo in Qt4
    QPointF pixelPosToViewPos(const QPointF& p);

/** Begin ATTRIBUTES **/
private:
    TrackBall trackball;    // Control rotation
    int m_distExp;          // Control zoom/scale
    QTimer *m_timer;        // Timer to control window refresh
    QColor clearColor;
};

#if 0
class StencilVisualizer
{
public:
    StencilVisualizer(int argc, char** argv);
    ~StencilVisualizer();

    void setNodes(NodeListType *stencil_nodes);
    void setWeights(WeightListType *stencil_weights);
    void setBases(BasesListType *stencil_bases);

    void setupCallbacks();
    void setupVisualStencils();

    void run();

protected:
    void setupLights();
    void initializeWindow(int argc, char** argv);


    void create_menu();


    void glCheckErrors();
    void draw_string(const char* str);
    static void menu_select(int mode);

    //TODO: routine inside this class to load visual stencil data from file
    // void loadVisualStencils(char* node_filename, char* weight_filename, char* bases_filename);

private:

    static int winW, winH;
    int mouseX, mouseY;
    static bool mouseState[3];
    static int modifiers;
    Vec3 cursor_loc;
    float camAlpha, camBeta, camZoom;
    static bool screen_dirty;

};

/**
  * A set of Callbacks used by the Visualizer application. These need to be global functions
  * so we use a namespace to keep them organized and avoid making them static
  */
namespace GLApp {
    namespace Callbacks {
        void display();
        void idlefunc();
        void reshape(int w, int h);
        void mouse(int button, int state, int, int);
        void mouseMotion(int x, int y);
        void keyboard(unsigned char key, int, int);
        void keyboard_special(int key, int, int);
    }
}
#endif

#endif //__STENCIL_VISUALIZER_CALLBACKS_H__
