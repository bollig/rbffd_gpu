#include <math.h>
#include <stdlib.h>
#include <vector>
#include "utils/geom/ellipsoid_patch.h"
#include "utils/geom/parametric_patch.h"
#include "utils/geom/octree.h"

using namespace std;

// GLOBAL VARIABLES

//----------------------------------------------------------------------
int main (int argc, char** argv)
{
	// domain dimensions
	double pi = acos(-1.);
	double a = 2.;
	double b = 3.;
	double c = 4.;

	int n1 = 80;
	int n2 = 80;
	ParametricPatch* ep = new EllipsoidPatch(0., pi, 0., 2.*pi, n1, n2, a, b, c);

	Octree oct(ep);
	oct.setDomain(-a-.5, a+.5, -b-.5, b+.5, -c-.5, c+.5);

	Node* root = oct.getRoot();
	root->intersectBoundary();

	Node::max_level = 5;
	oct.create();
	//root->printTree();

	// go through all the nodes, and print out associated boundary point and approx. distance from boundary
	BaseCheck* check = new TreeParseCheck;
	oct.parseTree(*check);  // argument: function to execute on each node

	BaseCheck* bnd_check = new TreeParseProject;
    // TODO: Figure out if this is correct. Was first case here
#if 0
    oct.parseTree(*check);
#else 
    oct.parseTree(*bnd_check);
#endif 

	// create lists of boundary points in each leaf node
	vector<Vec3> boundary_pts = ep->getBoundaryPoints();
	printf("nb boundary points (main): %d\n", (int)boundary_pts.size());
	Node::nb_hits = 0; // how many bnd pts get assigned to nodes

	#if 0
	for (int i=0; i < boundary_pts.size(); i++) {
		//printf("============== i = %d ======================\n", i);
		//boundary_pts[i].print("assign bndry pt to node");
		oct.assignNode(boundary_pts[i]);
	}
	printf("Node::nb_hits= %d\n", Node::nb_hits);
	#endif

	check = new TreeParseCreateBoundaryPts;
	Node::nb_nodes = 0;
	oct.parseTree(*check);

	exit(0);

	// print out the boundary points associated with each node
	check = new TreeParsePrintBoundaryPts;
	check->nb_hits = 0;
	Node::nb_nodes = 0;
	check->nb_prints = 0;
	oct.parseTree(*check);

	//printf("parse boundary points: nb of leaf bndry intersections : %d\n", check_nb_hits);
	printf("total nb leaf nodes at max_level: %d\n", Node::nb_nodes);
	printf("number of nodes with hit=1: %d\n", TreeParseCheck::nb_hits);
	//printf("number of nodes with hit=1: %d\n", check->nb_hits);
	printf("nb printouts: %d\n", check->nb_prints);
	exit(0);

	if (argc > 1) {
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
