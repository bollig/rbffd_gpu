#include <stdlib.h>

#include "octree.h"
#include "parametric_patch.h"

#define I(i,j,k) ((i)+2*((j)+2*(k)))

int Node::max_level = 3;
int Node::nb_hits = 0;
int BaseCheck::nb_hits = 0;
int Node::nb_nodes = 0;

//----------------------------------------------------------------------
Octree::Octree(ParametricPatch* patch_, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
: patch(*patch_)
{
	root = new Node();
	root->setPatch(&patch);
	// Set, manually for now, the limits of the domain

	setDomain(xmin, xmax, ymin, ymax, zmin, zmax);
}
//----------------------------------------------------------------------
Octree::~Octree()
{
}
//----------------------------------------------------------------------
void Octree::setDomain(double xmin, double xmax, double ymin, double ymax, 
  double zmin, double zmax)
{
	this->xmin = xmin;
	this->xmax = xmax;
	this->ymin = ymin;
	this->ymax = ymax;
	this->zmin = zmin;
	this->zmax = zmax;

	root->x[0] = xmin;
	root->x[1] = xmax;
	root->y[0] = ymin;
	root->y[1] = ymax;
	root->z[0] = zmin;
	root->z[1] = zmax;
}
//----------------------------------------------------------------------
void Octree::create()
{
	root->setLevel(0);
	root->subdivide();
}
//----------------------------------------------------------------------
#if 0
void Octree::intersectBoundary(Node& node)
{
	// does node intersection the boundary
	// check whether 8 corners are inside or outside the domain

	double p1 = patch.how_far(node.x[0], node.y[0], node.z[0]);
	double p;

	node.hit = 0; // node does not intersect the boundary
	// inefficient!!! Should unroll if posisble
	for (int k=0; k < 2; k++) {
	for (int j=0; j < 2; j++) {
	for (int i=0; i < 2; i++) {
		if ((i+j+k) == 0) continue;
		p = patch.how_far(node.x[i], node.y[j], node.z[k]);
		printf("-- intersectBoundary, how_far = %f", p);
		if (p*p1 < 0) goto inters;  // p*p1 < 0 implies overlap with boundary
	}}}

inters:

	if (p*p1 < 0) {   // node intersects boundary
		node.hit = 1; 
	}

	exit(0);
}
#endif 
//----------------------------------------------------------------------
void Octree::parseTree(BaseCheck& check)
{
	root->parseTree(check);
}
//----------------------------------------------------------------------
void Octree::assignNode(Vec3& pt)
{
	root->assignNode(pt);
}
//----------------------------------------------------------------------



#if 1
void Node::intersectBoundary()
{
	// does node intersection the boundary
	// check whether 8 corners are inside or outside the domain

	// PROBLEM: where to store patch? In each node? 

	double p1 = patch->how_far(x[0], y[0], z[0]);
	//printf("-- how_far(p1) = %f, (%f, %f, %f)\n", p1, x[0], y[0], z[0]);
	double p;

	hit = 0; // node does not intersect the boundary

	for (int k=0; k < 2; k++) {
	for (int j=0; j < 2; j++) {
	for (int i=0; i < 2; i++) {
		if ((i+j+k) == 0) continue;
		p = patch->how_far(x[i], y[j], z[k]);
		//printf("-- how_far = %f\n", p);
		if (p*p1 < 0) goto inters;   // break out of multiple loops
	}}}

inters:

	if (p*p1 < 0) {    // node intersects boundary
		hit = 1; // node intersects the boundary
	}
	printf("intersect, hit= %d\n", hit);
	//printf("=======================\n");
}
#endif
//----------------------------------------------------------------------
Node::Node()
{
}
//----------------------------------------------------------------------
Node::~Node()
{
}
//----------------------------------------------------------------------
void Node::subdivide()
{
	next = new Node [8];
	double xm = 0.5*(x[0]+x[1]);
	double ym = 0.5*(y[0]+y[1]);
	double zm = 0.5*(z[0]+z[1]);

// Need the brackets to isolate the references, which otherwise cannot
// be changed
	{
	Node& n = next[I(0,0,0)];
	n.x[0] = x[0];
	n.x[1] = xm;
	n.y[0] = y[0];
	n.y[1] = ym;
	n.z[0] = z[0];
	n.z[1] = zm;
	}
	{
	Node& n = next[I(1,0,0)];
	n.x[0] = xm;
	n.x[1] = x[1];
	n.y[0] = y[0];
	n.y[1] = ym;
	n.z[0] = z[0];
	n.z[1] = zm;
	}
	{
	Node& n = next[I(0,1,0)];
	n.x[0] = x[0];
	n.x[1] = xm;
	n.y[0] = ym;
	n.y[1] = y[1];
	n.z[0] = z[0];
	n.z[1] = zm;
	}
	{
	Node& n = next[I(1,1,0)];
	n.x[0] = xm;
	n.x[1] = x[1];
	n.y[0] = ym;
	n.y[1] = y[1];
	n.z[0] = z[0];
	n.z[1] = zm;
	}
	{
	Node& n = next[I(0,0,1)];
	n.x[0] = x[0];
	n.x[1] = xm;
	n.y[0] = y[0];
	n.y[1] = ym;
	n.z[0] = zm;
	n.z[1] = z[1];
	}
	{
	Node& n = next[I(0,1,1)];
	n.x[0] = x[0];
	n.x[1] = xm;
	n.y[0] = ym;
	n.y[1] = y[1];
	n.z[0] = zm;
	n.z[1] = z[1];
	}
	{
	Node& n = next[I(1,0,1)];
	n.x[0] = xm;
	n.x[1] = x[1];
	n.y[0] = y[0];
	n.y[1] = ym;
	n.z[0] = zm;
	n.z[1] = z[1];
	}
	{
	Node& n = next[I(1,1,1)];
	n.x[0] = xm;
	n.x[1] = x[1];
	n.y[0] = ym;
	n.y[1] = y[1];
	n.z[0] = zm;
	n.z[1] = z[1];
	}

	//printf("*** subdivide loop\n");
	for (int i=0; i < 8; i++) {
		next[i].setPatch(patch);
		next[i].intersectBoundary();
		next[i].next = 0;
		next[i].level = level+1;
		//printf("subdivid: level= %d\n", level);
		//printf("node %d, level %d\n", i, next[i].level);
		Node& n = next[i];
		//printf("node bounds: x: %f, %f\n", n.x[0], n.x[1]);
		//printf("           : y: %f, %f\n", n.y[0], n.y[1]);
		//printf("           : z: %f, %f\n", n.z[0], n.z[1]);

		if (next[i].level < Node::max_level 
		   && next[i].hit == 1) {
			next[i].subdivide();
		}
	}
}
//----------------------------------------------------------------------
void Node::print(const char* msg)
{
	if (msg != 0) {
		printf("%s\n", msg);
	}
	printf("x= %f ,%f\n", x[0], x[1]);
	printf("y= %f ,%f\n", y[0], y[1]);
	printf("z= %f ,%f\n", z[0], z[1]);
	printf("hit= %d, level= %d\n", hit, level);
}
//----------------------------------------------------------------------
void Node::printTree(const char* msg)
{
	if (msg != 0) {
		printf("%s\n", msg);
	}

	print();

	if (next == 0) {
		printf("no additional nodes\n");
		return;
	}

	for (int i=0; i < 8; i++) {
		printf("...\n");
		next[i].printTree();
	}
}
//----------------------------------------------------------------------
Vec3 Node::getBoundaryPt()
{
	// We know the node intersects the boundary
	// Find the boundary point by successive subdivision or other method
	// ...
	// pt->setValue(x, y, z);   // boundary point

	// center of the node
	double xc = (x[0] + x[1])*0.5;
	double yc = (y[0] + y[1])*0.5;
	double zc = (z[0] + z[1])*0.5;
	Vec3 grad = patch->gradient(xc,yc,zc);
	Vec3 pt(xc,yc,zc);

	printf("how_far: %g\n", patch->how_far(pt));
}
//----------------------------------------------------------------------
void Node::parseTree(BaseCheck& check)
// Should be able to combine the above routines
{
	check(this);
	if (!next) return;

	for (int i=0; i < 8; i++) {
		next[i].parseTree(check);
	}
}
//----------------------------------------------------------------------
void Node::assignNode(Vec3& pt)
{
	// check whether pt is inside node
	if (level == Node::max_level) {
		if (pt.x() >= x[0] && pt.x() <= x[1] &&
			pt.y() >= y[0] && pt.y() <= y[1] &&
			pt.z() >= z[0] && pt.z() <= z[1]) {

			//printf("... bnd point assigned ...\n");
			//pt.print("       bnd pt");
			//printf("node bounds: x: %f, %f\n", x[0], x[1]);
			//printf("           : y: %f, %f\n", y[0], y[1]);
			//printf("           : z: %f, %f\n", z[0], z[1]);
			boundary_pts.push_back(pt);
			nb_hits++;
			//printf("nb_hits= %d\n", nb_hits);
			return;
		}
	}

	if (!next) return;

	for (int i=0; i < 8; i++) {
		next[i].assignNode(pt);
	}
}
//----------------------------------------------------------------------





void TreeParseCheck::operator()(Node* node)
{
	if (node->level == Node::max_level) {
		nb_hits += 1;
	}
	Node::nb_nodes += 1;
}
//----------------------------------------------------------------------
void TreeParsePrintBoundaryPts::operator()(Node* node)
{
	if (node->level != Node::max_level) return;
	//if (node->hit != 1) return;

	Node& n = *node;
	//printf("node bounds: x: %f, %f\n", n.x[0], n.x[1]);
	//printf("           : y: %f, %f\n", n.y[0], n.y[1]);
	//printf("           : z: %f, %f\n", n.z[0], n.z[1]);
	//printf("    hit: %d\n", node->hit);


	nb_hits++;
	Node::nb_nodes++;

	printf("------- Boundary points in node ------\n");
	for (int i=0; i < node->boundary_pts.size(); i++) {
		node->boundary_pts[i].print("bndry pt");
		nb_prints++;
	}
}
//----------------------------------------------------------------------
void TreeParseCreateBoundaryPts::operator()(Node* node)
{
	Node::nb_nodes += 1;

	if (node->level != Node::max_level) return;
	if (node->hit == 0) return;
	if (node->boundary_pts.size() != 0) return;


	printf("++++++++++++++++++++++++++\n");
	Node& n = *node;
	double xc = 0.5*(n.x[0] + n.x[1]);
	double yc = 0.5*(n.y[0] + n.y[1]);
	double zc = 0.5*(n.z[0] + n.z[1]);
	Vec3 pt(xc,yc,zc);

	//printf("Intersect Boundary, node bounds: x: %f, %f\n", n.x[0], n.x[1]);
	//printf("                               : y: %f, %f\n", n.y[0], n.y[1]);
	//printf("                               : z: %f, %f\n", n.z[0], n.z[1]);

	Vec3 grad = n.patch->gradient(xc,yc,zc);
	//grad.print("grad");

	Vec3 bndpt = n.patch->projectToBoundary(pt, grad);
	//Vec3 bndpt = n.patch->project(pt);

	// Given the bndpt, move along the boundary until we end up with 
	// the projection

	//printf("old pt: how_far: %g\n", n.patch->how_far(pt));
	//printf("new pt: how_far: %g\n", n.patch->how_far(bndpt));

	//pt.print("old bndry point");
	//bndpt.print("new bndry point");

	Vec3 diff = bndpt-pt;
	Vec3 gradbnd = n.patch->gradient(diff.x(), diff.y(), diff.z());
	double cross_prod = gradbnd.cross(diff).magnitude();
	printf("cross_prod = %g\n", cross_prod);

	cross_prod = grad.cross(diff).magnitude();
	printf("cross_prod_orig = %g\n", cross_prod);


	// single boundary point per node
	node->boundary_pts.push_back(bndpt);
}
//----------------------------------------------------------------------
void TreeParseProject::operator()(Node* node)
{
// determine the boundary point associated with each node

	Node::nb_nodes += 1;

	if (node->level != Node::max_level) return;
	//if (node->level != 2) return;
	if (node->hit != 1) return;

// get boundary point by projection
// check that the boundary point found lies within the node (double checking)
// Store a single boundary point per node

	Node& n = *node;

	double xc = (n.x[0] + n.x[1])*0.5;
	double yc = (n.y[0] + n.y[1])*0.5;
	double zc = (n.z[0] + n.z[1])*0.5;
	
	// all sizes should be the same
	Vec3 sz(n.x[1]-n.x[0], n.y[1]-n.y[0], n.z[1]-n.z[0]);

	Vec3 center(xc,yc,zc);
	printf("--------------------------------\n");
	center.print("center");
	sz.print("cell size");
	printf("how_far: %f\n", n.patch->how_far(center));

	#if 0
	for (int k=0; k < 2; k++) {
	for (int j=0; j < 2; j++) {
	for (int i=0; i < 2; i++) {
		double p = n.patch->how_far(n.x[i], n.y[j], n.z[k]);
		//printf("how_far: %f\n", p);
	}}}
	#endif

	Vec3& bnd_pt = vec1;
	bnd_pt = n.patch->project(center);
	bnd_pt.print("bnd");

	nb_hits++;
	printf("nb_hits: %d\n", nb_hits);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
