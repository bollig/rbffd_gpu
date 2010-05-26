
#ifndef _OCTREE_H_
#define _OCTREE_H_

#include <stdio.h>
#include <vector>
#include "Vec3.h"

class ParametricPatch;
class Node;
//class Vec3;

// Encode the boundary of the 3D domain in an octree
// Assume the function

// Keep subdividing the octree for n levels
// In each leaf, store a single boundary point

// I could put all these parsing classes into their own namespace? 
//----------------------------------------------------------------------
// Base class
class BaseCheck
{
public:
	static int nb_hits;
	Vec3 vec1;
	int nb_prints;

	BaseCheck() {
		nb_hits = 0;
		nb_prints = 0;
	}
	virtual void operator()(Node* node) = 0;
};
//----------------------------------------------------------------------
class TreeParseCheck : public BaseCheck
{
public:
	void operator()(Node* node);
};
//----------------------------------------------------------------------
class TreeParseProject : public BaseCheck
{
public:
	void operator()(Node* node);
};
//----------------------------------------------------------------------
class TreeParsePrintBoundaryPts : public BaseCheck
{
public:
	void operator()(Node* node);
};
//----------------------------------------------------------------------
class TreeParseCreateBoundaryPts : public BaseCheck
{
public:
	void operator()(Node* node);
};
//----------------------------------------------------------------------
class Node
{
// design could be better
public:
	Node* next; // 8 nodes
	Vec3* pt; // boundary point
	double x[2];
	double y[2];
	double z[2];
	int hit;
	static int nb_hits;
	ParametricPatch* patch; // should not be necessary
	static int max_level;
	static int nb_nodes;
	int level; // the root is at level 0
	std::vector<Vec3> boundary_pts;

public:
	Node();
	~Node();
	void subdivide();
	void intersectBoundary();
	void setPatch(ParametricPatch* patch) {
		this->patch = patch;
	}
	void print(const char* msg=0);
	void printTree(const char* msg=0);
	void setLevel(int lev) { level = lev; }
	Vec3 getBoundaryPt();
	void parseTree(BaseCheck& check);
	void assignNode(Vec3& pt);
};
//----------------------------------------------------------------------
class Octree
{
private:
	// use the function how_far (neg if inside volume)
	ParametricPatch& patch;
	Node* root;
	double xmin, xmax;
	double ymin, ymax;
	double zmin, zmax;

public:
	Octree(ParametricPatch* patch);
	~Octree();
	void create();
	void intersectBoundary(Node& node);
	Node* getRoot() { return root; }
	void parseTree(BaseCheck& check);

	// set right after creating the octree
	void setDomain(double xmin, double xmax, double ymin, double ymax, 
  		double zmin, double zmax);

	// find the node that contains pt, and store its coordinates in a list
	// only fill the leaves (level == max_level)
	void assignNode(Vec3& pt);
};
//----------------------------------------------------------------------

#endif
