#include <stdlib.h>
#include "set"
#include "original_grid.h"

using namespace std;

OriginalGrid::OriginalGrid(ProjectSettings* settings) {
    // BEGIN JUNK: everything up to END JUNK should
    // be considered for removal to put into a subclass
    // for 2D regular grid.

    // grid size
    nb_x = 1;
    nb_y = 1;
    nb_z = 1;

    laplacian.resize(nb_x * nb_y * nb_z);

    // subgrid size
    nx = nb_x / 2;
    ny = nb_y / 2;
    nz = 1;
    subgrid_index_list = new ArrayT<vector<int> >(nx, ny);

    // END JUNK

    // BEGIN NECESSARY:

    this->stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE");
    this->dim = settings->GetSettingAs<int>("DIMENSION");
    this->forceSymmetricStencils = settings->GetSettingAs<int>("FORCE_SYMMETRIC_STENCILS");

    int nb_inner_bound = settings->GetSettingAs<int>("NB_INNER_BND");
    int nb_outer_bound = settings->GetSettingAs<int>("NB_OUTER_BND");
    this->nb_bnd = nb_inner_bound + nb_outer_bound;
	this->nb_rbf = settings->GetSettingAs<int>("NB_INTERIOR") + this->nb_bnd;

    cout << "FORCE SYMMETRIC " << forceSymmetricStencils << "\t" << settings->GetSettingAs<bool>("FORCE_SYMMETRIC_STENCILS") <<  endl;
    cout << "NB_BND = " << nb_bnd << endl;
    cout << "NB_RBF = " << nb_rbf << endl;
}

// Default constructor
//  Specify the size of the stencil we wish to generate
//  and handle
OriginalGrid::OriginalGrid(int dim_num, int stencil_size) {

    // BEGIN JUNK: everything up to END JUNK should
    // be considered for removal to put into a subclass
    // for 2D regular grid.

    // grid size
    nb_x = 1;
    nb_y = 1;
    nb_z = 1;

    laplacian.resize(nb_x * nb_y * nb_z);

    // subgrid size
    nx = nb_x / 2;
    ny = nb_y / 2;
    nz = 1;
    subgrid_index_list = new ArrayT<vector<int> >(nx, ny);

    // END JUNK

    // BEGIN NECESSARY:

    this->stencil_size = stencil_size;
    this->dim = dim_num;
    this->forceSymmetricStencils = false;

    // END NECESSARY
}

//----------------------------------------------------------------------
// regular grid constructor, allows us to specify the number of nodes to
// generate in 2D.
// This should be in a subclass of grid.(TODO)
OriginalGrid::OriginalGrid(int n_x, int n_y, int stencil_size) {
    // grid size
    nb_x = n_x;
    nb_y = n_y;
    nb_z = 1;

    laplacian.resize(nb_x * nb_y * nb_z);

    // subgrid size
    nx = nb_x / 2;
    ny = nb_y / 2;
    nz = 1;
    subgrid_index_list = new ArrayT<vector<int> >(nx, ny);

    this->stencil_size = stencil_size;
    this->forceSymmetricStencils = false;
}
//----------------------------------------------------------------------

OriginalGrid::~OriginalGrid() {
}
//----------------------------------------------------------------------
#if 1

class ltvec {
public:
    static Vec3 xi;
    static vector<Vec3>* rbf_centers;

    static void setXi(Vec3& xi) {
        ltvec::xi = xi;
    }

    static void setRbfCenters(vector<Vec3>& rbf_centers_) {
        rbf_centers = &rbf_centers_;
    }

    bool operator()(const int i, const int j) {
        double d1 = ((*rbf_centers)[i] - xi).square();
        double d2 = ((*rbf_centers)[j] - xi).square();
        // allows duplicates
        return d1 <= d2;
    }
};

Vec3 ltvec::xi;
vector<Vec3>* ltvec::rbf_centers;

#endif

//----------------------------------------------------------------------

void OriginalGrid::avgStencilRadius() {
    // assume that the nodes are the seeds of a non-uniform CVT
    // compute the average radius of each stencil, without changing the actual stencil size.
    // Thus, a stencil might have 15 points, but the average distance is computed only based
    // on the inner 5,6, or 7 nodes.

    //vector<double> avg_distance(nb_rbf);
    avg_distance.resize(nb_rbf);
    vector<double> avg_bnd(nb_bnd);
    vector<double> avg_int(nb_rbf - nb_bnd);
    vector<int> vor_poly_sz(nb_rbf);

    double min_dist2;
    double dist2;
    //printf("nb_bnd= %d\n", nb_bnd);

    for (int i = 0; i < 30; i++) {
        // distance between 2 successive boundary points
        min_dist2 = (rbf_centers[i + 1] - rbf_centers[i]).square();
        //printf("=====\n");
        //rbf_centers[i].print("rbf center");
        //printf("bnd pt %d, dist = %f\n", sqrt(min_dist2));
    }

    for (int i = 0; i < nb_rbf; i++) {
        //for (int i=0; i < 30; i++) {   // 1/2 boundary points (TEMP)
        //printf("---------------\n");
        Vec3& v = rbf_centers[i];
        vector<int>& st = stencil[i];

        // center points is the first point
        // points are ordered with increasing distance from the center

        avg_distance[i] = 0.;
        int icenter = st[0];
        //printf("(%d) st[0]= %d, i= %d\n", i, st[0], i);
        int first = st[1];

        //rbf_centers[icenter].print("rbf center");
        min_dist2 = (rbf_centers[icenter] - rbf_centers[first]).square();
        avg_distance[i] = sqrt(min_dist2);
        // stencil_size = max stencil_size
        vor_poly_sz[i] = 1;

        // Start at 1 because the 0th element is the center of stencil
        for (int k = 1; k < stencil_size; k++) {
            //printf("k= %d\n", k);
            int indx = st[k];
            dist2 = (rbf_centers[icenter] - rbf_centers[indx]).square();
            //printf("dist2= %f\n", dist2);
            if (dist2 / min_dist2 > (1.5 * 1.5)) {
                //printf("vorsize= %d, stencil size: %d\\n", vor_poly_sz[i], (int) stencil.size());
                //printf("   avg_dist= %f\n", avg_distance[i] / vor_poly_sz[i]);
                break;
            }
            avg_distance[i] += sqrt(dist2);
            vor_poly_sz[i]++;
        }

        avg_distance[i] /= vor_poly_sz[i];

        printf("%s:%d\tavg_dist[%d]= %f\n", __FILE__,__LINE__,i, avg_distance[i]);
        printf("    nb points in stencil (excluding center): %d\n", (int) vor_poly_sz[i]);
    }

#if 0
    double avgint = 0.;
    double avgbnd = 0.;
    for (int i = 0; i < avg_int.size(); i++) {
        avgint += avg_int[i];
    }
    for (int i = 0; i < avg_bnd.size(); i++) {
        avgbnd += avg_bnd[i];
    }
    avgint /= avg_int.size();
    avgbnd /= avg_bnd.size();

    printf("average interior distances: %f\n", avgint);
    printf("average boundary distances: %f\n", avgbnd);
#endif
}
//----------------------------------------------------------------------

void OriginalGrid::computeStencils(double *nodes, int st_size, int nb_boundary_nodes, int nb_tot_nodes) {
    this->stencil_size = st_size;
    this->rbf_centers.resize(nb_tot_nodes);
    for (int i = 0; i < nb_tot_nodes; i++) {
        Vec3 v;
        for (int j = 0; j < this->dim; j++) {
            v[j] = nodes[i*dim + j];
        }
        this->rbf_centers[i] = v;
    }
    this->nb_rbf = nb_tot_nodes;
    this->nb_bnd = nb_boundary_nodes;
    boundary.resize(nb_boundary_nodes);
    for (int i = 0; i < nb_boundary_nodes; i++) {
        boundary[i] = i;
    }

    this->computeStencils();
}

void OriginalGrid::computeStencils(double *nodes, int st_size, int nb_boundary_nodes, int nb_tot_nodes, KDTree* kdtree) {
    this->stencil_size = st_size;
    this->nb_bnd = nb_boundary_nodes;
    this->nb_rbf = nb_tot_nodes;
    this->computeStencils(nodes, kdtree);
}

void OriginalGrid::computeStencils(double* nodes, KDTree* kdtree) {
    this->rbf_centers.resize(nb_rbf);
    for (int i = 0; i < nb_rbf; i++) {
        Vec3 v;
        for (int j = 0; j < this->dim; j++) {
            v[j] = nodes[i*dim + j];
        }
        this->rbf_centers[i] = v;
    }
    boundary.resize(nb_bnd);
    for (int i = 0; i < nb_bnd; i++) {
        boundary[i] = i;
    }

    // This boolean is from the settings config file (Key: "FORCE_SYMMETRIC_STENCILS")
    if (this->forceSymmetricStencils) {
        this->computeSymmetricStencilsKDTree(kdtree);
    } else {
        this->computeStencilsKDTree(kdtree);
    }

}

//----------------------------------------------------------------------
void OriginalGrid::computeStencils() {

    if (nb_bnd == 0) {
        printf("**** WARNING! nb_bnd == 0; Did generateGrid() update this value properly?! ******\n");
        printf("**** Resetting nb_bnd and nb_rbf ****\n");
        this->nb_bnd = boundary.size();
        this->nb_rbf = rbf_centers.size();
    }

    if (stencil_size > nb_rbf) {
        int new_stencil_size = (int) (0.5 * nb_rbf);
        new_stencil_size = (new_stencil_size > 1) ? new_stencil_size : 1;
        printf("\n!!!!!!!!!!!!!!!!!!!\nWARNING! Not enough centers to reach specified stencil_size (size: %d for %d RBFs)! Using new stencil size: %d\n!!!!!!!!!!!!!!!!!!!\n\n", stencil_size, nb_rbf, new_stencil_size);
        stencil_size = new_stencil_size;
    }

    // for each node, a vector of stencil nodes (global indexing)
    stencil.resize(nb_rbf);
    //printf("stencil vec length: %d, nb_rbf= %d\n", stencil.size(), nb_rbf);

    ltvec ltvec_inst;
    //vector<double> avg_distance;
    vector<double> avg_bnd;
    vector<double> avg_int;

    this->avg_distance.resize(nb_rbf);
    avg_bnd.resize(nb_bnd);
    avg_int.resize(nb_rbf - nb_bnd);

    printf("nb_rbf= %d (bnd: %d)\n", nb_rbf, nb_bnd);
    //printf("nb_bnd= %d\n", nb_bnd);
    //exit(0);

    // O(n^2) algorithm, whose cost is independent of the number of nearest sought

    for (int i = 0; i < nb_rbf; i++) {
        Vec3& v = rbf_centers[i];
        vector<int>& st = stencil[i];
        set<int, ltvec> se;
        ltvec::setRbfCenters(rbf_centers);
        ltvec::setXi(v);

        // find nearest points to center
        for (int j = 0; j < nb_rbf; j++) {
            se.insert(j);
        }

        set<int, ltvec>::iterator sei = se.begin();
        set<int, ltvec>::iterator seii = se.begin();

        // minimimum distance:
        seii++; // I now access first point
        double min_dist = (rbf_centers[*seii] - rbf_centers[i]).square();
        min_dist = sqrt(min_dist);
        //printf("min distance: %f\n", min_dist);

#if 1
        this->avg_distance[i] = 0.;
        if (i < nb_bnd) {
            avg_bnd[i] = 0.;
        } else {
            avg_int[i - nb_bnd] = 0.;
        }
#endif

        // stencil_size = max stencil_size
        for (int k = 0; k < stencil_size; k++) {
            double d = (rbf_centers[*sei] - rbf_centers[i]).square();
            double ss = sqrt(d);

            // I am not sure code works yet
            // Errors of solution to heat equation do not stay small
            //if ((ss / min_dist) > 1.5) break;

            st.push_back(*sei);

#if 1
            this->avg_distance[i] += ss;
            if (i < nb_bnd) {
                avg_bnd[i] += ss;
            } else {
                avg_int[i - nb_bnd] += ss;
            }
#endif

           // printf("(%d, %d) dist= %f\n", i, k, ss);

            // printf("el %d, d= %f\n", *sei, d);
            sei++;
        }

#if 1
        this->avg_distance[i] /= (st.size() - 1.); // ignore center point
        if (i < nb_bnd) {
            avg_bnd[i] /= (st.size() - 1.);
        } else {
            avg_int[i - nb_bnd] /= (st.size() - 1.);
        }
#endif

        printf("avg_dist[%d]= %f\n", i, avg_distance[i]);
        printf("nb points in stencil: %d\n", (int) st.size());
    }

#if 1
    double avgint = 0.;
    double avgbnd = 0.;

    printf("avg_int.size() = %d\n", (int)avg_int.size());
    printf("avg_bnd.size() = %d\n", (int)avg_bnd.size());

    if (nb_rbf - nb_bnd > 0) {
        for (int i = 0; i < avg_int.size(); i++) {
            avgint += avg_int[i];
        }
        avgint /= avg_int.size();
    } else {
        avgint = 0.;
    }

    // There should always be boundary point(s)
    for (int i = 0; i < avg_bnd.size(); i++) {
        avgbnd += avg_bnd[i];
    }
    avgbnd /= avg_bnd.size();

    printf("mean of mean interior distances: %f (size: %d)\n", avgint, (int)avg_int.size());
    printf("mean of mean boundary distances: %f (size: %d)\n", avgbnd, (int)avg_bnd.size());
#endif
}
//----------------------------------------------------------------------
// Compute stencils using a KDTree, but add the extra constraint that
// stencil centers must be in the stencil set of any nodes used for its
// stencil set.
//----------------------------------------------------------------------
void OriginalGrid::computeSymmetricStencilsKDTree(KDTree* kdtree) {

    if (nb_bnd == 0) {
        printf("**** WARNING! nb_bnd == 0; Did generateGrid() update this value properly?! ******\n");
        printf("**** Resetting nb_bnd and nb_rbf ****\n");
        this->nb_bnd = boundary.size();
        this->nb_rbf = rbf_centers.size();
    }

    if (stencil_size > nb_rbf) {
        int new_stencil_size = (int) (0.5 * nb_rbf);
        new_stencil_size = (new_stencil_size > 1) ? new_stencil_size : 1;
        printf("\n!!!!!!!!!!!!!!!!!!!\nWARNING! Not enough centers to reach specified stencil_size (size: %d for %d RBFs)! Using new stencil size: %d\n!!!!!!!!!!!!!!!!!!!\n\n", stencil_size, nb_rbf, new_stencil_size);
        stencil_size = new_stencil_size;
    }

    std::vector< std::vector<int> > unsym_stencil;
    vector< vector<int> > nearest_ids;
    vector< vector<double> > nearest_dists;

    // for each node, a vector of stencil nodes (global indexing)
    stencil.resize(nb_rbf);
    unsym_stencil.resize(nb_rbf);
    nearest_ids.resize(nb_rbf);
    nearest_dists.resize(nb_rbf);

    printf("stencil vec length: %d, nb_rbf= %d\n", (int)stencil.size(), nb_rbf);

    this->avg_distance.resize(nb_rbf);

    printf("nb_rbf= %d (bnd: %d)\n", nb_rbf, nb_bnd);
    int node_count =0;

    for (int i = 0; i < nb_rbf; i++) {
        vector<double> center(dim, 0);
        for (int j = 0; j < dim; j++) {
            center[j] = rbf_centers[i][j];
        }

        unsym_stencil[i].resize(stencil_size);

        kdtree->k_closest_points(center, stencil_size, nearest_ids[i], nearest_dists[i]);

       // printf("Stencil[%d] = {", i);
        for (int j=0; j < stencil_size; j++) {
            int rev_indx = (stencil_size-1)-j;
            unsym_stencil[i][j] = nearest_ids[i][rev_indx];
         //   printf(" %d(%f) ", stencil[i][j], nearest_dists[rev_indx]);
            node_count ++;
        }
    }
    cout << "Added " << node_count << " nodes" << endl;

    int trim_count = 0;
     // TODO: TRIM STENCIL EDGES WHICH ARE NOT SYMMETRIC
    for (int i = 0; i < unsym_stencil.size(); i++) {
        int adjustedStencilSize = unsym_stencil[i].size();
        int orig_stencil_size = unsym_stencil[i].size();
        // Search each stencil to make sure edges are symmetric
        // (Brute force! sould be more intelligent if this consumes a lot of time!)
        for (int j=0; j < orig_stencil_size; j++) {
            int indxA = unsym_stencil[i][0];
            int indxB = unsym_stencil[i][j];
            if (unsym_stencil[indxB][0] != indxB) {
                cout << "WARNING! STENCIL " << indxB << " is not in the correct location. Row contains stecil: " << unsym_stencil[indxB][0] << endl;
                exit(EXIT_FAILURE);
            }
            if ( !isIndxInStencil(indxA, unsym_stencil[indxB]) ) {
                unsym_stencil[i][j] = -1;
                trim_count ++;
                adjustedStencilSize --;
            }
        }

        // Copy trimmed stencils back into our public stencil list
        stencil[i].resize(adjustedStencilSize);
        int k = 0;
        for (int j = 0; j < orig_stencil_size; j++) {
            if (unsym_stencil[i][j] != -1) {
                stencil[i][k] = unsym_stencil[i][j];

                int rev_indx = (stencil_size-1)-j;
                this->avg_distance[i] += nearest_dists[i][rev_indx];

                k++;
            }
        }
        if (k != adjustedStencilSize) {
            cout << "DID NOT TRIM PROPERLY. Off by: " << k - adjustedStencilSize << endl;
            exit(EXIT_FAILURE);
        }
        this->avg_distance[i] /= (adjustedStencilSize-1);
    }
    cout << "Trimmed " << trim_count << " nodes" << endl;

    cout << "Percentage cut: " << ((double)trim_count / (double)node_count ) * 100. << "%" << endl;

    printf("DONE GENERATING STENCILS\n");
}
//----------------------------------------------------------------------

// An edge is symmetric if A exists in stencilB and B exists in stencilA
// Of course we form stencilA and then test each member to see if
// A exists in stencilB
bool OriginalGrid::isIndxInStencil(int indxA, std::vector<int> stencilB) {
    for (int i = 0; i < stencilB.size(); i++) {
        if (stencilB[i] == indxA)
            return true;
    }
    return false;
}

//----------------------------------------------------------------------
void OriginalGrid::computeStencilsKDTree(KDTree* kdtree) {

    if (nb_bnd == 0) {
        printf("**** WARNING! nb_bnd == 0; Did generateGrid() update this value properly?! ******\n");
        printf("**** Resetting nb_bnd and nb_rbf ****\n");
        this->nb_bnd = boundary.size();
        this->nb_rbf = rbf_centers.size();
    }

    if (stencil_size > nb_rbf) {
        int new_stencil_size = (int) (0.5 * nb_rbf);
        new_stencil_size = (new_stencil_size > 1) ? new_stencil_size : 1;
        printf("\n!!!!!!!!!!!!!!!!!!!\nWARNING! Not enough centers to reach specified stencil_size (size: %d for %d RBFs)! Using new stencil size: %d\n!!!!!!!!!!!!!!!!!!!\n\n", stencil_size, nb_rbf, new_stencil_size);
        stencil_size = new_stencil_size;
    }

    // for each node, a vector of stencil nodes (global indexing)
    stencil.resize(nb_rbf);
    printf("stencil vec length: %d, nb_rbf= %d\n", (int)stencil.size(), nb_rbf);

   // vector<double> avg_bnd;
   // vector<double> avg_int;

    this->avg_distance.resize(nb_rbf);
   // avg_bnd.resize(nb_bnd);
   // avg_int.resize(nb_rbf - nb_bnd);

    printf("nb_rbf= %d (bnd: %d)\n", nb_rbf, nb_bnd);
    //printf("nb_bnd= %d\n", nb_bnd);
    //exit(0);

    // O(n^2) algorithm, whose cost is independent of the number of nearest sought
    for (int i = 0; i < nb_rbf; i++) {
        vector<double> center(dim, 0);
        for (int j = 0; j < dim; j++) {
            center[j] = rbf_centers[i][j];
        }


        stencil[i].resize(stencil_size);

        vector<double> nearest_dists;
        vector<int> nearest_ids;
        kdtree->k_closest_points(center, stencil_size, nearest_ids, nearest_dists);

       // printf("Stencil[%d] = {", i);
        for (int j=0; j < stencil_size; j++) {
            int rev_indx = (stencil_size-1)-j;
            stencil[i][j] = nearest_ids[rev_indx];
         //   printf(" %d(%f) ", stencil[i][j], nearest_dists[rev_indx]);
            this->avg_distance[i] += nearest_dists[rev_indx];
        }
        //printf("}\n");
        this->avg_distance[i] /= (stencil_size-1); // ignore the center node
       // printf("avg_dist[%d]= %f\n", i,  this->avg_distance[i]);
       // printf("nb points in stencil: %d\n", (int) stencil[i].size());
    }

    printf("DONE GENERATING STENCILS\n");

#if 0
    double avgint = 0.;
    double avgbnd = 0.;

    printf("avg_int.size() = %d\n", avg_int.size());
    printf("avg_bnd.size() = %d\n", avg_bnd.size());

    if (nb_rbf - nb_bnd > 0) {
        for (int i = 0; i < avg_int.size(); i++) {
            avgint += avg_int[i];
        }
        avgint /= avg_int.size();
    } else {
        avgint = 0.;
    }

    // There should always be boundary point(s)
    for (int i = 0; i < avg_bnd.size(); i++) {
        avgbnd += avg_bnd[i];
    }
    avgbnd /= avg_bnd.size();

    printf("mean of mean interior distances: %f (size: %d)\n", avgint, avg_int.size());
    printf("mean of mean boundary distances: %f (size: %d)\n", avgbnd, avg_bnd.size());
#endif
}
//----------------------------------------------------------------------

void OriginalGrid::computeStencilsRegular() {

    if (stencil_size != 4) {
        printf("computeStencilsRegular requires a 4-point stencil (stencil_size)\n");
    }

    //	stencil_size: not really used

    // create stencils
    // USE ALL POINTS based on a subgrid structure
#if 0
    stencil.resize(nb_rbf);
    //printf("stencil vec length: %d, nb_rbf= %d\n", stencil.size(), nb_rbf);
    //exit(0);

    for (int i = 0; i < nb_rbf; i++) {
        int which_cell = sub_cell[i];
        int wy = which_cell / nx;
        int wx = which_cell - nx*wy;
        //printf("wx,wy= %d, %d\n", wx, wy);

        // Create stencils for each node
        int w;
        for (int wwx = wx - 1; wwx <= wx + 1; wwx++) {
            if (wwx < 0 || wwx == nx) continue;
            for (int wwy = wy - 1; wwy <= wy + 1; wwy++) {
                if (wwy < 0 || wwy == ny) continue;

                vector<int> lst = (*subgrid_index_list)(wwx, wwy);
                for (int l = 0; l < lst.size(); l++) {
                    stencil[i].push_back(lst[l]);
                }
            }
        }
        //printf("rbf %d, stencil_size= %d\n", i, stencil[i].size());
    }
#endif

    // Stencil is the simple 9 point stencil of finite difference
    // All points that surround the given point, including the point itself
#if 1
    stencil.resize(nb_rbf);

    if (nb_rbf != (nb_x * nb_y)) {
        printf("computeStencilsRegular()::nb_rbf should equal nb_x*nb_y\n");
    }

    for (int i = 0; i < nb_rbf; i++) {
        int which_cell = sub_cell[i];
        int wy = which_cell / nx;
        int wx = which_cell - nx*wy;

        // Create stencils for each node
        int w;
        for (int wwx = wx - 1; wwx <= wx + 1; wwx++) {
            if (wwx < 0 || wwx == nx) continue;
            for (int wwy = wy - 1; wwy <= wy + 1; wwy++) {
                if (wwy < 0 || wwy == ny) continue;

                vector<int> lst = (*subgrid_index_list)(wwx, wwy);
                for (int l = 0; l < lst.size(); l++) {
                    stencil[i].push_back(lst[l]);
                }
            }
        }
    }
#endif

    printf("computeStencilsRegular::nb_rbf= %d\n", nb_rbf);
}
//----------------------------------------------------------------------

void OriginalGrid::generateGrid(const char* file, int nb_bnd, int npts)
// npts: total number of points
{
    FILE* fd = fopen(file, "r");
    printf("fd= %ld\n", (long) fd);
    if (fd == 0) {
        printf("could not open file %s\n", file);
        exit(0);
    }

    float x, y, z;

    z = 0.;

    nb_rbf = npts;
    printf("READING %d POINTS FROM FILE: %s\n", nb_rbf, file);
    coord = new ArrayT<double>(3, nb_rbf);
    ArrayT<double>& coordr = *coord;

    for (int i = 0; i < npts; i++) {
        fscanf(fd, "%g%g\n", &x, &y);
        //printf("x,y= %f, %f\n", x,y);
        coordr(0, i) = x;
        coordr(1, i) = y;
        coordr(2, i) = 0.;
        rbf_centers.push_back(Vec3(x, y, z));
    }

    for (int i = 0; i < nb_bnd; i++) {
        boundary.push_back(i);
    }

    nb_rbf = rbf_centers.size();
    nb_bnd = boundary.size();
}
//----------------------------------------------------------------------

void OriginalGrid::generateGrid() {

    // Create a regular perturbed grid
    rmax = 1.;
    xmin = -rmax;
    xmax = rmax;
    ymin = -rmax;
    ymax = rmax;
    zmin = 0.; // 2D
    zmax = 0.;
    dx = (xmax - xmin) / (nb_x - 1.); // nb_x = nb points
    dy = (ymax - ymin) / (nb_y - 1.);
    dz = 1.0;
    pert = 0.0 * dx;

    printf("dx = %f\n", dx);
    printf("dy = %f\n", dy);

    nb_rbf = nb_x * nb_y * nb_z;
    maxint = (double) ((1 << 31) - 1);

    coord = new ArrayT<double>(3, nb_rbf); //  1000 rbfs.
    ArrayT<double>& coordr = *coord;

    int count = 0;

    for (int k = 0; k < 1; k++) { // 2D
        for (int j = 0; j < nb_y; j++) {
            for (int i = 0; i < nb_x; i++) {
                double x = xmin + i * dx + randf(-pert, pert) + 0.12 * dx * 0.5;
                double y = ymin + j * dy + randf(-pert, pert) + 0.12 * dy * 0.5;
                double z = 0.0; // 2D
                coordr(0, count) = x;
                coordr(1, count) = y;
                coordr(2, count) = z;
                rbf_centers.push_back(Vec3(x, y, z));
                // given i,j, rbf index i: i+nb_x*j
                if (i == 0 || i == (nb_x - 1) || j == 0 || j == (nb_y - 1)) {
                    boundary.push_back(i + nb_x * j); // boundary point
                }
                count++;
            }
        }
    }

    nb_rbf = rbf_centers.size();
    nb_bnd = boundary.size();

    printf("count= %d\n", count);
    printf("nbx_x, nby_y= %d, %d\n", nb_x, nb_y);
    printf("rbf_centers.size= %d\n", (int) rbf_centers.size());
    printf("boundary.size= %d\n", (int) boundary.size());
}
//----------------------------------------------------------------------

void OriginalGrid::generateSubGrid() {
    // subgrid size
    count = 0;

    ArrayT<double>& coordr = *coord;

    // make the grid overlay extend beyond original grid
    double sdx = (xmax - xmin) / (nx); // number of cells: nx*ny
    double sdy = (ymax - ymin) / (ny);
    sdx += sdx / nx;
    sdy += sdy / ny;

    double sdxmin = xmin - 0.5 * sdx;
    double sdxmax = xmax + 0.5 * sdx;
    double sdymin = ymin - 0.5 * sdy;
    double sdymax = ymax + 0.5 * sdy;
    sdx = (sdxmax - sdxmin) / nx; // number of cells: nx*ny
    sdy = (sdymax - sdymin) / ny;

    double sdz;
    if (nz == 1) {
        sdz = 0.; // 2D
    } else {
        sdz = (zmax - zmin) / (nz - 1.);
    }

    printf("sdx,sdy= %f, %f\n", sdx, sdy);
    printf("xmin,ymin= %f, %f\n", xmin, ymin);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            (*subgrid_index_list)(i, j).resize(0);
        }
    }

    //printf("nb_rbf= %d\n", nb_rbf);
    //printf("====\n");
    for (int i = 0; i < nb_rbf; i++) {
        int ix = (int) ((coordr(0, i) - sdxmin) / sdx);
        int iy = (int) ((coordr(1, i) - sdymin) / sdy);
        //printf("ix,iy= %d, %d\n", ix, iy);
        //printf("coord: %f, %f\n", (coord(0,i)), (coord(1,i)));
        int iz = 1; // 2D
        if (ix == nx) ix--;
        if (iy == ny) iy--;
        (*subgrid_index_list)(ix, iy).push_back(i); // global index

        // replace by 2D array
        sub_cell.push_back(ix + nx * iy); // which sub_cell is rbf[i] within?
    }

    int sum = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            //printf("size subgrid[%d,%d] = %d\n", i, j, (*subgrid_index_list)(i,j).size());
            sum += (*subgrid_index_list)(i, j).size();
        }
    }
    printf("sum = %d\n", sum);
}
//----------------------------------------------------------------------

int OriginalGrid::randi(int i1, int i2) {
    double r = (double) (random() / maxint);
    return (i1 + r * (i2 - i1));
}
//----------------------------------------------------------------------

double OriginalGrid::randf(double f1, double f2) {
    return f1 + (f2 - f1)*(random() / maxint);
}
//----------------------------------------------------------------------
// Laplace operator using 5-point 2nd order stencil
// use a stencil that is [-1 2 -1] in x and [-1 2 -1] in y
// nothing along the boundaries since Dirichlet (use zero)

void OriginalGrid::laplace() {
    // specialized to Cartesian only

    int nb_rbf = rbf_centers.size();
    cart_stencil.resize(nb_rbf);
    cart_weights.resize(nb_rbf);

    double dxi2 = 1. / (dx * dx);
    double dyi2 = 1. / (dy * dy);
    //dxi2 = dyi2 = 1.;

    for (int j = 1; j < (nb_y - 1); j++) {
        for (int i = 1; i < (nb_x - 1); i++) {
            int ic = i + nb_x*j;
            int im = ic - 1;
            int ip = ic + 1;
            int jm = ic - nb_x;
            int jp = ic + nb_x;
            vector<int>& st = cart_stencil[ic];
            st.push_back(ic);
            st.push_back(im);
            st.push_back(ip);
            st.push_back(jm);
            st.push_back(jp);
            vector<double>& w = cart_weights[ic];
            w.push_back(-(2. * dxi2 + 2 * dyi2));
            w.push_back(dxi2);
            w.push_back(dxi2);
            w.push_back(dyi2);
            w.push_back(dyi2);
        }
    }

#if 0
    for (int i = 0; i < cart_weights.size(); i++) {
        vector<double>& w = cart_weights[i];
        printf("weight %d, nb pts in stencil: %d\n", i, w.size());
        for (int j = 0; j < w.size(); j++) {
            printf("%f ", w[j]);
        }
        printf("\n");
    }
#endif
}
//----------------------------------------------------------------------
// computes lapl(scal) and stores in laplace[]

vector<double>& OriginalGrid::computeCartLaplacian(vector<double>& scalar) {
    double der;

    for (int i = 0; i < cart_weights.size(); i++) {
        vector<double>& w = cart_weights[i];
        vector<int>& st = cart_stencil[i];
        der = 0.0;
        int n = st.size();
        for (int s = 0; s < n; s++) {
            der += w[s] * scalar[st[s]];
            //printf("w[%d]= %f\n", i, w[s]);
        }
        laplacian[i] = der;
    }
    printf("exit computeCartLaplacian\n");
    return laplacian;
}
//----------------------------------------------------------------------

double OriginalGrid::minimum(vector<double>& vec) {
    double min = 1.e10;

    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] < min) {
            min = vec[i];

        }
    }
    return min;
}
//----------------------------------------------------------------------
#if 0

vector< vector< int > > OriginalGrid::decomposeDomain(int num_cpus) {
    for (int i = 0; i < num_cpus; i++) {
        decomposed_domain.pushback(stencil);
    }

    return decomposed_domain;
}
#endif
//----------------------------------------------------------------------

void OriginalGrid::sortNodes() {
    ArrayT<double>& coordr = *coord;
    for (int i = 0; i < this->boundary.size(); i++) {
        // We only need to roughly sort the nodes so the boundary is first and the
        // interior is second

        // Run through all boundary nodes. If the node is in the boundary set (which should be ordered),

        if (boundary[i] != i) {
            // backup interior
            double tempx = coordr(0, i);
            double tempy = coordr(1, i);
            double tempz = coordr(2, i);

            // overwrite interior with boundary
            coordr(0, i) = coordr(0,boundary[i]);
            coordr(1, i) = coordr(1,boundary[i]);
            coordr(2, i) = coordr(2,boundary[i]);

            // restore interior in the boundary position
            coordr(0, boundary[i]) = tempx;
            coordr(1, boundary[i]) = tempy;
            coordr(2, boundary[i]) = tempz;

            // Update the boundary index into coords
            boundary[i] = i;
        }
    }

}