#ifndef __VTU_PDE_WRITER_H__
#define __VTU_PDE_WRITER_H__

#define USE_POLY_DATA 0

#include "pde_writer.h"
#include <vtkXMLWriter.h>

#if USE_POLY_DATA
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#else 
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#endif 

#include <vtkPoints.h> 
#include <vtkPointData.h>
#include <vtkDoubleArray.h>  // scalars
#include <vtkFloatArray.h>  // scalars
#include <vtkCellArray.h>
#include <vtkPolyVertex.h>
#include <vtkVertex.h> 
#include <vtkLine.h>

class VtuPDEWriter : public PDEWriter
{
    protected: 
        // Why do we choose unstructuredgrid and not polydata? 
        // "The main difference between vtkPolyData and vtkUnstructuredGrid is 
        // whether 3D cells (tetrahedra, hexahedra, etc.) may be used. 
        // vtkUnstructuredGrid supports such cells; vtkPolyData does not." 
        // http://vtk.1045678.n5.nabble.com/Are-point-clouds-defined-as-Unstructured-Points-td1238809.html
        // We dont want cells, but we do want vtk_poly_vertex
#if USE_POLY_DATA
        vtkPolyDataWriter* uwriter; 
        vtkPolyData* ugrid;
#else 
        vtkUnstructuredGridWriter* uwriter;
        vtkUnstructuredGrid* ugrid; 
#endif 

        vtkPoints* pts; 
        //vtkCellArray* stns; 

        vtkDoubleArray* sol;
        vtkDoubleArray* exact;
        vtkDoubleArray* abs_err;
        vtkDoubleArray* rel_err;
        vtkDoubleArray* diff;
 
        char* sol_name;
        char* exact_name;
        char* abs_err_name;
        char* rel_err_name;
        char* diff_name;

    public: 
        VtuPDEWriter(Domain* subdomain_, TimeDependentPDE* heat_, Communicator* comm_unit_, int local_write_freq_, int global_write_freq_)
            : PDEWriter(subdomain_, heat_, comm_unit_, local_write_freq_, global_write_freq_) 
        { 
#if USE_POLY_DATA
            uwriter = vtkPolyDataWriter::New();
            ugrid = vtkPolyData::New(); 
#else 
            uwriter = vtkUnstructuredGridWriter::New();
            ugrid = vtkUnstructuredGrid::New(); 
#endif 

            // Get Points: 
            pts = vtkPoints::New();
            pts->SetDataTypeToFloat();
            pts->SetNumberOfPoints(subdomain->getNodeListSize());

            std::vector<NodeType>& nodes = subdomain->getNodeList(); 
            for (size_t i = 0; i < subdomain->getNodeListSize(); i++) {
                NodeType& n = nodes[i]; 
                pts->SetPoint(i, n.x(), n.y(), n.z()); 
            }


            //            stns = vtkCellArray::New(); 
            //            stns->SetNumberOfCells(subdomain->getStencilsSize());
            vtkCellArray* cell_array = vtkCellArray::New(); 
            vtkIdType cell_type;  
            for (size_t i = 0; i < subdomain->getStencilsSize(); i++){
                size_t ssize = subdomain->getStencilSize(i); 
                StencilType& st = subdomain->getStencil(i);
#if 0
                vtkPolyVertex* cell = vtkPolyVertex::New(); 
                cell_type = cell->GetCellType();
                cell->GetPointIds()->SetNumberOfIds(ssize); 
                for (int j = 0; j < ssize; j++) {
                    cell->GetPointIds()->SetId(j, st[j]); 
                }
                cell_array->InsertNextCell(cell);//->GetCellType(), cell->GetPointIds());
                cell->Delete();
#else 
#if 0
                vtkVertex* cell = vtkVertex::New();
                cell_type = cell->GetCellType();
                cell->GetPointIds()->SetNumberOfIds(1);
                cell->GetPointIds()->SetId(0,i);
                ugrid->InsertNextCell(cell->GetCellType(), cell->GetPointIds());
                cell->Delete();
#else 
#if 0
                vtkIntArray* sten_id = vtkIntArray::New();
            sten_id->SetName("StencilId"); 
            sten_id->SetNumberOfValues(subdomain->getNodeListSize());

            for (int i = 0; i < subdomain->getStencilsSize(); i++) {
                sten_id->SetValue(i, subdomain->getStencil(i)[0]);
            }
#endif
#if 1
                for (int j = 0; j < ssize; j++) {
                    vtkLine* cell = vtkLine::New();
                    cell_type = cell->GetCellType();
                    cell->GetPointIds()->SetNumberOfIds(2);

                    cell->GetPointIds()->SetId(0,i);
                    cell->GetPointIds()->SetId(1, st[j]); 
                    cell_array->InsertNextCell(cell); //cell->GetCellType(), cell->GetPointIds());

                    // This deletes the local copy, but not the copy that was
                    // instantiated inside ugrid
                    cell->Delete();
                }
#endif 
#endif 
#endif 
            }

#if USE_POLY_DATA
            ugrid->SetPolys(cell_array);
#else 
            ugrid->SetCells(cell_type, cell_array);
#endif 

            sol = vtkDoubleArray::New();
            sol_name = "Computed Solution";
            sol->SetName(sol_name); 
            sol->SetNumberOfComponents(1);
            sol->SetNumberOfValues(subdomain->getNodeListSize());

            exact = vtkDoubleArray::New();
            exact_name = "Exact Solution";
            exact->SetName(exact_name); 
            exact->SetNumberOfComponents(1);
            exact->SetNumberOfValues(subdomain->getNodeListSize());


            abs_err = vtkDoubleArray::New();
            abs_err_name = "Absolute Error By Node";
            abs_err->SetName(abs_err_name); 
            abs_err->SetNumberOfComponents(1);
            abs_err->SetNumberOfValues(subdomain->getNodeListSize());

            rel_err = vtkDoubleArray::New();
            rel_err_name = "Relative Error By Node";
            rel_err->SetName(rel_err_name); 
            rel_err->SetNumberOfComponents(1);
            rel_err->SetNumberOfValues(subdomain->getNodeListSize());

            diff = vtkDoubleArray::New();
            diff_name = "Diffusivity";
            diff->SetName(diff_name); 
            diff->SetNumberOfComponents(1);
            diff->SetNumberOfValues(subdomain->getNodeListSize());


            ugrid->SetPoints(pts);
            // This sets the primary: 
            ugrid->GetPointData()->SetScalars(sol);
            // These expand with secondary arrays
            ugrid->GetPointData()->AddArray(exact);
            ugrid->GetPointData()->AddArray(abs_err);
            ugrid->GetPointData()->AddArray(rel_err);
            ugrid->GetPointData()->AddArray(diff);

            uwriter->SetInput(ugrid);

            char fname[FILENAME_MAX];
            sprintf(fname, "subdomain_%d.vtk", comm_unit->getRank());
            uwriter->SetFileName(fname);
           // uwriter->SetNumberOfTimeSteps(10);
           // uwriter->Start();
        }

        virtual ~VtuPDEWriter() {
            this->writeFinal(); 
          //  uwriter->Stop();
            uwriter->Delete();
            ugrid->Delete();
            pts->Delete();
            //stns->Delete();
            sol->Delete();
            exact->Delete();
            abs_err->Delete();
            rel_err->Delete();
        }

        // MASTER process only!
        void writeMetaData() {
            // write the metadata xml file (partitioned VTU) assuming
            // we have # mpi ranks equal to the number of partitioned files
        }

        void writeVTU(int iter) {
            // construct unstructured grid by getting data from classes
            // write to file

            // Update the solution: 
            vtkDoubleArray* s0 = (vtkDoubleArray*)ugrid->GetPointData()->GetArray(sol_name); 
            for (int i = 0; i < s0->GetSize(); i++) {
                s0->SetValue(i, heat->getLocalSolution(i));
            }

            vtkDoubleArray* s1 = (vtkDoubleArray*)ugrid->GetPointData()->GetArray(exact_name); 
            for (int i = 0; i < s1->GetSize(); i++) {
                s1->SetValue(i, heat->getExactSolution(i));
            }

#if 1
#define USE_PDE_ERR 1
            // Update the abs_error: 
            vtkDoubleArray* s2 = (vtkDoubleArray*)ugrid->GetPointData()->GetArray(abs_err_name); 
            for (int i = 0; i < s2->GetSize(); i++) {
#if USE_PDE_ERR
                s2->SetValue(i, heat->getAbsoluteError(i));
#else 
                double absval = fabs(s1->GetValue(i) - s0->GetValue(i)); 
                s2->SetValue(i, absval);
#endif 
            }

            // Update the abs_error: 
            vtkDoubleArray* s3 = (vtkDoubleArray*)ugrid->GetPointData()->GetArray(rel_err_name); 
            for (int i = 0; i < s3->GetSize(); i++) {
#if USE_PDE_ERR
                s3->SetValue(i, heat->getRelativeError(i));
#else 
                 if (fabs(s1->GetValue(i)) > 1e-10) {
                    s3->SetValue(i, s2->GetValue(i) / fabs( s1->GetValue(i)));
                 } else {
                    s3->SetValue(i, 0.);
                 }
#endif 
            }

            // Update the abs_error: 
            vtkDoubleArray* s4 = (vtkDoubleArray*)ugrid->GetPointData()->GetArray(diff_name); 
            for (int i = 0; i < s4->GetSize(); i++) {
                s4->SetValue(i, heat->getDiffusivityAtNode(i));
            }
#endif 
            char fname[FILENAME_MAX]; 
            sprintf(fname, "subdomain_rank%d_of%d-%04d.vtk", comm_unit->getRank(),comm_unit->getSize(), iter);
 //           sprintf(fname, "subdomain_%d.vtk", comm_unit->getRank(), iter);
            uwriter->SetFileName(fname);
//            uwriter->WriteNextTime(iter);
            uwriter->Write();

            std::cout << "VTU FILE WRITTEN" << std::endl;
            //Write ASCII files: 
            //heat->writeLocalSolutionToFile(iter);
        }

        void writeGlobalVTU(int iter) {
            // construct unstructured grid for GLOBAL values
            // write only a subset of data to file (solution, grid)
        }

        /** 
         * Tasks to perform only on the initial iteration.
         * For example, maybe only write the grid nodes ONCE
         * at the first iteration. Or maybe you want to override
         * this to only write METADATA once at initialization. 
         */
        virtual void writeInitial() { 
//            subdomain->writeToFile();
            if (comm_unit->isMaster()) 
                this->writeMetaData(); 
        }

        /** 
         * Write the local solution, error, etc for a subdomain
         */
        virtual void writeLocal(int iter) { 
            this->writeVTU(iter); 
        }

        /** Consolidate and write the global solution and grid to file 
         *  NOTE: this is currently a subset of the data available when
         *      writing local information (e.g., no error)
         */
        virtual void writeGlobal(int iter) {
            comm_unit->consolidateObjects(subdomain);
            comm_unit->barrier();

            this->writeGlobalVTU(iter);
            //subdomain->writeGlobalSolutionToFile(iter);
        }

};

#endif
