#ifndef __VTU_PDE_WRITER_H__
#define __VTU_PDE_WRITER_H__

#include "pde_writer.h"
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkXMLWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkPoints.h> 
#include <vtkPointData.h>
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
        vtkUnstructuredGridWriter* uwriter;
        vtkUnstructuredGrid* ugrid; 
        vtkPoints* pts; 
        vtkCellArray* stns; 

    public: 
        VtuPDEWriter(Domain* subdomain_, Heat* heat_, Communicator* comm_unit_, int local_write_freq_, int global_write_freq_)
            : PDEWriter(subdomain_, heat_, comm_unit_, local_write_freq_, global_write_freq_) 
        { 
//            uwriter = vtkUnstructuredGridWriter::New(); 
            uwriter = vtkUnstructuredGridWriter::New();
            ugrid = vtkUnstructuredGrid::New(); 

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

            for (size_t i = 0; i < subdomain->getStencilsSize(); i++) {
                size_t ssize = subdomain->getStencilSize(i); 
                StencilType& st = subdomain->getStencil(i);
#if 0
                vtkPolyVertex* cell = vtkPolyVertex::New(); 
                cell->GetPointIds()->SetNumberOfIds(ssize); 
                for (int j = 0; j < ssize; j++) {
                    cell->GetPointIds()->SetId(j, st[j]); 
                }
                ugrid->InsertNextCell(cell->GetCellType(), cell->GetPointIds());
                cell->Delete();
#else 
#if 0
                vtkVertex* cell = vtkVertex::New();
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


                for (int j = 0; j < ssize; j++) {
                    vtkLine* cell = vtkLine::New();
                    cell->GetPointIds()->SetNumberOfIds(2);

                    cell->GetPointIds()->SetId(0,i);
                    cell->GetPointIds()->SetId(1, st[j]); 
                    ugrid->InsertNextCell(cell->GetCellType(), cell->GetPointIds());
//                    cell->Delete();
                }
#endif 
#endif 


            }

            vtkFloatArray* sol = vtkFloatArray::New();
            sol->SetName("Solution"); 
            sol->SetNumberOfValues(subdomain->getNodeListSize());

            ugrid->SetPoints(pts);
            ugrid->GetPointData()->SetScalars(sol);

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
            vtkFloatArray* s = (vtkFloatArray*)ugrid->GetPointData()->GetScalars(); 
            for (int i = 0; i < s->GetSize(); i++) {
                s->SetValue(i, subdomain->U_G[i]);
            }

            char fname[FILENAME_MAX]; 
            sprintf(fname, "subdomain_%d-%04d.vtk", comm_unit->getRank(), iter);
 //           sprintf(fname, "subdomain_%d.vtk", comm_unit->getRank(), iter);
            uwriter->SetFileName(fname);
//            uwriter->WriteNextTime(iter);
            uwriter->Write();

            // subdomain->writeLocalToFile(iter);
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
            //    subdomain->writeToFile(); 

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
        }

};

#endif
