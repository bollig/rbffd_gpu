#ifndef __VTU_DOMAIN_WRITER_H__
#define __VTU_DOMAIN_WRITER_H__

#ifdef USE_VTK

// Writes a Domain class grid to file. 

#define USE_POLY_DATA 0

#include "grids/domain.h"
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
#include <vtkCellArray.h>
#include <vtkPolyVertex.h>
#include <vtkVertex.h> 
#include <vtkLine.h>

class VtuDomainWriter 
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
	public: 

		virtual ~VtuDomainWriter() {
			ugrid->Delete();
			pts->Delete();
			uwriter->Delete();
		}


		VtuDomainWriter(Domain* subdomain, int mpi_rank, int mpi_size)
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
			for (unsigned int i = 0; i < subdomain->getNodeListSize(); i++) {
				NodeType& n = nodes[i]; 
				pts->SetPoint(i, n.x(), n.y(), n.z()); 
			}


			//            stns = vtkCellArray::New(); 
			//            stns->SetNumberOfCells(subdomain->getStencilsSize());
			vtkCellArray* cell_array = vtkCellArray::New(); 
			vtkIdType cell_type = NULL;  
			for (unsigned int i = 0; i < subdomain->getStencilsSize(); i++){
				unsigned int ssize = subdomain->getStencilSize(i); 
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
				for (unsigned int j = 0; j < ssize; j++) {
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
			cell_array->Delete();

			ugrid->SetPoints(pts);

			uwriter->SetInput(ugrid);

			char fname[FILENAME_MAX];
			sprintf(fname, "subdomain_%d.%d.vtk", mpi_size, mpi_rank);
			uwriter->SetFileName(fname);
			uwriter->Write();
		}
};

#endif
#endif 
