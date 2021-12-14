This is the code for my final project of CS556

The original code is copied from 
https://github.com/SealUofI/SemPy

We do re-write 80%+ of the code and put it under the folder `my_sem/`
The main driver are listed below:

### drivers:
- `example_fix.py`: minimum efforts to make the example runnable.
   only do up to (single) non-deformed element, but the main operators consoder deformed geometry

- `ref2d_v1.py`: clean up, revisit all functions, start to use mine version under (`my_sem/`)
   - D-N BCs, E=1, non-deformed
   - cg, pcg (mass, jacobi, fdm, chebyshev+jac)

- `ref2d_v2.py`: for multigrid
   - 2 levels vcycle 
   - 3 levels vcycle 
   - TODO: flexible pre/post smoothing 

### modules and functions

- `sempy/`

   Currently, we only call the following function to genrate the mesh.
   ```
      from sempy.meshes.box import reference_2d 
   ```
   Potentially, we should be able to read any gmsh (`.msh`) via the sempy reader.


- `my_sem/`

   - `gen_geom.py`    
      compute the geometric factors, jacobian


   - `gen_semhat.py`
      generate the GLL points, Derivative matrix. This is copied from 
      ```
         sempy/quadrature.py
         sempy/derivative.py
      ```

   - `linear_solvers.py`   
      Right now, we have three
      ```
         cg
         pcg
         arnoldi
      ```
      
      CG and PCG support any dimension as long as user provide the matrix-vector operations (both forward operator and the preconditioner)
      and the "mask" (zero out the non-degree of freedon points, can be weighted)

      Arnoldi reshape the matrices to eventually build a 1D matrix for estimating the maximum eigenvalues.


   - `preconditioners.py`     
      This only works with PCG. For each preconditioner, call `*_setup` to pre-compute the needed info into the module wise global variables. Then the actual precoditioner will use.

      So far, we have
      ```
      precon_mass
      precon_jac
      precon_fdm_2d
      precon_chebyshev
      ```


   - `multigrid.py`     
      The main interface is twolevels and threelevels. We do setup the matrices at each levels and setup the FDM as the coarse grid solver. Note that, the current example will not read arbitrary funAx. We build our own matrix-vec here (see fun_MG_Ax)


   - `interp1d.py`      
      This build the interpolation matrix between two grids. 
      The code is copied from Nek5000 and modified into python

      Use `test_interp()` to see demo figures


   - 'util.py'    
      Some utility functions such as norm and the timer     



