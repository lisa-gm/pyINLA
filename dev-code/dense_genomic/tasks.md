# State of the project


Next:
- Try to instanciate the Model and play with it trivially
- Dive into Optimizer/ObjectiveFunction/FD/GradientStrategy
	- It seems liek this buffer and commit_buffer with the HPM has some more design weirdness to handle. Maybe the explicit array interface is need when trying tentative points. The buffer should likely always only contain current tentative (but un-perturbed by the FD) points.




# Tasks
1. Monolithic pipeline using DenseMatrix data-structure
	a. Finite Diff Dense
	b. AutoDiff Dense
	c. Distributed Finite Diff
	d. Distributed AutoDiff
2. Implementation of the BlockMatrix, relying under the hood on DenseMatrix data-structures
3. Implementation of the DiagMatrix, then replacement in the BlockMatrix


## 1. Monolithic pipeline using DenseMatrix data-structure
Next steps
- Abstraction for the hyper-parameters and optimization
- Model() abstraction for precision matrices assembly
- Simple dense matrix pipeline