# Fluctuations in a Dual Labor Market

This repository contains the replication codes for my paper *Fluctuations in a Dual Labor Market*. They are written in Julia 1.4; `Manifest.toml` and `Project.toml` compound the necessary packages and dependencies. I amended a bit the `SymEngine.jl` package to include the Gauss error function; it is necessary to generate symbolic derivatives of the normal distributions c.d.f. The file pointer must be set to the root of the repository for the relative paths to work. 

Most functions are documented in the code itself. Here is the description of the main files

- `calibration.jl`: calibrates the dual model and its classic couterpart
- `smc.jl`: carries out the different Sequential Monte Carlo estimations using parallelization; the estimations of the dual and classic model, the estimations of the dual model with different filters and multiple estimations of the classic and dual models to compute the marginal data density. The results are stored in the `output` folder
- `simulations.jl`: simulates the model to compute the posterior-related fit measures using parallelization and stores the results into the `output` folder
- `output.jl`: returns the ouput tables in LaTeX format and saves figures into the `figures` folder
- `inflation.jl`: simulates the model with the different components of inflation and returns a relevant table in LaTeX format
- `perturbation.jl`: solves the model with a third-order perturbation method and plots relevant impulse response functions
- `var.jl`: estimates VAR with different combinations of time series for the sake of robustness.

The `functions` folder contains the necessary modules. Some modules define the models that are to be estimated through Sequential Monte Carlo.

- `EstDual.jl`: defines the prior distributions, loads the data for estimation from the folder `input` and filters it
- `Classic.jl`, `Dual.jl` and `Dual_pi.jl` define the classic model, the dual model and the dual model with inflation components
- `SMC_main.jl`: main functions to carry out the Sequential Monte Carlo estimation
- `SMC_aux.jl`: auxiliary functions that will be used by functions in `SMC_main.jl` in parallel
- `OutputFunctions.jl`: functions to shape output from estimates and simulations
- `InflationFunctions.jl`: solves and simulates the dual model with the relevant inflation component.

Other modules implement the third-order perturbation method and the model to be solved through the latter. I relied on Oren Levintahl's paper *Fifth-Order Perturbation Solution to DSGE Models*, JEDC (2017) to do it aesthetically.

- `SymDual`: defines the symbolic version of the dual model
- `EconModel.jl`: defines a class of economic models and the associated differentiation and solving tools
- `Kamenik.jl`: implements Kamenik's method to solve Sylvester equations as describe in his paper *Solving sdge models: A new algorithm for the Sylvester equation*, Computational Economics (2005)
- `SparseSym.jl`: tools to manipulate sparse arrays of symbols in 2 dimensions and more.
- `IRF.jl`: functions to computes the stochastic steady-state and plot impulse response functions starting from it.

Finally, some modules are more general, carrying out technical tasks

- `FilterData.jl`: implements linear detrending as well as first difference, Hodrick-Prescott and Hamilton filtering
- `Macros.jl`: useful macros to avoid copy-paste errors and shorten the code
- `VAR.jl`: estimates VAR and simulates VAR impulse response functions

Beyond the simple replication of my paper, my code can be reused in other frameworks. For a Sequential Monte Carlo estimation, all it takes is modifying `Dual.jl`, updating priors as well as the filtering method in `EstDual.jl` and setting the estimation data in `input`. In the same manner, solving a model using a third-order perturbation method only requires changing the `SymDual.jl` file.