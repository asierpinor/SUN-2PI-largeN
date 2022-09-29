#!/bin/bash

# Constants

# M_PI=3.14159265358979323846     # Pi

# Parameters have to be written in command line in the order specified in the "varlist" of the .hpp file

# System parameters
N=10                # Number of lattice points
DIM=1               # Lattice dimension
Nx=1
Ny=1
Nz=1
J=0.25					# Interaction parameter = t^2/U
Nm=2				# Number of magnetic sublevels

# Initial Conditions
IC=1                    # Type of initial condition. 0: sz, 1: sx, 2: sy, 3: tilted, 31: symmetric 3-level, 32: asymmetric 3-level, 33: random 3-level, 99: weird (N>2)
qx=$((N/2))					# x-component of spiral vector in powers of 2*pi/Nx. Neel state: $((N/2))
qy=$qx					# y-component ...
qz=$qx					# z-component ...

# Approximation
approx=1                # Type of approximation. 0: mean-field, 1: NLO
channel=0               # Channel resummation: 0: s-channel, 1: u-channel, 2: t-channel

# Dynamics
Nt=200                   # Number of time steps + 1 (i.e. number of times saved starting at 0)
Ntcut=$Nt               # Memory cutoff, i.e. number of time array points that will be saved. For t>Ntcut forget past.
dt=0.02                  # Time step size
difmethod=11				# Method to compute differential part. 0: Euler, 1k: P(EC)^k, 2k: P(EC)^kE

# Sonstiges
tnum=2                  # Number of threads to be used in parallel
seed=1                 # seed for random number generator. If negative seed -> extract from current time.
max_mem=0.5				# Maximal memory to be allocated by correlators and self-energies (in Gigabytes)

# Output
nfile=8                 # This number will be appended to the name of the output files
Nbins=250               # Number of bins for Fourier momenta
spinstep=1              # Print correlators after each outstep time interval. NEGATIVE -> no output
energystep=1           # Output energy in steps of energystep. NEGATIVE -> no output

# Folders
folder=data             # Folder to save the data
#folder=../Simulations/spirals_experiments/N$((Nm))

# Note: careful with bash arithmetics. Bash does not perform double divisions automatically. For this use "bc" or use a C++ expression parser.


./fermions_2PI $N $DIM $Nx $Ny $Nz $J $Nm $IC $qx $qy $qz $approx $channel $Nt $Ntcut $dt $difmethod $tnum $seed $max_mem $nfile $Nbins $spinstep $energystep $folder











