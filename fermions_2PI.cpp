#include <iostream>
#include <fstream>
//#include <stdio.h>
//#include <unistd.h>
#include <vector>               // std:: vector
#include <complex>
#include <cmath>                // sin, cos etc.
#include <fftw3.h>              // FFTW
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <string>
#include <sstream>              // for input/ouput with strings

#include <Eigen/Dense>

#include "fermions_2PI.hpp"

using namespace std;
using namespace Eigen;

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   // pi
#endif

#ifndef TYPE_DEF
#define TYPE_DEF
typedef double my_double;
typedef complex<my_double> comp;
typedef vector< comp > state_type;  // The type of container used to hold the state, vector of spins.
#endif


// : r(rndm)
fermions_2PI::fermions_2PI(gsl_rng * rndm)
{
	I = comp(0.0,1.0);
	
	// Set up system size and constants
    dim = 0;
    if (Nx>1) { dim++; Nlist.push_back(Nx); }
    if (Ny>1) { dim++; Nlist.push_back(Ny); }
    if (Nz>1) { dim++; Nlist.push_back(Nz); }
    if (dim==0) { dim = DIM; Nlist.resize(dim,N); }
    
	funny_constants();
	
	if (dim>=1) { Q.push_back(2*M_PI*qx/Nlist[0]); Qind.push_back(qx); }
	if (dim>=2) { Q.push_back(2*M_PI*qy/Nlist[1]); Qind.push_back(qy); }
	if (dim>=3) { Q.push_back(2*M_PI*qz/Nlist[2]); Qind.push_back(qz); }
	
	// Special arrays
	fill_table_nn(); // S and U-channels
	if (channel==1) fill_table_spacemom(); // U-channel
	
	// Define sizes of arrays
    switch (channel) {
        case 0: // S-channel
			size_Frho = Nm2 * Ntcut2halfsize;
			size_D = Nco * Ntcut2halfsize;
			size_Pi = size_D;
			size_chi = 0;
			size_rhsFrho = Nm2 * Ntcut;
			size_Jk = 0;
			size_Jijinvk = 0;
			size_effmass = Nm2;
            break;
			
	    case 1: // U-channel
			size_Frho = Nm2 * Ntcut2halfsize;
			size_D = Nm4 * Ntcut2halfsize * Ntotal;
			size_Pi = Nm4 * Ntcut2halfsize;
			size_chi = Nm2 * Ntcut;
			size_rhsFrho = Nm2 * Ntcut;
			size_Jk = Nm2 * Ntotal;
			size_Jijinvk = Ntotal;
			size_effmass = Nm2;
	        break;
            
        default:
            cout << "Error: Wrong type of channel chosen.\n";
            break;
    }
	
	// Check memory requirements
	memoryOK = false;
	int64_t num_doubles = 2 * ( 4*size_Frho + 2*size_D + 2*size_Pi + size_chi + 4*size_rhsFrho + size_Jk + size_Jijinvk + size_effmass );  // assume PEC
	double memory =  num_doubles * 8 / pow(1024,3); // in Gigabytes
	cout << "Estimated memory required: " << memory << " Gb" << endl;
	
	if (memory<max_mem) memoryOK = true;
	else cout << "ERROR: Memory required larger than max_mem limit provided. End program.\n";
	
	
	// Initialize containers
	if (memoryOK)
	{
		F = new comp [ size_Frho ];
		rho = new comp [ size_Frho ];
	    selfF = new comp [ size_Frho ];
	    selfrho = new comp [ size_Frho ];
	
		chi = new comp [ size_chi ];
		DF = new comp [ size_D ];
		Drho = new comp [ size_D ];
		PiF = new comp [ size_Pi ];
		Pirho = new comp [ size_Pi ];
		
		Jk = new double [ size_Jk ];
		Jijinvk = new double [ size_Jijinvk ];
		effmass = new double [ size_effmass ];
	
	    if (difmethod==0)	// Euler
		{
			rhs_F = new comp [ size_rhsFrho ];
			rhs_rho = new comp [ size_rhsFrho ];
	    }
	
		if (difmethod>10 && difmethod<30)	// P(EC)^k or P(EC)^kE
		{
			ECiter = difmethod % 10;
		
			rhs_F = new comp [ size_rhsFrho ];
			rhs_rho = new comp [ size_rhsFrho ];
			rhs_F_plus1 = new comp [ size_rhsFrho ];
			rhs_rho_plus1 = new comp [ size_rhsFrho ];
		}
		
		// Fill Jk, effmass
		if (channel==1) { fill_Jk(); fill_Jijinvk();} // U-channel
		fill_effmass();
	}
    
}


fermions_2PI::~fermions_2PI()
{
	if (memoryOK)
	{
	    delete [] F;
	    delete [] rho;
	    delete [] selfF;
	    delete [] selfrho;
		
		delete [] chi;
		delete [] DF;
		delete [] Drho;
		delete [] PiF;
		delete [] Pirho;
		
		delete [] Jk;
		delete [] Jijinvk;
		delete [] effmass;
	
	    if (difmethod==0)	// Euler
		{
			delete [] rhs_F;
			delete [] rhs_rho;
	    }
	
		if (difmethod>10 && difmethod<30)	// P(EC)^k or P(EC)^kE
		{
			delete [] rhs_F;
			delete [] rhs_rho;
			delete [] rhs_F_plus1;
			delete [] rhs_rho_plus1;
		}
	}
    
	
	/*
    delete [] ind_nneigh;
    
    delete [] physMom;
    delete [] momPerBin;
    delete [] binAvFactor;
    delete [] whichBin;
    delete [] kBin;
    
    delete [] spectrum;
    delete [] spectrum_phi2cl;
    delete [] spectrum_pi2cl;
    
    fftw_free(Fqu_k_realsym); fftw_free(Fqu_x_realsym);
    fftw_free(Kqu_k_realsym); fftw_free(Kqu_x_realsym);
    fftw_destroy_plan(fft_Fqu_realsym_xk);  fftw_destroy_plan(fft_Fqu_realsym_kx);
    fftw_destroy_plan(fft_Kqu_realsym_xk);  fftw_destroy_plan(fft_Kqu_realsym_kx);
    
	*/
}





void fermions_2PI::funny_constants ()
{
    Ntotal=1; for (int i=0; i<dim; i++) { Ntotal*=Nlist[i]; }
	Nco = 2*dim;
	Nco2 = Nco*Nco;
    Ntcut2halfsize = (Ntcut*(Ntcut+1))/2;
	Nm2 = Nm*Nm;
	Nm4 = Nm2*Nm2;
	dim2 = dim*dim;
    oneoverNtotal = 1.0/Ntotal;
    dt2 = dt*dt;
}




void fermions_2PI::fill_table_nn ()
{
	// Each lattice point has 2*dim nearest neighbors (nn).
	// Each nn has a different relative distance vector with the given lattice point, and we index each nn from 0 to Nco-1.
	// This function fills out a table which contains at a given nn index (table[nn]) the relative vector of the nn.
	// E.g.: n=0 --> (1,0,0); n=1 --> (-1,0,0); n=2 --> (0,1,0)...
	// table_nn_vectors is the same as table_nn_inds, but in double format
	
	table_nn_inds.resize(Nco);
	table_nn_vectors.resize(Nco);
	
	vector<int> indices(dim,0);
	vector<double> indices_double(dim,0.0);
	
	for (int n=0; n<Nco; n++)
	{
		get_nn_indices(indices,n);
		for (int i=0; i<dim; i++) indices_double[i] = indices[i];
		table_nn_inds[n] = indices;
		table_nn_vectors[n] = indices_double;
	}
}


void fermions_2PI::fill_table_spacemom ()
{
	// Each lattice point in spaca/momentum is defined by a vector of indices (x,y,z), depending on the dimension.
	// Each of this points is associated an index n(x,y,z) through get_spacemom_array_position (transforming it into a 1D array).
	// This function fills a table where at each position n we get the vector of indices (x,y,z) associated with lattice point n.
	// table_spacemom_vectors is the same as table_spacemom_inds, but rescaled to give the Fourier momentum 2*pi*(x/Nx,y/Ny,z/Nz)
	
	table_spacemom_vectors.resize(Ntotal);
	table_spacemom_inds.resize(Ntotal);
	vector<int> indices(dim,0);
	vector<double> vectors(dim,0.0);
	
	for (int n=0; n<Ntotal; n++)
	{
		get_spacemom_indices(indices,n);
		for (int i=0; i<dim; i++) vectors[i] = 2*M_PI*indices[i]/Nlist[i];
		table_spacemom_inds[n] = indices;
		table_spacemom_vectors[n] = vectors;
	}
}


void fermions_2PI::fill_Jk ()
{
	double phase, sum;
	vector<double> mom;
	
	for (int m=0; m<Nm; m++)
	{
		for (int n=0; n<Nm; n++)
		{
			for (int k=0; k<Ntotal; k++)
			{
				sum = 0;
				mom = (m-n)*Q+table_spacemom_vectors[k];
				for (int d=0; d<dim; d++) sum += 2*cos(mom[d]);
				
				Jk[ind_JkU(m,n,k)] = 4*J*sum;
				
				//cout << Jk[ind_JkU(m,n,k)] << endl;
				//cout << m << '\t' << n << '\t' << Q[0] << '\t' << table_spacemom_vectors[k][0] << endl;
				//cout << Jk[ind_JkU(m,n,k)] << endl;
			}
		}
	}
}


void fermions_2PI::fill_Jijinvk ()
{
	MatrixXd Jij(Ntotal,Ntotal);
	MatrixXd Jinv(Ntotal,Ntotal);
    comp * Jxarray = (comp *) fftw_malloc(sizeof(comp) * Ntotal);
    comp * Jkarray = (comp *) fftw_malloc(sizeof(comp) * Ntotal);
	vector<int> indices1(dim,0);
	vector<int> indices2(dim,0);
	int dist, temp;
	
	// Fill Jij matrix
	for (int n=0; n<Ntotal; n++)
	{
		for (int m=0; m<Ntotal; m++)
		{
			get_spacemom_indices(indices1,n);
			get_spacemom_indices(indices2,m);
			
			dist=0;
			for (int k=0; k<dim; k++)
			{
				temp = min( abs(indices1[k]-indices2[k]) , abs(indices1[k]-indices2[k]+Nlist[k]) );
				temp = min ( temp , abs(indices1[k]-indices2[k]-Nlist[k]) );
				dist += temp*temp;
			}
			
			if (dist==1) Jij(n,m)=1;
			else Jij(n,m)=0;
			
		}
	}
	
	// Invert Jij if invertible
	double Jdet = Jij.determinant();
	if (abs(Jdet)<0.001) invertibleJ = false;
	else invertibleJ = true;
	if (invertibleJ) { Jinv = Jij.inverse(); cout << "Matrix Jij has been inverted\n"; }
	else cout << "\nERROR/fill_Jijinvk: Matrix Jij not invertible\n";
	
	//cout << Jdet << endl;
	//if (invertibleJ) cout << "Invertible!\n";
	//else cout << "Singular...\n";
	//cout << Jinv << endl;
	
	// Fourier trafo
	int * NlistAsArray = new int [dim];  for (int i=0; i<dim; i++) NlistAsArray[i] = Nlist[i];
	fftw_plan fft_Jxk = fftw_plan_dft(dim, NlistAsArray, (fftw_complex *)Jxarray, (fftw_complex *)Jkarray, FFTW_FORWARD, FFTW_MEASURE);
	for (int n=0; n<Ntotal; n++)  Jxarray[n] = comp(Jinv(0,n),0); // Write in single array (should be already in row-major format)
	fftw_execute(fft_Jxk);
	
	// Keep only real part and check that imaginary part is zero.
	for (int n=0; n<Ntotal; n++)
	{
		Jijinvk[n] = Jkarray[n].real() / (4*J);
		if (abs(Jkarray[n].imag())>0.0000001) cout << "\nERROR/fill_Jijinvk: Problem with Fourier trafo of Jij. Imaginary part nonzero.\n.";
	}
	
	// Free memory
    fftw_free(Jxarray); fftw_free(Jkarray);
    fftw_destroy_plan(fft_Jxk);
}


void fermions_2PI::fill_effmass ()
{
	double phase, sum;
	vector<double> mom;
	
	for (int m=0; m<Nm; m++)
	{
		for (int n=0; n<Nm; n++)
		{
			sum = 0;
			mom = (m-n)*Q;
			for (int d=0; d<dim; d++) sum += 2*cos(mom[d]);
			
			effmass[ind_effmass(m,n)] = 4*J*sum;
			
			//cout << m << '\t' << n << '\t' << Jk[ind_JkU(m,n,0)] << '\t' << effmass[ind_effmass(m,n)] << endl;
			//cout << Jk[ind_JkU(m,n,k)] << endl;
			
		}
	}
	
}





// Row-order
void fermions_2PI::get_spacemom_indices (vector<int> & indices, int n)
{
	// For a given index n, returns the (space or momentum) lattice indices (x,y,z) of the corresponding lattice point
	
    int temp = 0;
    int rest = n;
    int block = Ntotal;
    
    while (temp<dim)
    {
        block = block/Nlist[temp];
        indices[temp] = rest/block;     //should be able to do this more efficiently
        rest -= indices[temp]*block;
        temp++;
    }
}



int fermions_2PI::get_spacemom_array_position (vector<int> indices)
{
	// For a given lattice vector (x,y,z) returns the index of the lattice point (row-major order)
	
    int pos = indices[0];
    for (int i=1; i<dim; i++) pos = pos*Nlist[i] + indices[i];
    
    return pos;
}





void fermions_2PI::get_nn_indices (vector<int> & indices, int n)
{
	// Returns relative space vector connecting a given array point with the nearest neighbor specified by n
	// E.g.: n=0 --> (1,0,0); n=1 --> (-1,0,0); n=2 --> (0,1,0)...
	
	for (int i=0; i<dim; i++) indices[i] = 0;
	
	int achse = n/2;
	indices[achse] = 1 - 2*( n - 2*achse );
}




int fermions_2PI::get_nn_array_position (vector<int> indices)
{
	// For a given nn vector returns the index/array position of that particular nn (inverse trafo of get_nn_indices)
	
    int pos = 0;
    for (int i=0; i<dim; i++) if (indices[i]!=0) pos += 2*i + (1-indices[i]) / 2;
    
    return pos;
}





vector<int> fermions_2PI::modN (vector<int> v)
{
	if (v.size() != dim) cout << "Wrong length of vector for modN.\n";
	vector<int> out(dim);
	for (int i=0; i<dim; i++) out[i] = ( v[i] % Nlist[i] + Nlist[i]) % Nlist[i];
	return out;
}



int fermions_2PI::kron_del (int a, int b)
{
	if (a==b) return 1;
	else return 0;
}


/*





vector<double> fermions_2PI::getPhysMom (int binNumber)
{
    if (binNumber>=Nbins) cout << "Wrong use of getPhysMom: n larger than Nbins.\n";
    
    vector<double> temp;
    
    temp.push_back(kBin[binNumber]);    // Physical momentum of bin n
    temp.push_back(momPerBin[binNumber]);   // Number of momenta in that bin
    
    return temp;
}


*/




/////////////////////////////////////////////////////////////////////////

///////////////            INITIAL CONDITIONS          //////////////////

/////////////////////////////////////////////////////////////////////////



// Set all containers to zero to start from the beginning
void fermions_2PI::setToZero ()
{
    for (int n=0; n < size_Frho; n++) { F[n]=0; rho[n]=0; selfF[n]=0; selfrho[n]=0; }
	
	for (int n=0; n < size_chi; n++) { chi[n]=0; }
	
    for (int n=0; n < size_D; n++) { DF[n]=0; Drho[n]=0; }
	
    for (int n=0; n < size_Pi; n++) { PiF[n]=0; Pirho[n]=0; }
}



void fermions_2PI::initialConditions ()
{
	// Initialize F
	vector<comp> amp(Nm,0.0);
	
	switch (IC)
	{
        case 0: // All in magnetic level 0: sz
            amp[0] = 1.0;
            break;
			
        case 1: // Superposition of magnetic levels 0 and 1: sx
			if (Nm<2) cout << "Nm is too small for IC chosen.\n";
            amp[0] = 1.0/sqrt(2.0);
			amp[1] = 1.0/sqrt(2.0);
            break;
			
        case 2: // Superposition of magnetic levels 0 and 1: sy
			if (Nm<2) cout << "Nm is too small for IC chosen.\n";
            amp[0] = 1.0/sqrt(2.0);
			amp[1] = -I/sqrt(2.0);
            break;
			
        case 3: // Tilted superposition of magnetic levels 0 and 1
			if (Nm<2) cout << "Nm is too small for IC chosen.\n";
            amp[0] = sqrt(3.0)/2.0;
			amp[1] = 1.0/2.0;
            break;
			
        case 31: // Symmetric 3-level state
			if (Nm<3) cout << "Nm is too small for IC chosen.\n";
            amp[0] = 1.0/sqrt(3.0);
			amp[1] = 1.0/sqrt(3.0);
			amp[2] = 1.0/sqrt(3.0);
            break;
			
        case 32: // Asymmetric 3-levels state
			if (Nm<3) cout << "Nm is too small for IC chosen.\n";
            amp[0] = 1.0/sqrt(3.0);
			amp[1] = 1.0/sqrt(3.0);
			amp[2] = -1.0/sqrt(3.0);
            break;
			
        case 33: // Random 3-levels state
			if (Nm<3) cout << "Nm is too small for IC chosen.\n";
            amp[0] = 1.0/sqrt(6.0);
			amp[1] = sqrt(2.0/6.0);
			amp[2] = -sqrt(3.0/6.0);
            break;
			
        case 99: // Superposition of magnetic levels 0 and 1: sy
			if (Nm<3) cout << "Nm is too small for IC chosen.\n";
            amp[0] = sqrt(3.0)/sqrt(5.0);
			amp[1] = -I/sqrt(5.0);
			amp[2] = 1.0/sqrt(5.0);
            break;
		
        default:
            cout << "Invalid type of IC chosen.\n";
            break;
	}
	
	
    // Initialize F and rho with vector amp and equal-time anticommutation relations
	for (int m=0; m<Nm; m++)
	{
		for (int n=0; n<Nm; n++)
		{	
			if (m==n)
			{
				F[ind_fermion(m,n,0,0)] = 0.5 - sqrabs(amp[m]);
				rho[ind_fermion(m,n,0,0)] = I;
			}
			else
			{
				F[ind_fermion(m,n,0,0)] = - conj(amp[m])*amp[n];
				rho[ind_fermion(m,n,0,0)] = 0;
			}
		}
	}
    
    // Fill self-energies with values of initial conditions
    compute_selfenergies(0);
	
	// Set current time to zero
	t=0;
    
}




    
    
    
    
    
    
    
    
    /////////////////////////////////////////////////////////////////////////
    
    ///////////////////            DYNAMICS          ////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    


void fermions_2PI::dynamics ()
{
    switch (difmethod) {
            
        case 0: // Euler
			dynamics_euler();
			break;
			
		case 11: // P(EC)^k
		case 12:
		case 13:
		case 14:
		case 15:
		case 16:
		case 17:
		case 18:
		case 19:
			dynamics_PEC();
			break;
		
		case 21: // P(EC)^kE
		case 22:
		case 23:
		case 24:
		case 25:
		case 26:
		case 27:
		case 28:
		case 29:
			dynamics_PECE();
			break;
            
        default:
            cout << "Invalid type of difmethod chosen.\n";
            break;
    }
    
}




void fermions_2PI::dynamics_euler ()
{
	// Compute right-hand-side
	compute_rhs_F(t,rhs_F);
	compute_rhs_rho(t,rhs_rho);
	
	// Compute F, rho at t+1 from Euler step
	update_euler();
	
	// Advance in time t -> t+1
	t++;
	
	// Compute self-energies
	compute_selfenergies (t);
	
	// Shift arrays if time has exhausted memory allocated
    if (t>=Ntcut-1) { t=t-1; shift_arrays(); }
	
	
	if (t==3)
	{
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						for (int k=0; k<Ntotal; k++)
						{
							int t1 = t;
							
							for (int t2=0; t2<=t1; t2++)
							{
								int ind1 = ind_aux2U_D(m1,n1,m2,n2,t1,t2,k);
								int ind2 = ind_aux2U_D(m2,n2,m1,n1,t1,t2, get_spacemom_array_position( modN(-table_spacemom_inds[k]) ) );
								
								cout << (DF[ind1]-conj(DF[ind2])).real() << '\t' << (DF[ind1]-conj(DF[ind2])).imag() << endl;
							}
							
							
							
						}
					}
				}
			}
		}
		
	}
	
}



void fermions_2PI::dynamics_PEC ()
{
	// Compute right-hand-side at t=0, since it has not been computed yet
	if (t==0)
	{
		compute_rhs_F(t,rhs_F);
		compute_rhs_rho(t,rhs_rho);
	}
	
	// Predict F, rho at t+1 from Euler step (assumes RHS has been saved before)
	update_euler();
	
	// Evaluate-Correct loop
	for (int k=0; k<ECiter; k++)
	{
		// Compute self-energies at t+1
		compute_selfenergies (t+1);
		
		// Compute right-hand-side at t+1
		compute_rhs_F(t+1,rhs_F_plus1);
		compute_rhs_rho(t+1,rhs_rho_plus1);
		
		// Correct F, rho at t+1
		update_trapezoidal();
	}
	
	// Compute self-energies at t+1
	compute_selfenergies (t+1);
	
	// Save RHS for next step
	save_rhs();
	
	// Advance in time t -> t+1
	t++;
	
	// Shift arrays if time has exhausted memory allocated
    if (t>=Ntcut-1) { t=t-1; shift_arrays(); }
}
	
	
	
void fermions_2PI::dynamics_PECE ()
{
	// Compute right-hand-side
	compute_rhs_F(t,rhs_F);
	compute_rhs_rho(t,rhs_rho);
	
	// Predict F, rho at t+1 from Euler step
	update_euler();
	
	// Evaluate-Correct loop
	for (int k=0; k<ECiter; k++)
	{
		// Compute self-energies at t+1
		compute_selfenergies (t+1);
		
		// Compute right-hand-side at t+1
		compute_rhs_F(t+1,rhs_F_plus1);
		compute_rhs_rho(t+1,rhs_rho_plus1);
		
		// Correct F, rho at t+1
		update_trapezoidal();
	}
	
	// Compute self-energies at t+1 with final F, rho values
	compute_selfenergies (t+1);
	
	// Advance in time t -> t+1
	t++;
	
	// Shift arrays if time has exhausted memory allocated
    if (t>=Ntcut-1) { t=t-1; shift_arrays(); }
}






void fermions_2PI::update_euler ()
{
	// Compute t+1
	for (int t2=0; t2<=t; t2++)
	{
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{
				F[ind_fermion(m,n,t+1,t2)] = F[ind_fermion(m,n,t,t2)] + dt * rhs_F[ind_rhs(m,n,t,t2)];
				rho[ind_fermion(m,n,t+1,t2)] = rho[ind_fermion(m,n,t,t2)] + dt * rhs_rho[ind_rhs(m,n,t,t2)];
			}
		}
	}
	
	// Diagonal elements
	for (int m=0; m<Nm; m++)
	{
		for (int n=0; n<Nm; n++)
		{
			F[ind_fermion(m,n,t+1,t+1)] = F[ind_fermion(m,n,t,t)] + dt * ( rhs_F[ind_rhs(m,n,t,t)] + conj(rhs_F[ind_rhs(n,m,t,t)]) );
			if (m==n) rho[ind_fermion(m,n,t+1,t+1)] = I;
			else rho[ind_fermion(m,n,t+1,t+1)] = 0;
		}
	}
}


void fermions_2PI::update_trapezoidal ()
{
	// Compute t+1
	for (int t2=0; t2<=t; t2++)
	{
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{
				F[ind_fermion(m,n,t+1,t2)] = F[ind_fermion(m,n,t,t2)] + dt/2.0 * ( rhs_F[ind_rhs(m,n,t,t2)] + rhs_F_plus1[ind_rhs(m,n,t+1,t2)] );
				rho[ind_fermion(m,n,t+1,t2)] = rho[ind_fermion(m,n,t,t2)] + dt/2.0 * ( rhs_rho[ind_rhs(m,n,t,t2)] + rhs_rho_plus1[ind_rhs(m,n,t+1,t2)] );
			}
		}
	}
	
	// Diagonal elements
	for (int m=0; m<Nm; m++)
	{
		for (int n=0; n<Nm; n++)
		{
			F[ind_fermion(m,n,t+1,t+1)] = F[ind_fermion(m,n,t,t)] + dt/2.0 * ( rhs_F[ind_rhs(m,n,t,t)] + conj(rhs_F[ind_rhs(n,m,t,t)])
					 														   + rhs_F_plus1[ind_rhs(m,n,t+1,t+1)] + conj(rhs_F_plus1[ind_rhs(n,m,t+1,t+1)]) );
			if (m==n) rho[ind_fermion(m,n,t+1,t+1)] = I;
			else rho[ind_fermion(m,n,t+1,t+1)] = 0;
		}
	}
	
}





void fermions_2PI::save_rhs ()
{
	for (int k=0; k < Nm2 * Ntcut; k++)
	{
		rhs_F[k] = rhs_F_plus1[k];
		rhs_rho[k] = rhs_rho_plus1[k];
	}
}




void fermions_2PI::compute_rhs_F (int t1, comp * rhs_array)
{
	#pragma omp parallel for
	for (int t2=0; t2<=t1; t2++)
	{
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{
				double phase;
				comp z, mass, mem_F1, mem_F2;
				int ind_r1, ind_r2;
				int ind_w = ind_rhs(m,n,t1,t2);
				
				
				// Mass term
				mass = 0;
				for (int a=0; a<Nm; a++)
				{	
					mass += - effmass[ind_effmass(m,a)] * F[ind_fermion(m,a,t1,t1)] * F[ind_fermion(a,n,t1,t2)];
					
					// With equal-time correction
					//mass += - ( effmass[ind_effmass(m,a)] * F[ind_fermion(m,a,t1,t1)] - 0.5 * kron_del(m,a) ) * F[ind_fermion(a,n,t1,t2)];
				}
				
				/*
				mass = 0;
				for (int a=0; a<Nm; a++)
				{
					for (int s=0; s<Nco; s++)
					{
						phase = (m-a) * ( Q * table_nn_vectors[s] );
						z = comp(cos(phase),-sin(phase));

						mass += z * F[ind_fermion(m,a,t1,t1)] * F[ind_fermion(a,n,t1,t2)];
					}
				}
				mass *= - 4 * J;*/
				
				
				//cout << (mass-mass2).real() << '\t' << (mass-mass2).imag() << endl;
				//cout << (mass-mass2).real() << endl;
				//mass = mass2;
				
				
				
				
				// Memory integral 1
				mem_F1 = 0;
				for (int a=0; a<Nm; a++)
				{
					if (t1>0)
					{
						ind_r1 = ind_fermion(m,a,t1,0);
						ind_r2 = ind_fermion(n,a,t2,0);
						mem_F1 += 0.5 * selfrho[ind_r1] * conj(F[ind_r2]);
					
						ind_r1 = ind_fermion(m,a,t1,t1);
						ind_r2 = ind_fermion(a,n,t1,t2);
						mem_F1 += 0.5 * selfrho[ind_r1] * F[ind_r2];
					
						for (int l=1; l<=t2-1; l++)
						{
							ind_r1 = ind_fermion(m,a,t1,l);
							ind_r2 = ind_fermion(n,a,t2,l);
							mem_F1 += selfrho[ind_r1] * conj(F[ind_r2]);
						}
						for (int l=max(t2,1); l<=t1-1; l++)
						{
							ind_r1 = ind_fermion(m,a,t1,l);
							ind_r2 = ind_fermion(a,n,l,t2);
							mem_F1 += selfrho[ind_r1] * F[ind_r2];
						}
					}
				}
				mem_F1 *= dt;
				
				//if (t1==0) cout << mem_F1 << endl;
				
				// Memory integral 2
				mem_F2 = 0;
				for (int a=0; a<Nm; a++)
				{
					if (t2>0)
					{
						ind_r1 = ind_fermion(m,a,t1,0);
						ind_r2 = ind_fermion(n,a,t2,0);
						mem_F2 += 0.5 * selfF[ind_r1] * conj(rho[ind_r2]);
					
						ind_r1 = ind_fermion(m,a,t1,t2);
						ind_r2 = ind_fermion(a,n,t2,t2);
						mem_F2 += -0.5 * selfF[ind_r1] * rho[ind_r2];
					
						for (int l=1; l<=t2-1; l++)
						{
							ind_r1 = ind_fermion(m,a,t1,l);
							ind_r2 = ind_fermion(n,a,t2,l);
							mem_F2 += selfF[ind_r1] * conj(rho[ind_r2]);
						}
					}
				}
				mem_F2 *= dt;
				
				//if (t2==0) cout << mem_F2 << endl;
				
				// Putting all together
				rhs_array[ind_w] = -I * ( mass + mem_F1 + mem_F2 );
				
			}
		}
	}
	
}


void fermions_2PI::compute_rhs_rho (int t1, comp * rhs_array)
{
	#pragma omp parallel for
	for (int t2=0; t2<=t1; t2++)
	{
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{
				double phase;
				comp z, mass, mem_rho;
				int ind_r1, ind_r2;
				int ind_w = ind_rhs(m,n,t1,t2);
				
				// Mass term
				mass = 0;
				for (int a=0; a<Nm; a++)
				{	
					mass += - effmass[ind_effmass(m,a)] * F[ind_fermion(m,a,t1,t1)] * rho[ind_fermion(a,n,t1,t2)];
					
					// With equal-time correction
					//mass += - ( effmass[ind_effmass(m,a)] * F[ind_fermion(m,a,t1,t1)] - 0.5 * kron_del(m,a) ) * rho[ind_fermion(a,n,t1,t2)];
				}
				
				/*
				mass = 0;
				for (int a=0; a<Nm; a++)
				{
					for (int s=0; s<Nco; s++)
					{
						phase = (m-a) * ( Q * table_nn_vectors[s] );
						z = comp(cos(phase),-sin(phase));
						
						mass += z * F[ind_fermion(m,a,t1,t1)] * rho[ind_fermion(a,n,t1,t2)];
					}
				}
				mass *= - 4 * J;
				*/
				
				// Memory integral
				mem_rho = 0;
				for (int a=0; a<Nm; a++)
				{
					if (t1>t2)
					{
						ind_r1 = ind_fermion(m,a,t1,t1);
						ind_r2 = ind_fermion(a,n,t1,t2);
						mem_rho += 0.5 * selfrho[ind_r1] * rho[ind_r2];
					
						ind_r1 = ind_fermion(m,a,t1,t2);
						ind_r2 = ind_fermion(a,n,t2,t2);
						mem_rho += 0.5 * selfrho[ind_r1] * rho[ind_r2];
					
						for (int l=t2+1; l<=t1-1; l++)
						{
							ind_r1 = ind_fermion(m,a,t1,l);
							ind_r2 = ind_fermion(a,n,l,t2);
							mem_rho += selfrho[ind_r1] * rho[ind_r2];
						}
					}
				}
				mem_rho *= dt;
				
				//if (t1==t2) cout << mem_rho << endl;
				
				// Putting all together
				rhs_array[ind_w] = -I * ( mass + mem_rho );
				
			}
		}
	}
}











void fermions_2PI::compute_selfenergies (int t1)
{
    switch (channel) {
        case 0: // S-channel
			if (approx==1) // NLO
			{
				compute_selfenergies_S_channel(t1);			
			}
            break;
			
        case 1: // U-channel
			if (approx==1) // NLO
			{
				compute_selfenergies_U_channel(t1);			
			}
            break;
			
        default:
            cout << "Wrong type of channel chosen\n";
            break;
    }
}



// Riemann integral
/*
void fermions_2PI::compute_selfenergies_S_channel (int t1)
{
	//clock_t cpu_time = clock();
	
	// Auxiliary self-energy: PiF, Pirho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int s=0; s<Nco; s++)
		{
			double phase;
			comp z;
			
			int ind_w = ind_aux2S(s,t1,t2);
			int ind_r;
			
			PiF[ind_w] = 0;
			Pirho[ind_w] = 0;	// Just in case
			
			for (int m1=0; m1<Nm; m1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					phase = (m1-m2) * Q * table_nn_vectors[s];
					z = comp(cos(phase),-sin(phase));
					
					ind_r = ind_fermion(m1,m2,t1,t2);
					
					PiF[ind_w] += z * ( F[ind_r] * conj(F[ind_r]) - 0.25 * rho[ind_r] * conj(rho[ind_r]) );
					Pirho[ind_w] += z * ( rho[ind_r] * conj(F[ind_r]) + F[ind_r] * conj(rho[ind_r]) );
				}
			}

		}
	}
	
	
	
	
	
	// Auxiliary propagator: Drho, DF
	#pragma omp parallel for
    for (int t2=0; t2<t1; t2++)
    {
		// Drho
		for (int s=0; s<Nco; s++)
		{
			comp mem_rho;
			int ind_r0, ind_r1, ind_r2;
			int ind_w = ind_aux2S(s,t1,t2);
			
			// Mass term
			ind_r0 = ind_w;
			
			// Memory integral
			mem_rho = 0;
			
			if (t1>t2)
			{
				for (int l=t2; l<=t1-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(s,l,t2);
					mem_rho += Pirho[ind_r1] * Drho[ind_r2];
				}
			}
				
			mem_rho *= - 4 * J * dt;
			
			// All together
			Drho[ind_w] = - Pirho[ind_r0] + mem_rho;
		}
		
		
		
		
		// DF
		for (int s=0; s<Nco; s++)
		{
			comp mem_F1, mem_F2;
			int ind_r0, ind_r1, ind_r2;
			int ind_w = ind_aux2S(s,t1,t2);
			int minus_s = get_nn_array_position(-table_nn_inds[s]);
			
			// Mass term
			ind_r0 = ind_w;
			
			// Memory integral 1
			mem_F1 = 0;
			
			if (t1>0)
			{
				for (int l=0; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
				for (int l=t2; l<=t1-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(s,l,t2);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
			}

			mem_F1 *= - 4 * J * dt;
			
			//if (t1==0) cout << mem_F1 << endl;
			
			// Memory integral 2
			mem_F2 = 0;
				
			if (t2>0)
			{
				for (int l=0; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F2 += PiF[ind_r1] * Drho[ind_r2];
				}
			}
				
			mem_F2 *= - 4 * J * dt;
			
			// All together
			DF[ind_w] = - PiF[ind_r0] + mem_F1 + mem_F2;
		}
		
	}
	
	
	
	
	// Diagonal DF, Drho
    {
		int t2 = t1;
		
		// Drho
		for (int s=0; s<Nco; s++)
		{
			int ind_w = ind_aux2S(s,t1,t2);
			int ind_r0 = ind_w;
			
			// All together
			Drho[ind_w] = - Pirho[ind_r0];
		}

		
		// DF
		for (int s=0; s<Nco; s++)
		{
			comp mem_F1, mem_F2;
			int ind_r0, ind_r1, ind_r2;
			int ind_w = ind_aux2S(s,t1,t2);
			int minus_s = get_nn_array_position(-table_nn_inds[s]);
			
			// Mass term
			ind_r0 = ind_w;
			
			// Memory integral 1
			mem_F1 = 0;
			
			if (t1>0)
			{
				for (int l=0; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
				for (int l=t2; l<=t1-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(s,l,t2);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
			}

			mem_F1 *= - 4 * J * dt;
			
			//if (t1==0) cout << mem_F1 << endl;
			
			// Memory integral 2
			mem_F2 = 0;
				
			if (t2>0)
			{
				for (int l=0; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F2 += PiF[ind_r1] * Drho[ind_r2];
				}
			}
				
			mem_F2 *= - 4 * J * dt;
			
			// All together
			DF[ind_w] = - PiF[ind_r0] + mem_F1 + mem_F2;
		}
		
	}
	
	
	
	// Fermion self-energy: selfF, selfrho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{
				double phase;
				comp z;
				
				int ind_w = ind_fermion(m,n,t1,t2);
				int ind_r1, ind_r2;
				
				selfF[ind_w] = 0;
				selfrho[ind_w] = 0;	// Just in case
				
				for (int s=0; s<Nco; s++)
				{
					phase = (m - n) * Q * table_nn_vectors[s];
					z = comp(cos(phase),-sin(phase));
					
					ind_r1 = ind_fermion(m,n,t1,t2);
					ind_r2 = ind_aux2S(s,t1,t2);
						
					selfF[ind_w] += z * ( F[ind_r1] * conj(DF[ind_r2]) - 0.25 * rho[ind_r1] * conj(Drho[ind_r2]) );
					selfrho[ind_w] += z * ( rho[ind_r1] * conj(DF[ind_r2]) + F[ind_r1] * conj(Drho[ind_r2]) );
				}
				
				selfF[ind_w] *= - 16 * J * J;
				selfrho[ind_w] *= - 16 * J * J;
					
			}
		}
	}
	
	
    //cpu_time = clock()-cpu_time;
    //cout << ((float)cpu_time)/CLOCKS_PER_SEC << endl;
}
*/


// Trapezoidal rule
void fermions_2PI::compute_selfenergies_S_channel (int t1)
{
	//clock_t cpu_time = clock();
	
	// Auxiliary self-energy: PiF, Pirho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int s=0; s<Nco; s++)
		{
			double phase;
			comp z;
			
			int ind_w = ind_aux2S(s,t1,t2);
			int ind_r;
			
			PiF[ind_w] = 0;
			Pirho[ind_w] = 0;	// Just in case
			
			for (int m1=0; m1<Nm; m1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					phase = (m1-m2) * Q * table_nn_vectors[s];
					z = comp(cos(phase),-sin(phase));
					
					ind_r = ind_fermion(m1,m2,t1,t2);
					
					PiF[ind_w] += z * ( F[ind_r] * conj(F[ind_r]) - 0.25 * rho[ind_r] * conj(rho[ind_r]) );
					Pirho[ind_w] += z * ( rho[ind_r] * conj(F[ind_r]) + F[ind_r] * conj(rho[ind_r]) );
				}
			}

		}
	}
	
	
	
	
	
	// Auxiliary propagator: Drho, DF
	#pragma omp parallel for
    for (int t2=0; t2<t1; t2++)
    {
		// Drho
		for (int s=0; s<Nco; s++)
		{
			comp mem_rho;
			int ind_r0, ind_r1, ind_r2;
			int ind_w = ind_aux2S(s,t1,t2);
			
			// Mass term
			ind_r0 = ind_w;
			
			// Memory integral
			mem_rho = 0;
			
			if (t1>t2)
			{
				ind_r1 = ind_aux2S(s,t1,t2);
				ind_r2 = ind_aux2S(s,t2,t2);
				mem_rho += 0.5 * Pirho[ind_r1] * Drho[ind_r2];
				
				for (int l=t2+1; l<=t1-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(s,l,t2);
					mem_rho += Pirho[ind_r1] * Drho[ind_r2];
				}
			}
				
			mem_rho *= - 4 * J * dt;
			
			// All together
			if (t1>t2)
			{
				Drho[ind_w] = ( - Pirho[ind_r0] + mem_rho ) / ( 1.0 + 2 * J * dt * Pirho[ind_aux2S(s,t1,t1)] );
			}
			else
			{
				Drho[ind_w] = - Pirho[ind_r0] + mem_rho;
			}
			
		}
		
		
		
		
		// DF
		for (int s=0; s<Nco; s++)
		{
			comp mem_F1, mem_F2;
			int ind_r0, ind_r1, ind_r2;
			int ind_w = ind_aux2S(s,t1,t2);
			int minus_s = get_nn_array_position(-table_nn_inds[s]);
			
			// Mass term
			ind_r0 = ind_w;
			
			// Memory integral 1
			mem_F1 = 0;
			
			if (t1>0)
			{
				ind_r1 = ind_aux2S(s,t1,0);
				ind_r2 = ind_aux2S(minus_s,t2,0);
				mem_F1 += 0.5 * Pirho[ind_r1] * DF[ind_r2];
				
				for (int l=1; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
				for (int l=max(t2,1); l<=t1-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(s,l,t2);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
			}

			mem_F1 *= - 4 * J * dt;
			
			//if (t1==0) cout << mem_F1 << endl;
			
			// Memory integral 2
			mem_F2 = 0;
				
			if (t2>0)
			{
				ind_r1 = ind_aux2S(s,t1,0);
				ind_r2 = ind_aux2S(minus_s,t2,0);
				mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
				
				ind_r1 = ind_aux2S(s,t1,t2);
				ind_r2 = ind_aux2S(minus_s,t2,t2);
				mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
				
				for (int l=1; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F2 += PiF[ind_r1] * Drho[ind_r2];
				}
			}
				
			mem_F2 *= - 4 * J * dt;
			
			// All together
			if (t1>0)
			{
				DF[ind_w] = ( - PiF[ind_r0] + mem_F1 + mem_F2 ) / ( 1.0 + 2 * J * dt * Pirho[ind_aux2S(s,t1,t1)] );
			}
			else
			{
				DF[ind_w] = - PiF[ind_r0] + mem_F1 + mem_F2;
			}
			
		}
		
	}
	
	
	
	
	// Diagonal DF, Drho
    {
		int t2 = t1;
		
		// Drho
		for (int s=0; s<Nco; s++)
		{
			int ind_w = ind_aux2S(s,t1,t2);
			int ind_r0 = ind_w;
			
			// All together
			Drho[ind_w] = - Pirho[ind_r0];
		}

		
		// DF
		for (int s=0; s<Nco; s++)
		{
			comp mem_F1, mem_F2;
			int ind_r0, ind_r1, ind_r2;
			int ind_w = ind_aux2S(s,t1,t2);
			int minus_s = get_nn_array_position(-table_nn_inds[s]);
			
			// Mass term
			ind_r0 = ind_w;
			
			// Memory integral 1
			mem_F1 = 0;
			
			if (t1>0)
			{
				ind_r1 = ind_aux2S(s,t1,0);
				ind_r2 = ind_aux2S(minus_s,t2,0);
				mem_F1 += 0.5 * Pirho[ind_r1] * DF[ind_r2];
				
				for (int l=1; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
				for (int l=max(t2,1); l<=t1-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(s,l,t2);
					mem_F1 += Pirho[ind_r1] * DF[ind_r2];
				}
			}

			mem_F1 *= - 4 * J * dt;
			
			//if (t1==0) cout << mem_F1 << endl;
			
			// Memory integral 2
			mem_F2 = 0;
			
			if (t2>0)
			{
				ind_r1 = ind_aux2S(s,t1,0);
				ind_r2 = ind_aux2S(minus_s,t2,0);
				mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
				
				ind_r1 = ind_aux2S(s,t1,t2);
				ind_r2 = ind_aux2S(minus_s,t2,t2);
				mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
				
				for (int l=1; l<=t2-1; l++)
				{
					ind_r1 = ind_aux2S(s,t1,l);
					ind_r2 = ind_aux2S(minus_s,t2,l);
					mem_F2 += PiF[ind_r1] * Drho[ind_r2];
				}
			}
				
			mem_F2 *= - 4 * J * dt;
			
			// All together
			if (t1>0)
			{
				DF[ind_w] = ( - PiF[ind_r0] + mem_F1 + mem_F2 ) / ( 1.0 + 2 * J * dt * Pirho[ind_aux2S(s,t1,t1)] );
			}
			else
			{
				DF[ind_w] = - PiF[ind_r0] + mem_F1 + mem_F2;
			}
		}
		
	}
	
	
	
	// Fermion self-energy: selfF, selfrho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{
				double phase;
				comp z;
				
				int ind_w = ind_fermion(m,n,t1,t2);
				int ind_r1, ind_r2;
				
				selfF[ind_w] = 0;
				selfrho[ind_w] = 0;	// Just in case
				
				for (int s=0; s<Nco; s++)
				{
					phase = (m - n) * Q * table_nn_vectors[s];
					z = comp(cos(phase),-sin(phase));
					
					ind_r1 = ind_fermion(m,n,t1,t2);
					ind_r2 = ind_aux2S(s,t1,t2);
						
					selfF[ind_w] += z * ( F[ind_r1] * conj(DF[ind_r2]) - 0.25 * rho[ind_r1] * conj(Drho[ind_r2]) );
					selfrho[ind_w] += z * ( rho[ind_r1] * conj(DF[ind_r2]) + F[ind_r1] * conj(Drho[ind_r2]) );
				}
				
				selfF[ind_w] *= - 16 * J * J;
				selfrho[ind_w] *= - 16 * J * J;
					
			}
		}
	}
	
	
    //cpu_time = clock()-cpu_time;
    //cout << ((float)cpu_time)/CLOCKS_PER_SEC << endl;
}


/*
// Riemann integral
void fermions_2PI::compute_selfenergies_U_channel (int t1)
{
	//clock_t cpu_time = clock();
	
	// Auxiliary self-energy: PiF, Pirho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						int ind_w = ind_aux2U_Pi(m1,n1,m2,n2,t1,t2);
						int ind_r1 = ind_fermion(n1,m2,t1,t2);
						int ind_r2 = ind_fermion(m1,n2,t1,t2);
						
						PiF[ind_w] = 0;
						Pirho[ind_w] = 0;	// Just in case
						
						PiF[ind_w] += F[ind_r1] * conj(F[ind_r2]) - 0.25 * rho[ind_r1] * conj(rho[ind_r2]);
						Pirho[ind_w] += rho[ind_r1] * conj(F[ind_r2]) + F[ind_r1] * conj(rho[ind_r2]);
					}
				}
			}
		}
	}
	
	
	
	
	// Auxiliary propagator: Drho, DF
	#pragma omp parallel for
    for (int t2=0; t2<t1; t2++)
    {
		// Drho
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						for (int k=0; k<Ntotal; k++)
						{
							comp mem_rho, mass_rho;
							int ind_r0, ind_r1, ind_r2;
							int ind_J0, ind_J1, ind_J2;
							int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_rho = - Jk[ind_J0] * Jk[ind_J1] * Pirho[ind_r0];
							
							// Memory integral
							mem_rho = 0;
							ind_J2 = ind_JkU(n1,m1,k);
			
							if (t1>t2)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										for (int l=t2; l<=t1-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m3,n3,m2,n2,l,t2,k);
											mem_rho += Pirho[ind_r1] * Drho[ind_r2];
										}
									}
								}
							}
				
							mem_rho *= Jk[ind_J2] * dt;		// Maybe can be improved by omitting terms where Jk=0.
							
							// All together
							Drho[ind_w] = mass_rho + mem_rho;
						}
					}
				}
			}
		}
		
		

		
		// DF (maybe can be combined with Drho)
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						for (int k=0; k<Ntotal; k++)
						{
							comp mem_F1, mem_F2, mass_F;
							int ind_r0, ind_r1, ind_r2;
							int ind_J0, ind_J1, ind_J2;
							int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
							int minus_k = get_spacemom_array_position( modN(-table_spacemom_inds[k]) );
							
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_F = - Jk[ind_J0] * Jk[ind_J1] * PiF[ind_r0];
							
							
							// Memory integral 1
							mem_F1 = 0;
							ind_J2 = ind_JkU(n1,m1,k);
			
							if (t1>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										for (int l=0; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
										for (int l=t2; l<=t1-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m3,n3,m2,n2,l,t2,k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
									}
								}
							}

							mem_F1 *= Jk[ind_J2] * dt;
							
							
							// Memory integral 2
							mem_F2 = 0;
				
							if (t2>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										for (int l=0; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F2 += PiF[ind_r1] * Drho[ind_r2];
										}
									}
								}
							}
				
							mem_F2 *= Jk[ind_J2] * dt;
							
							// All together
							DF[ind_w] = mass_F + mem_F1 + mem_F2;
						}
					}
				}
			}
		}
		
	}
	
	
	
	
	// Diagonal DF, Drho
    {
		int t2 = t1;
		
		// Drho (same as before but no integral)
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						for (int k=0; k<Ntotal; k++)
						{
							comp mass_rho;
							int ind_r0;
							int ind_J0, ind_J1;
							int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_rho = - Jk[ind_J0] * Jk[ind_J1] * Pirho[ind_r0];
							
							// All together
							Drho[ind_w] = mass_rho;
						}
					}
				}
			}
		}
		
		
		
		// DF (same as before)
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						for (int k=0; k<Ntotal; k++)
						{
							comp mem_F1, mem_F2, mass_F;
							int ind_r0, ind_r1, ind_r2;
							int ind_J0, ind_J1, ind_J2;
							int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
							int minus_k = get_spacemom_array_position( modN(-table_spacemom_inds[k]) );
							
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_F = - Jk[ind_J0] * Jk[ind_J1] * PiF[ind_r0];
							
							
							// Memory integral 1
							mem_F1 = 0;
							ind_J2 = ind_JkU(n1,m1,k);
			
							if (t1>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										for (int l=0; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
										for (int l=t2; l<=t1-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m3,n3,m2,n2,l,t2,k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
									}
								}
							}

							mem_F1 *= Jk[ind_J2] * dt;
							
							
							// Memory integral 2
							mem_F2 = 0;
				
							if (t2>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										for (int l=0; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F2 += PiF[ind_r1] * Drho[ind_r2];
										}
									}
								}
							}
				
							mem_F2 *= Jk[ind_J2] * dt;
							
							// All together
							DF[ind_w] = mass_F + mem_F1 + mem_F2;
						}
					}
				}
			}
		}
		
	}
	


	
	// Fermion self-energy: selfF, selfrho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{	
				int ind_w = ind_fermion(m,n,t1,t2);
				int ind_r1, ind_r2;
				comp DFtemp, Drhotemp;
				
				selfF[ind_w] = 0;
				selfrho[ind_w] = 0;	// Just in case
				
				for (int a=0; a<Nm; a++)
				{
					for (int b=0; b<Nm; b++)
					{
						DFtemp = 0;
						Drhotemp = 0;
						ind_r1 = ind_fermion(a,b,t1,t2);
						
						for (int k=0; k<Ntotal; k++) // Fourier trafo of x=0
						{
							ind_r2 = ind_aux2U_D(m,a,b,n,t1,t2,k);
							DFtemp += DF[ind_r2];
							Drhotemp += Drho[ind_r2];
						}
						
						DFtemp *= oneoverNtotal;
						Drhotemp *= oneoverNtotal;
						
						selfF[ind_w] += - ( F[ind_r1] * DFtemp - 0.25 * rho[ind_r1] * Drhotemp );
						selfrho[ind_w] += - ( rho[ind_r1] * DFtemp + F[ind_r1] * Drhotemp );
					}
				}
					
			}
		}
	}
	
	
    //cpu_time = clock()-cpu_time;
    //cout << ((float)cpu_time)/CLOCKS_PER_SEC << endl;
}
*/


// Trapezoidal rule
void fermions_2PI::compute_selfenergies_U_channel (int t1)
{
	//clock_t cpu_time = clock();
	
	double sign = 1.0;
	
	// Auxiliary self-energy: PiF, Pirho
	comp * PiF_temp = new comp [ size_Pi ];
	comp * Pirho_temp = new comp [ size_Pi ];
	
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{
						int ind_w = ind_aux2U_Pi(m1,n1,m2,n2,t1,t2);
						int ind_r1 = ind_fermion(n1,m2,t1,t2);
						int ind_r2 = ind_fermion(m1,n2,t1,t2);
						
						//PiF[ind_w] = 0;
						//Pirho[ind_w] = 0;	// Just in case
						
						//PiF[ind_w] += sign * ( F[ind_r1] * conj(F[ind_r2]) - 0.25 * rho[ind_r1] * conj(rho[ind_r2]) );
						//Pirho[ind_w] += sign * ( rho[ind_r1] * conj(F[ind_r2]) + F[ind_r1] * conj(rho[ind_r2]) );
						
						PiF_temp[ind_w] = 0;
						Pirho_temp[ind_w] = 0;	// Just in case
						
						PiF_temp[ind_w] += sign * ( F[ind_r1] * conj(F[ind_r2]) - 0.25 * rho[ind_r1] * conj(rho[ind_r2]) );
						Pirho_temp[ind_w] += sign * ( rho[ind_r1] * conj(F[ind_r2]) + F[ind_r1] * conj(rho[ind_r2]) );
						
					}
				}
			}
		}
	}
	
	/*
	// Correct for extra chi
	#pragma omp parallel for
	for (int t2=0; t2<=t1; t2++)
	{
		for (int m1=0; m1<Nm; m1++)
		{
			for (int n1=0; n1<Nm; n1++)
			{
				for (int m2=0; m2<Nm; m2++)
				{
					for (int n2=0; n2<Nm; n2++)
					{	
						int inv[2];
						inv[0] = 1;
						inv[1] = 0;
						
						if (m1==n1 && m2!=n2)
						{
							PiF[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = (PiF_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] - PiF_temp[ind_aux2U_Pi(inv[m1],inv[n1],m2,n2,t1,t2)])/sqrt(2.0);
							Pirho[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = (Pirho_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] - Pirho_temp[ind_aux2U_Pi(inv[m1],inv[n1],m2,n2,t1,t2)])/sqrt(2.0);
						}
						if (m2==n2 && m1!=n1)
						{
							PiF[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = (PiF_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] - PiF_temp[ind_aux2U_Pi(m1,n1,inv[m2],inv[n2],t1,t2)])/sqrt(2.0);
							Pirho[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = (Pirho_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] - Pirho_temp[ind_aux2U_Pi(m1,n1,inv[m2],inv[n2],t1,t2)])/sqrt(2.0);
						}
						if (m1==n1 && m2==n2)
						{
							PiF[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = (PiF_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] - PiF_temp[ind_aux2U_Pi(inv[m1],inv[n1],m2,n2,t1,t2)] - PiF_temp[ind_aux2U_Pi(m1,n1,inv[m2],inv[n2],t1,t2)] + PiF_temp[ind_aux2U_Pi(inv[m1],inv[n1],inv[m2],inv[n2],t1,t2)])/sqrt(2.0);
							Pirho[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = (Pirho_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] - Pirho_temp[ind_aux2U_Pi(inv[m1],inv[n1],m2,n2,t1,t2)] - Pirho_temp[ind_aux2U_Pi(m1,n1,inv[m2],inv[n2],t1,t2)] + Pirho_temp[ind_aux2U_Pi(inv[m1],inv[n1],inv[m2],inv[n2],t1,t2)])/sqrt(2.0);
						}
						if (m1!=n1 && m2!=n2)
						{
							PiF[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = PiF_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)];
							Pirho[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)] = Pirho_temp[ind_aux2U_Pi(m1,n1,m2,n2,t1,t2)];
						}
					
					}
				}
			}
		}
	}
	*/
	
	
	
	
	
	
	// Auxiliary propagator: Drho, DF
	#pragma omp parallel for
    for (int t2=0; t2<t1; t2++)
    {
		for (int k=0; k<Ntotal; k++)
		{
			MatrixXcd Amatrix(Nm2,Nm2);
			MatrixXcd Irho(Nm2,Nm2);
			MatrixXcd IF(Nm2,Nm2);
			
			// Compute inversion matrix
			for (int m1=0; m1<Nm; m1++) {
				for (int n1=0; n1<Nm; n1++)	{
					for (int m2=0; m2<Nm; m2++)	{
						for (int n2=0; n2<Nm; n2++)	{
							Amatrix(ind_mn(n1,m1),ind_mn(m2,n2)) = comp(kron_del(n1,m2)*kron_del(m1,n2),0) - 0.5 * dt * Jk[ind_JkU(n1,m1,k)] * Pirho[ind_aux2U_Pi(m1,n1,m2,n2,t1,t1)];
						}
					}
				}
			}
			if (sqrabs(Amatrix.determinant())<0.000001) cout << "\nWARNING/compute_selfenergies_U_channel: Amatrix not invertible.\n";
			MatrixXcd Ainverse = Amatrix.inverse();
			
			
			// Drho
			for (int m1=0; m1<Nm; m1++)
			{
				for (int n1=0; n1<Nm; n1++)
				{
					for (int m2=0; m2<Nm; m2++)
					{
						for (int n2=0; n2<Nm; n2++)
						{
							comp mem_rho, mass_rho;
							int ind_r0, ind_r1, ind_r2;
							int ind_J0, ind_J1, ind_J2;
								
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_rho = - Jk[ind_J0] * Jk[ind_J1] * Pirho[ind_r0];
							
							// Memory integral
							mem_rho = 0;
							ind_J2 = ind_JkU(n1,m1,k);
			
							if (t1>t2)
							{	
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,t2);
										ind_r2 = ind_aux2U_D(m3,n3,m2,n2,t2,t2,k);
										mem_rho += 0.5 * Pirho[ind_r1] * Drho[ind_r2];
										
										for (int l=t2+1; l<=t1-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m3,n3,m2,n2,l,t2,k);
											mem_rho += Pirho[ind_r1] * Drho[ind_r2];
										}
									}
								}
							}
				
							mem_rho *= Jk[ind_J2] * dt;		// Maybe can be improved by omitting terms where Jk=0.
							
							// Store rhs in matrix
							Irho(ind_mn(n1,m1),ind_mn(m2,n2)) = mass_rho + mem_rho;
							
							// All together
							//Drho[ind_w] = mass_rho + mem_rho;
							
							/*
							// Correct for extra chi
							if (m1==n1 && m1==1 && (m2!=1 || n2!=1))
							{
								Irho(ind_mn(1,1),ind_mn(m2,n2)) = - Irho(ind_mn(0,0),ind_mn(m2,n2));
							}
							if (m2==n2 && m2==1 && (m1!=1 || n1!=1))
							{
								Irho(ind_mn(n1,m1),ind_mn(1,1)) = - Irho(ind_mn(n1,m1),ind_mn(0,0));
							}
							if (m2==n2 && m2==1 && m1==n1 && m1==1)
							{
								Irho(ind_mn(1,1),ind_mn(1,1)) = Irho(ind_mn(0,0),ind_mn(0,0));
							}*/
							
						}
					}
				}
			}
			
			if (t1>t2) // This if-clauses are trivial, but well
			{
				// Matrix multiplication
				MatrixXcd AIrho = Ainverse * Irho;
				
				// Fill Drho
				for (int m1=0; m1<Nm; m1++) {
					for (int n1=0; n1<Nm; n1++) {
						for (int m2=0; m2<Nm; m2++) {
							for (int n2=0; n2<Nm; n2++)	{
								int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								Drho[ind_w] = AIrho(ind_mn(n1,m1),ind_mn(m2,n2));
								
								/*
								// Correct for extra chi
								if (m1==n1 && m1==1)
								{
									Drho[ind_aux2U_D(1,1,m2,n2,t1,t2,k)] = - Drho[ind_aux2U_D(0,0,m2,n2,t1,t2,k)];
								}
								if (m2==n2 && m2==1)
								{
									Drho[ind_aux2U_D(n1,m1,1,1,t1,t2,k)] = - Drho[ind_aux2U_D(n1,m1,0,0,t1,t2,k)];
								}*/
							}
						}
					}
				}
			}
			
			
			// DF
			for (int m1=0; m1<Nm; m1++)
			{
				for (int n1=0; n1<Nm; n1++)
				{
					for (int m2=0; m2<Nm; m2++)
					{
						for (int n2=0; n2<Nm; n2++)
						{
							comp mem_F1, mem_F2, mass_F;
							int ind_r0, ind_r1, ind_r2;
							int ind_J0, ind_J1, ind_J2;
							int minus_k = get_spacemom_array_position( modN(-table_spacemom_inds[k]) );
							
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_F = - Jk[ind_J0] * Jk[ind_J1] * PiF[ind_r0];
							
							
							// Memory integral 1
							mem_F1 = 0;
							ind_J2 = ind_JkU(n1,m1,k);
			
							if (t1>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,0);
										ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,0,minus_k);
										mem_F1 += 0.5 * Pirho[ind_r1] * DF[ind_r2];
										
										for (int l=1; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
										for (int l=t2; l<=t1-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m3,n3,m2,n2,l,t2,k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
									}
								}
							}

							mem_F1 *= Jk[ind_J2] * dt;
							
							
							// Memory integral 2
							mem_F2 = 0;
				
							if (t2>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,0);
										ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,0,minus_k);
										mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
										
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,t2);
										ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,t2,minus_k);
										mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
										
										for (int l=1; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F2 += PiF[ind_r1] * Drho[ind_r2];
										}
									}
								}
							}
				
							mem_F2 *= Jk[ind_J2] * dt;
							
							// Store rhs in matrix
							IF(ind_mn(n1,m1),ind_mn(m2,n2)) = mass_F + mem_F1 + mem_F2;
							
							// Fill DF if t1==0
							if (t1==0)
							{
								int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								DF[ind_w] = IF(ind_mn(n1,m1),ind_mn(m2,n2));
							}
							
							// All together
							//DF[ind_w] = mass_F + mem_F1 + mem_F2;
							
							/*
							// Correct for extra chi
							if (m1==n1 && m1==1 && (m2!=1 || n2!=1))
							{
								IF(ind_mn(1,1),ind_mn(m2,n2)) = - IF(ind_mn(0,0),ind_mn(m2,n2));
							}
							if (m2==n2 && m2==1 && (m1!=1 || n1!=1))
							{
								IF(ind_mn(n1,m1),ind_mn(1,1)) = - IF(ind_mn(n1,m1),ind_mn(0,0));
							}
							if (m2==n2 && m2==1 && m1==n1 && m1==1)
							{
								IF(ind_mn(1,1),ind_mn(1,1)) = IF(ind_mn(0,0),ind_mn(0,0));
							}*/
							
						}
					}
				}
			}
		
			if (t1>0)
			{
				// Matrix multiplication
				MatrixXcd AIF = Ainverse * IF;
				
				// Fill DF
				for (int m1=0; m1<Nm; m1++) {
					for (int n1=0; n1<Nm; n1++) {
						for (int m2=0; m2<Nm; m2++) {
							for (int n2=0; n2<Nm; n2++)	{
								int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								DF[ind_w] = AIF(ind_mn(n1,m1),ind_mn(m2,n2));
								
								/*
								// Correct for extra chi
								if (m1==n1 && m1==1 && (m2!=1 || n2!=1))
								{
									DF[ind_aux2U_D(1,1,m2,n2,t1,t2,k)] = - DF[ind_aux2U_D(0,0,m2,n2,t1,t2,k)];
								}
								if (m2==n2 && m2==1 && (m1!=1 || n1!=1))
								{
									DF[ind_aux2U_D(n1,m1,1,1,t1,t2,k)] = - DF[ind_aux2U_D(n1,m1,0,0,t1,t2,k)];
								}
								if (m2==n2 && m2==1 && m1==n1 && m1==1)
								{
									DF[ind_aux2U_D(1,1,1,1,t1,t2,k)] = DF[ind_aux2U_D(0,0,0,0,t1,t2,k)];
								}*/
								
							}
						}
					}
				}
			}
		
		
		}
		
		
	}
	
	
	
	
	// Diagonal DF, Drho
    {
		int t2 = t1;
		
		// Drho (same as before but no integral)
		for (int k=0; k<Ntotal; k++)
		{
			for (int m1=0; m1<Nm; m1++)
			{
				for (int n1=0; n1<Nm; n1++)
				{
					for (int m2=0; m2<Nm; m2++)
					{
						for (int n2=0; n2<Nm; n2++)
						{
							comp mass_rho;
							int ind_r0;
							int ind_J0, ind_J1;
							int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_rho = - Jk[ind_J0] * Jk[ind_J1] * Pirho[ind_r0];
							
							// All together
							Drho[ind_w] = mass_rho;
						}
					}
				}
			}
		}
		
		
		
		// DF (same as before)
		for (int k=0; k<Ntotal; k++)
		{
			MatrixXcd Amatrix(Nm2,Nm2);
			MatrixXcd IF(Nm2,Nm2);
		
			// Compute inversion matrix (same as before)
			for (int m1=0; m1<Nm; m1++) {
				for (int n1=0; n1<Nm; n1++)	{
					for (int m2=0; m2<Nm; m2++)	{
						for (int n2=0; n2<Nm; n2++)	{
							Amatrix(ind_mn(n1,m1),ind_mn(m2,n2)) = comp(kron_del(n1,m2)*kron_del(m1,n2),0) - 0.5 * dt * Jk[ind_JkU(n1,m1,k)] * Pirho[ind_aux2U_Pi(m1,n1,m2,n2,t1,t1)];
						}
					}
				}
			}
			if (sqrabs(Amatrix.determinant())<0.000001) cout << "\nWARNING/compute_selfenergies_U_channel: Amatrix not invertible.\n";
			MatrixXcd Ainverse = Amatrix.inverse();
			
			for (int m1=0; m1<Nm; m1++)
			{
				for (int n1=0; n1<Nm; n1++)
				{
					for (int m2=0; m2<Nm; m2++)
					{
						for (int n2=0; n2<Nm; n2++)
						{
							comp mem_F1, mem_F2, mass_F;
							int ind_r0, ind_r1, ind_r2;
							int ind_J0, ind_J1, ind_J2;
							int minus_k = get_spacemom_array_position( modN(-table_spacemom_inds[k]) );
							
							// Mass term
							ind_r0 = ind_aux2U_Pi(m1,n1,n2,m2,t1,t2);
							ind_J0 = ind_JkU(n1,m1,k);
							ind_J1 = ind_JkU(n2,m2,k);
							
							mass_F = - Jk[ind_J0] * Jk[ind_J1] * PiF[ind_r0];
							
							
							// Memory integral 1
							mem_F1 = 0;
							ind_J2 = ind_JkU(n1,m1,k);
			
							if (t1>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,0);
										ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,0,minus_k);
										mem_F1 += 0.5 * Pirho[ind_r1] * DF[ind_r2];
										
										for (int l=1; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
										for (int l=t2; l<=t1-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m3,n3,m2,n2,l,t2,k);
											mem_F1 += Pirho[ind_r1] * DF[ind_r2];
										}
									}
								}
							}

							mem_F1 *= Jk[ind_J2] * dt;
							
							
							// Memory integral 2
							mem_F2 = 0;
				
							if (t2>0)
							{
								for (int m3=0; m3<Nm; m3++)
								{
									for (int n3=0; n3<Nm; n3++)
									{
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,0);
										ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,0,minus_k);
										mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
										
										ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,t2);
										ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,t2,minus_k);
										mem_F2 += 0.5 * PiF[ind_r1] * Drho[ind_r2];
										
										for (int l=1; l<=t2-1; l++)
										{
											ind_r1 = ind_aux2U_Pi(m1,n1,m3,n3,t1,l);
											ind_r2 = ind_aux2U_D(m2,n2,m3,n3,t2,l,minus_k);
											mem_F2 += PiF[ind_r1] * Drho[ind_r2];
										}
									}
								}
							}
				
							mem_F2 *= Jk[ind_J2] * dt;
							
							// Store rhs in matrix
							IF(ind_mn(n1,m1),ind_mn(m2,n2)) = mass_F + mem_F1 + mem_F2;
							
							// Fill DF if t1==0
							if (t1==0)
							{
								int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								DF[ind_w] = IF(ind_mn(n1,m1),ind_mn(m2,n2));
							}
							
							//if (t1==1) cout << mass_F + mem_F1 + mem_F2 << '\t' << IF(ind_mn(n1,m1),ind_mn(m2,n2)) << endl;
							
							// All together
							//DF[ind_w] = mass_F + mem_F1 + mem_F2;
							
							/*
							// Correct for extra chi
							if (m1==n1 && m1==1 && (m2!=1 || n2!=1))
							{
								IF(ind_mn(1,1),ind_mn(m2,n2)) = - IF(ind_mn(0,0),ind_mn(m2,n2));
							}
							if (m2==n2 && m2==1 && (m1!=1 || n1!=1))
							{
								IF(ind_mn(n1,m1),ind_mn(1,1)) = - IF(ind_mn(n1,m1),ind_mn(0,0));
							}
							if (m2==n2 && m2==1 && m1==n1 && m1==1)
							{
								IF(ind_mn(1,1),ind_mn(1,1)) = IF(ind_mn(0,0),ind_mn(0,0));
							}*/
							
						}
					}
				}
			}
		
			if (t1>0)
			{
				// Matrix multiplication
				MatrixXcd AIF = Ainverse * IF;
				
				//if (t1==1) cout << Ainverse(0,0) << '\t' << IF(0,0) << endl;
				//if (t1==1) cout << AIF(0,0) << endl;
				
				// Fill DF
				for (int m1=0; m1<Nm; m1++) {
					for (int n1=0; n1<Nm; n1++) {
						for (int m2=0; m2<Nm; m2++) {
							for (int n2=0; n2<Nm; n2++)	{
								int ind_w = ind_aux2U_D(n1,m1,m2,n2,t1,t2,k);
								DF[ind_w] = AIF(ind_mn(n1,m1),ind_mn(m2,n2));
								
								/*
								// Correct for extra chi
								if (m1==n1 && m1==1 && (m2!=1 || n2!=1))
								{
									DF[ind_aux2U_D(1,1,m2,n2,t1,t2,k)] = - DF[ind_aux2U_D(0,0,m2,n2,t1,t2,k)];
								}
								if (m2==n2 && m2==1 && (m1!=1 || n1!=1))
								{
									DF[ind_aux2U_D(n1,m1,1,1,t1,t2,k)] = - DF[ind_aux2U_D(n1,m1,0,0,t1,t2,k)];
								}
								if (m2==n2 && m2==1 && m1==n1 && m1==1)
								{
									DF[ind_aux2U_D(1,1,1,1,t1,t2,k)] = - DF[ind_aux2U_D(0,0,0,0,t1,t2,k)];
								}*/
								
							}
						}
					}
				}
			}
			
			
		}
		
	}
	


	
	// Fermion self-energy: selfF, selfrho
	#pragma omp parallel for
    for (int t2=0; t2<=t1; t2++)
    {
		for (int m=0; m<Nm; m++)
		{
			for (int n=0; n<Nm; n++)
			{	
				int ind_w = ind_fermion(m,n,t1,t2);
				int ind_r1, ind_r2;
				comp DFtemp, Drhotemp;
				
				selfF[ind_w] = 0;
				selfrho[ind_w] = 0;	// Just in case
				
				for (int a=0; a<Nm; a++)
				{
					for (int b=0; b<Nm; b++)
					{
						DFtemp = 0;
						Drhotemp = 0;
						ind_r1 = ind_fermion(a,b,t1,t2);
						
						for (int k=0; k<Ntotal; k++) // Fourier trafo of x=0
						{
							ind_r2 = ind_aux2U_D(m,a,b,n,t1,t2,k);
							DFtemp += DF[ind_r2];
							Drhotemp += Drho[ind_r2];
						}
						
						DFtemp *= oneoverNtotal;
						Drhotemp *= oneoverNtotal;
						
						selfF[ind_w] += sign * ( - ( F[ind_r1] * DFtemp - 0.25 * rho[ind_r1] * Drhotemp ) );
						selfrho[ind_w] += sign * ( - ( rho[ind_r1] * DFtemp + F[ind_r1] * Drhotemp ) );
					}
				}
					
			}
		}
	}
	
	
    //cpu_time = clock()-cpu_time;
    //cout << ((float)cpu_time)/CLOCKS_PER_SEC << endl;
}





void fermions_2PI::shift_arrays ()
{
	int ind_w, ind_r;
	
    // Shift F, rho, selfF, selfrho
    for (int t1=0; t1<Ntcut-1; t1++)
    {
        for (int t2=0; t2<=t1; t2++)
        {
            for (int m=0; m<Nm; m++)
            {
                for (int n=0; n<Nm; n++)
                {
					ind_w = ind_fermion(m,n,t1,t2);
					ind_r = ind_fermion(m,n,t1+1,t2+1);
					
                    F[ind_w] = F[ind_r];
                    rho[ind_w] = rho[ind_r];
                    selfF[ind_w] = selfF[ind_r];
                    selfrho[ind_w] = selfrho[ind_r];
                }
            }
        }
    }
    
    // Set last line of F, rho, selfF, selfrho to zero (just in case)
    for (int t2=0; t2<=Ntcut-1; t2++)
    {
        for (int m=0; m<Nm; m++)
        {
            for (int n=0; n<Nm; n++)
            {
				ind_w = ind_fermion(m,n,Ntcut-1,t2);
				
                F[ind_w] = 0;
                rho[ind_w] = 0;
                selfF[ind_w] = 0;
                selfrho[ind_w] = 0;
            }
        }
    }
	
	
	
	if (channel==0)  // S-channel
	{
	    // Shift DF, Drho, PiF, Pirho
	    for (int t1=0; t1<Ntcut-1; t1++)
	    {
	        for (int t2=0; t2<=t1; t2++)
	        {
	            for (int s=0; s<Nco; s++)
	            {
					ind_w = ind_aux2S(s,t1,t2);
					ind_r = ind_aux2S(s,t1+1,t2+1);
				
                    DF[ind_w] = DF[ind_r];
                    Drho[ind_w] = Drho[ind_r];
                    PiF[ind_w] = PiF[ind_r];
                    Pirho[ind_w] = Pirho[ind_r];
	            }
	        }
	    }
		
		// Set last line of DF, Drho, PiF, Pirho to zero (just in case)
        for (int t2=0; t2<=Ntcut-1; t2++)
        {
            for (int s=0; s<Nco; s++)
            {
				ind_w = ind_aux2S(s,Ntcut-1,t2);
			
                DF[ind_w] = 0;
                Drho[ind_w] = 0;
                PiF[ind_w] = 0;
                Pirho[ind_w] = 0;
            }
        }
		
	}
	
	
	
	if (channel==1)  // U-channel
	{
	    // Shift DF, Drho, PiF, Pirho
	    for (int t1=0; t1<Ntcut-1; t1++)
	    {
	        for (int t2=0; t2<=t1; t2++)
	        {
				
				for (int m1=0; m1<Nm; m1++)
				{
					for (int n1=0; n1<Nm; n1++)
					{
						for (int m2=0; m2<Nm; m2++)
						{
							for (int n2=0; n2<Nm; n2++)
							{
								ind_w = ind_aux2U_Pi(m1,n1,m2,n2,t1,t2);
								ind_r = ind_aux2U_Pi(m1,n1,m2,n2,t1+1,t2+1);
								
			                    PiF[ind_w] = PiF[ind_r];
			                    Pirho[ind_w] = Pirho[ind_r];
								
								for (int k=0; k<Ntotal; k++)
								{
									ind_w = ind_aux2U_D(m1,n1,m2,n2,t1,t2,k);
									ind_r = ind_aux2U_D(m1,n1,m2,n2,t1+1,t2+1,k);
									
				                    DF[ind_w] = DF[ind_r];
				                    Drho[ind_w] = Drho[ind_r];
								}
								
							}
						}
					}
				}
				
			}
		}
		
		// Set last line of DF, Drho, PiF, Pirho to zero (just in case)
        for (int t2=0; t2<=Ntcut-1; t2++)
        {
			
			for (int m1=0; m1<Nm; m1++)
			{
				for (int n1=0; n1<Nm; n1++)
				{
					for (int m2=0; m2<Nm; m2++)
					{
						for (int n2=0; n2<Nm; n2++)
						{
							ind_w = ind_aux2U_Pi(m1,n1,m2,n2,Ntcut-1,t2);
							
		                    PiF[ind_w] = 0;
		                    Pirho[ind_w] = 0;
							
							for (int k=0; k<Ntotal; k++)
							{
								ind_w = ind_aux2U_D(m1,n1,m2,n2,Ntcut-1,t2,k);
								
			                    DF[ind_w] = 0;
			                    Drho[ind_w] = 0;
							}
							
						}
					}
				}
			}
			
        }
	}
	

}













    /////////////////////////////////////////////////////////////////////////
    
    ////////////////////            OUTPUT          /////////////////////////
    
    /////////////////////////////////////////////////////////////////////////



vector<double> fermions_2PI::compute_energy ()
{
	switch (channel) {
	    case 0: // S-channel
			return compute_energy_S_channel();			
	        break;
			
	    case 1: // U-channel
			return compute_energy_U_channel();			
	        break;
		
	    default:
	        cout << "Wrong type of channel chosen\n";
	        break;
	}
}




vector<double> fermions_2PI::compute_energy_S_channel ()
{
    vector<double> energy;
	comp E_D = 0;
    
    // Energy in F
	for (int s=0; s<Nco; s++)
	{
		E_D += DF[ind_aux2S(s,t,t)];
	}
	E_D *= - 4 * J / 2;
    
    energy.push_back(E_D.real());
	
	//energy.push_back(PiF[ind_aux2S(0,t,0)].real());
	//energy.push_back(selfF[ind_fermion(1,0,t,0)].real());
	
	/*
    for (int t2=0; t2<=t; t2++)
    {
		for (int s=0; s<Nco; s++)
		{
			int minus_s = get_nn_array_position(-table_nn_inds[s]);
			int ind1 = ind_aux2S(s,t,t2);
			int ind2 = ind_aux2S(minus_s,t,t2);
			
			cout << sqrabs(DF[ind1]-conj(DF[ind2])) << '\t' << DF[ind1] << '\t' << DF[ind2] << endl;
			cout << sqrabs(Drho[ind1]-conj(Drho[ind2])) << '\t' << Drho[ind1] << '\t' << Drho[ind2] << endl;
			cout << sqrabs(PiF[ind1]-conj(PiF[ind2])) << '\t' << PiF[ind1] << '\t' << PiF[ind2] << endl;
			cout << sqrabs(Pirho[ind1]-conj(Pirho[ind2])) << '\t' << Pirho[ind1] << '\t' << Pirho[ind2] << endl;
		}
	}*/
	
    
    return energy;
}




vector<double> fermions_2PI::compute_energy_U_channel ()
{
    vector<double> energy;
	comp E_chi = 0;
	comp E_D = 0;
    
	// Energy in chi
	for (int m=0; m<Nm; m++)
	{	
		for (int n=0; n<Nm; n++)
		{
			E_chi += 0.5 * effmass[ind_effmass(m,n)] * F[ind_fermion(m,n,t,t)] * F[ind_fermion(n,m,t,t)];
		}
	}
	
    // Energy in F
	int shiftedk, ind_D;
	for (int m=0; m<Nm; m++)
	{	
		for (int n=0; n<Nm; n++)
		{
			for (int k=0; k<Ntotal; k++)
			{
				shiftedk = get_spacemom_array_position( modN( (m-n)*Qind-table_spacemom_inds[k] ) );
				ind_D = ind_aux2U_D(n,m,m,n,t,t,shiftedk);
				
				E_D += Jijinvk[k] * DF[ind_D];
			}
		}
	}
	E_D *= 1.0/(2*Ntotal);
    
	energy.push_back(E_chi.real());
    energy.push_back(E_D.real());
	energy.push_back(E_chi.real()+E_D.real());
    
    return energy;
}




vector<double> fermions_2PI::compute_populations_and_coherences ()
{
	vector<double> pops;
	vector<double> spins;
	
	// Compute populations
	for (int m=0; m<Nm; m++)
	{
		pops.push_back( (0.5-F[ind_fermion(m,m,t,t)]).real() );
	}
	
	// Compute sigma_x and sigma_y for each pair of levels
	if (Nm>=2)
	{
		for (int m=0; m<Nm; m++)
		{
			for (int n=m+1; n<Nm; n++)
			{
				double sx=0, sy=0;
		
				sx = ( -2.0 * F[ind_fermion(n,m,t,t)] ).real();
				sy = ( 2.0 * F[ind_fermion(n,m,t,t)] ).imag();
				
				spins.push_back(sx);
				spins.push_back(sy);
				
			}
		}
	}
	else
	{
		cout << "Nm must be 2 or larger to output spin observables.\n";
	}
	
	// Put vectors together
	vector<double> out(pops);
	out.insert( out.end(), spins.begin(), spins.end() );
	
	return out;
}




vector<double> fermions_2PI::compute_spin ()
{
	vector<double> spins;
	
	if (Nm>=2)
	{
		comp sx=0, sy=0, sz=0, n=0;
		comp Dtest=0;
		
		sx = -( F[ind_fermion(0,1,t,t)] + F[ind_fermion(1,0,t,t)] );
		sy = -( I * F[ind_fermion(0,1,t,t)] - I * F[ind_fermion(1,0,t,t)] );
		sz = -( F[ind_fermion(0,0,t,t)] - F[ind_fermion(1,1,t,t)] );
		n = -( F[ind_fermion(0,0,t,t)] + F[ind_fermion(1,1,t,t)] );
		
		spins.push_back(sx.real());
		spins.push_back(sy.real());
		spins.push_back(sz.real());
		spins.push_back(n.real());
		
		if (channel==1)
		{
			// Test only valid for U-channel
			for (int m=0; m<Nm; m++)
			{	
				for (int k=0; k<Ntotal; k++)
				{
					Dtest += DF[ind_aux2U_D(m,m,0,0,t,t,k)];
				}
			}
			
			spins.push_back(Dtest.real());
			spins.push_back(Dtest.imag());
		}
		
	}
	else
	{
		cout << "Nm must be 2 or larger to output spin observables.\n";
	}
	
	return spins;
}




/*
void fermions_2PI::bin_log()
{
    physMom = new double [Ntotal];
    momPerBin = new int [Nbins];
    binAvFactor = new my_double [Nbins];
    whichBin = new int [Ntotal];
    kBin = new my_double [Nbins];
    
    for(int i=0; i<Nbins; i++) momPerBin[i]=0.0;
    
    int maxN = Nlist[0];
    for (int i=1; i<DIM; i++)
    {
        if (Nlist[i]>maxN) maxN = Nlist[i];
    }// longest side of grid
    
    double smallest = 2*sin(M_PI/maxN);      // smallest momentum on grid (apart from 0)
    
    double biggest = sqrt(DIM*4.0);          // largest momentum on grid
    
    logstep = (log(biggest)-log(smallest))/(Nbins-1);    // log width of bin
    double smlog = log(smallest);
    double linstep=exp(logstep);
    
    
    kBin[0]=0.0;
    double binleft = smallest; // *exp(0.5*step);
    for(int i=1; i<Nbins; i++) { kBin[i]=binleft; binleft*=linstep; }     // fill kBin with left momentum of each bin
    
    
    double fakt = 4.0;
    
    vector<int> indices(DIM,0);
    
    for (int n=0; n<Ntotal; n++)
    {
        get_spacemom_indices(indices,n);
        
        double ksq = 0; // Physical momentum squared
        for (int i=0; i<DIM; i++) ksq += sin(M_PI*indices[i]/Nlist[i])*sin(M_PI*indices[i]/Nlist[i]);
        ksq *= fakt;
        
        physMom[n] = ksq;       // Save physical momentum of index n
        
        double logk=0.5*log(ksq);
        int mybin=(int)((logk-smlog)/logstep)+1;
        if (ksq==0.0) mybin=0;
        if (mybin>=Nbins) mybin=Nbins-1;        // calculate corresponding bin for given k
        
        momPerBin[mybin]++;                         // 1 momentum more in bin mybin
        whichBin[n] = mybin;                      // associate bin to given fourier-momentum
        
    }
    
    for(int i=0; i<Nbins; i++)
    {
        if (momPerBin[i]!=0) binAvFactor[i]=1.0/momPerBin[i];
        else binAvFactor[i]=0.0;
    }
    
    
}// bin_log

*/




























