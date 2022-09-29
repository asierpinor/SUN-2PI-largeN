#ifndef FERMIONS_2PI_H
#define FERMIONS_2PI_H

using namespace std;



//This makes a list of input variables with their type, as well as a running index {1,2,...,varnum}. See meaning of variables in .sh file
#define varlist VAR(N,int,1) VAR(DIM,int,2) VAR(Nx,int,3) VAR(Ny,int,4) VAR(Nz,int,5) VAR(J,double,6) VAR(Nm,int,7) \
    VAR(IC,int,8) VAR(qx,int,9) VAR(qy,int,10) VAR(qz,int,11) \
    VAR(approx,int,12) VAR(channel,int,13) VAR(Nt,int,14) VAR(Ntcut,int,15) VAR(dt,double,16) VAR(difmethod,int,17) \
    VAR(tnum,int,18) VAR(seed,long int,19) VAR(max_mem,double,20) VAR(nfile,int,21) VAR(Nbins,int,22) VAR(spinstep,int,23) VAR(energystep,int,24)
#define strlist STR(folder,25)
#define varnum 25


// Define extern variables of previous list of variables
#define VAR(aa,bb,cc) extern bb aa;
varlist
#undef VAR
extern char folder[1024];
//




#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   // pi
#endif

#ifndef TYPE_DEF
#define TYPE_DEF
typedef double my_double;
typedef complex<my_double> comp;
typedef vector< comp > state_type;  // The type of container used to hold the state, vector of spins.
#endif





template <class T>
vector<T> operator+ (vector<T> v, vector<T> w)
{
	int length = v.size();
	if (length!=w.size()) cout << "Vectors of different lengths provided.\n";
	vector<T> out(length);
	for (int i=0; i<length; i++) out[i] = v[i] + w[i];
	return out;
}


template <class T>
vector<T> operator- (vector<T> v, vector<T> w)
{
	int length = v.size();
	if (length!=w.size()) cout << "Vectors of different lengths provided.\n";
	vector<T> out(length);
	for (int i=0; i<length; i++) out[i] = v[i] - w[i];
	return out;
}


template <class T>
vector<T> operator- (vector<T> v)
{
	int length = v.size();
	vector<T> out(length);
	for (int i=0; i<length; i++) out[i] = -v[i];
	return out;
}


template<class T,class S>
vector<T> operator* (S scalar, vector<T> v)
{
	int length = v.size();
	vector<T> out(length);
	for (int i=0; i<length; i++) out[i] = scalar*v[i];
	return out;
}


template<class T>
T operator* (vector<T> v, vector<T> w)
{
	int length = v.size();
	if (length!=w.size()) cout << "Vectors of different lengths provided.\n";
	T out = v[0]*w[0];
	for (int i=1; i<length; i++) out = out + v[i]*w[i];
	return out;
}






class fermions_2PI {
    
	
    // Data arrays
    comp * F, * rho;
    comp * selfF, * selfrho;
	comp * chi;
	comp * DF, * Drho;
	comp * PiF, * Pirho;
	double * Jk;
	double * Jijinvk;
	double * effmass;
	int64_t size_Frho, size_D, size_Pi, size_chi;
	int64_t size_rhsFrho;
	int64_t size_Jk, size_Jijinvk, size_effmass;
	
    // For integration
    int t;      // Current time. With memory cutoff it does not correspond to the time t of main.cpp
    comp I;
	comp * rhs_F, * rhs_rho;
	comp * rhs_F_plus1, * rhs_rho_plus1;
	int ECiter;
    //int * ind_nneigh;   // Indices of nearest neighbours for gradient computation
	
    // Other
	int dim;
    vector<int> Nlist;
	vector< vector<double> > table_spacemom_vectors, table_nn_vectors;		// space/momentum (physical) and nearest-neighbor vectors corresponding to each index
	vector< vector<int> > table_spacemom_inds, table_nn_inds;		// space/momentum and nearest-neighbor vectors corresponding to each index (ints)
	vector<double> Q;							// Spiral vector
	vector<int> Qind;
	bool memoryOK;
	bool invertibleJ;
	
    // Dummy constants
    int Ntotal;
	int Nco, Nco2;  // Coordination number
    int Ntcut2halfsize;
	int Nm2, Nm4;
	int dim2;
    double oneoverNtotal;
    double dt2;
	
	/*
    // Initial conditions
    gsl_rng * r;
    double sigma;
    
    // Fourier transforms in quantum case
    double * Fqu_k_realsym, * Fqu_x_realsym;
    double * Kqu_k_realsym, * Kqu_x_realsym;
    fftw_plan fft_Fqu_realsym_xk, fft_Fqu_realsym_kx;
    fftw_plan fft_Kqu_realsym_xk, fft_Kqu_realsym_kx;
    
    // Fourier transforms in classical case
    double * phicl_x_FT, * picl_x_FT;
    comp * phicl_k_FT, * picl_k_FT;
    fftw_plan fft_phicl_xk, fft_phicl_kx;
    fftw_plan fft_picl_xk, fft_picl_kx;
    
    double * Fcl_x_FT, * Kcl_x_FT;
    comp * Fcl_k_FT, * Kcl_k_FT;
    fftw_plan fft_Fcl_xk, fft_Kcl_xk;
    
    
    // Physical Momenta and binning
    double * physMom;           // associates physical momentum to lattice momentum (i,j,...)
    double logstep;
    int * momPerBin;          // number of k's in one bin
    my_double * binAvFactor;        // 1/(number of k's in one bin)
    int * whichBin;                 // associates a bin number to each fourier-momentum (i,j,k)
    my_double * kBin;               // contains left values of k that define bins
    
    // Output
    my_double * spectrum;           // will hold final binned spectrum
    my_double * spectrum_phi2cl;           // will hold final binned |phi(p)|^2 for classical case
    my_double * spectrum_pi2cl;           // will hold final binned |pi(p)|^2 for classical case
    my_double * spectrum_Fcl;           // will hold final binned F for classical case
    my_double * spectrum_Kcl;           // will hold final binned F for classical case
    
    
    
    
    
	
	*/
    
    
    
public:
    
    fermions_2PI(gsl_rng * rndm);
    ~fermions_2PI();
	
    inline double sqr (double a) {return a*a;}
    inline double sqrabs (comp z) {return sqr(z.real())+sqr(z.imag());}    // Absolute value squared of complex number
	
	
	inline int ind_fermion (int m, int n, int t1, int t2) { return (m*Nm + n)*Ntcut2halfsize + (t1*(t1+1))/2 + t2; } // Returns array position for F, rho two-point functions
	inline int ind_rhs (int m, int n, int t1, int t2) { return (m*Nm + n)*(t1+1) + t2; } // Returns array position for right-hand-side of eom of F, rho for fixed t1 (only t2 varies)
	//inline int ind_aux1S (int s, int t) { return s*Ntcut+t; }  // Returns array position for auxiliary one-point function chi in S-channel
	inline int ind_aux1U (int m, int n, int t) { return (m*Nm + n)*Ntcut + t; }  // Returns array position for auxiliary one-point function chi in U-channel
	inline int ind_aux2S (int s, int t1, int t2) { return s * Ntcut2halfsize + (t1*(t1+1))/2 + t2; } // Returns array position for auxiliary DF, Drho, PiF, Pirho two-point functions
	inline int ind_aux2U_Pi (int m1, int n1, int m2, int n2, int t1, int t2) { return ( (m1*Nm + n1)*Nm2 + (m2*Nm + n2) ) * Ntcut2halfsize + (t1*(t1+1))/2 + t2; } // Returns array position for auxiliary PiF, Pirho two-point functions
	inline int ind_aux2U_D (int m1, int n1, int m2, int n2, int t1, int t2, int k)
		{ return ( ( (m1*Nm + n1)*Nm2 + (m2*Nm + n2) ) * Ntcut2halfsize + (t1*(t1+1))/2 + t2 ) * Ntotal + k; } // Returns array position for auxiliary DF, Drho two-point functions. Note definition of indices: D^{m1m2}_{n1n2}
	inline int ind_JkU (int m, int n, int k) { return (m*Nm + n) * Ntotal + k; } // Returns array position for J in U-channel
	inline int ind_J (int i, int j) { return i * Ntotal + j; } // Returns array position for Jij matrix
	inline int ind_effmass (int m, int n) { return m*Nm + n; } // Returns array position for J in U-channel
	inline int ind_mn (int m, int n) { return m*Nm + n; } // Returns array position for m,n index
	
    

    void get_spacemom_indices (vector<int> & indices, int n);    // Gets indices of each dimension assuming row-order for space/momentum vector
    int get_spacemom_array_position (vector<int> indices);    // Computes the position in the array given the individual indices of each dimension for space/momentum vector
    void get_nn_indices (vector<int> & indices, int n);    // Gets indices of each dimension assuming row-order for nearest-neighbor vector
    int get_nn_array_position (vector<int> indices);    // Computes the position in the array given the individual indices of each dimension for nearest-neighbor vector
	
	vector<int> modN (vector<int> v);
	int kron_del (int a, int b);
    
	/*
    void fill_ind_nneigh ();        // Fills array with indices of nearest neighbours for gradient computation
	*/
	
    void funny_constants ();
	
	void fill_table_spacemom ();		// Fill table with space/momentum vectors
	void fill_table_nn ();		// Fill table with coordination number vectors
	void fill_Jk ();		// Fill Jk array
	void fill_Jijinvk ();		// Fill Jijinvk array
	void fill_effmass ();		// Fill effmass array
	
	bool check_memory () {return memoryOK;}
    
	/*
    vector<double> getPhysMom (int binNumber);  // Outputs physical momentum and momenta per bin for given bin number n
    
    */
///////////////            INITIAL CONDITIONS          //////////////////
    
	
	void setToZero();
	void initialConditions ();    // Choose which noise to add to initial condition

	
///////////////////            DYNAMICS          ////////////////////////
    
    
	void dynamics ();    // Calculates F, rho, DF, Drho, etc at next time step
	void dynamics_euler ();
	void dynamics_PEC ();
	void dynamics_PECE ();
	
	void update_euler();
	void update_trapezoidal();
	
	void save_rhs();	// For P(EC)^k method. Save RHS for next step
	
	void compute_rhs_F (int t1, comp * rhs_array);
	void compute_rhs_rho (int t1, comp * rhs_array);
	
	void compute_selfenergies (int t1);
	void compute_selfenergies_S_channel (int t1);
	void compute_selfenergies_U_channel (int t1);
	
    void shift_arrays ();
	
	/*
	void compute_Frho_memoryInt (double * AF, double * Arho, double * BF, double * Brho, double * CF, double * Crho, double * KF, double * Krho);
    */

    
////////////////////            OUTPUT          /////////////////////////
    
	int time() { return t; };
	int dimension() { return dim; };
	vector<double> Qvector() { return (1.0/M_PI)*Q; };
	
    vector<double> compute_energy ();
	vector<double> compute_energy_S_channel ();
	vector<double> compute_energy_U_channel ();
	
    vector<double> compute_spin ();
	vector<double> compute_populations_and_coherences ();
	
    /*
    vector<double> read_data_quantum();
    
    void bin_log();     // Divides the physical momentum space into bins
    
    */
};
//]



















#endif // FERMIONS_2PI_H


