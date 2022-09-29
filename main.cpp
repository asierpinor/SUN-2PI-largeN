#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <vector>               // std:: vector
//#include <array>                // std::array
#include <cmath>                // sin, cos etc.
#include <complex>
#include <string>
#include <sstream>              // for input/ouput with strings
#include <algorithm>            // std::min_element
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <fftw3.h>              // FFTW
#include <omp.h>                        // OpenMP
#include <Eigen/Dense>

#include "fermions_2PI.hpp"



using namespace std;
using namespace Eigen;



#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   // pi
#endif



// Define global variables of list of variables in boson_gpe.hpp
#define VAR(aa,bb,cc) bb aa;
varlist
#undef VAR
char folder[1024];
//




// DO WE NEED THIS ????????????????????
#ifndef TYPE_DEF
#define TYPE_DEF
typedef double my_double;
typedef complex<my_double> comp;
typedef vector< comp > state_type;  // The type of container used to hold the state
#endif






// Prints execution time in hours, minutes and seconds
void print_time (int t)
{
    int hours = (int)t/3600;
    int minutes = (int)(t-3600*hours)/60;
    int seconds = (int)(t-3600*hours-60*minutes);
    printf("Execution time: %i hours, %i minutes, %i seconds\n",hours,minutes,seconds);
}//









int main(int argc, char* argv[])
{	
    
    long int runtime = (long)time(NULL);
    
// Initialises global variables with values as given in command line
    
    if (argc==varnum+1)
    {
        #define STR(aa,bb) snprintf(aa,1024,argv[bb]);
        #define VAR(aa,bb,cc) aa=(bb)atof(argv[cc]);
        varlist
        strlist
        #undef VAR
        #undef STR
    }
    else { cout << "Error: Missing variable" << endl; return 0;}
		
		
// Naming output
	
    char discr[1024];       // Include in filename
    switch (channel) {
        case 0: // S-channel
			if (approx==0) snprintf(discr,1024,"_Smf");
            if (approx==1) snprintf(discr,1024,"_Snlo");
            break;
			
        case 1: // U-channel
			if (approx==0) snprintf(discr,1024,"_Umf");
            if (approx==1) snprintf(discr,1024,"_Unlo");
            break;
            
        default:
            cout << "Error: Wrong type of channel chosen.\n";
            break;
    }
	
    char string_difmethod[1024];       // Include in filename
    switch (difmethod) {
        case 0: // Euler
            snprintf(string_difmethod,1024,"_euler");
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
			snprintf(string_difmethod,1024,"_PEC%i",difmethod%10);
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
			snprintf(string_difmethod,1024,"_PECE%i",difmethod%10);
		    break;
		
        default:
            cout << "Invalid type of difmethod chosen (string).\n";
            break;
    }
	
    char string_IC[1024];       // Include in filename
    switch (IC) {
        case 0: // All in magnetic level 0
            snprintf(string_IC,1024,"_ICz");
            break;
			
        case 1: // Superposition of magnetic levels 0 and 1
			snprintf(string_IC,1024,"_ICx");
            break;
			
        case 2: // Superposition of magnetic levels 0 and 1
			snprintf(string_IC,1024,"_ICy");
            break;
			
        case 3: // Tilted superposition of magnetic levels 0 and 1
			snprintf(string_IC,1024,"_ICtilted");
            break;
			
        case 31: // Symmetric 3-level superposition
			snprintf(string_IC,1024,"_IC3sym");
            break;
			
        case 32: // Asymmetric 3-level superposition
			snprintf(string_IC,1024,"_IC3asym");
            break;
		
        case 33: // Random 3-level superposition
			snprintf(string_IC,1024,"_IC3rand");
            break;
			
        case 99: // Weird IC for Nm>=3
			snprintf(string_IC,1024,"_ICweird");
            break;
		
        default:
            cout << "Invalid type of IC chosen (string).\n";
			snprintf(string_IC,1024,"_ICdefault");
            break;
    }
	
    char string_memCut[1024];       // Include in filename
    if (Ntcut<Nt) snprintf(string_memCut,1024,"_cut%i",Ntcut);
    else snprintf(string_memCut,1024,"");
	
    
    
	
// OMP and FFTW settings
    
    int numprocs = omp_get_num_procs();
    cout << "Number of threads available: " << numprocs << endl;
    
    if (tnum!=0&&tnum<numprocs) {
        numprocs = tnum;
    }
    omp_set_num_threads(numprocs);
    cout << "Using OMP with " << numprocs << " threads\n";
    
    if (!fftw_init_threads()) cout << "Error with fftw_thread\n";
    fftw_plan_with_nthreads(numprocs);
    
    
    
// Initialize random number generator
    
    long int random_seed = seed;
    if (seed<0) random_seed = (long)time(NULL)+1234*(long)getpid();
    //long int random_seed2 = (rdtsc() % (long)pow(2,30)); //different type of random seed, use this for different seeds within same program
    
    gsl_rng * r = gsl_rng_alloc (gsl_rng_mt19937);
    gsl_rng_set (r,random_seed);
    
    
    
	
// Containers for outputing data
    
    vector<double> times_energy;
    vector< vector<double> > data_energy;
	
	vector<double> times_spin;
    vector< vector<double> > data_spin;
	
	/*
    vector< vector<double> > physMomenta;    // Holds physical momenta of non-empty bins and number of momenta per bin
	
    */
    
	
	
// Settings
    
	fermions_2PI system(r);         // initialize fermion class
	
	if (!system.check_memory()) return 0;
    
	/*
    for (int n=0; n<Nbins; n++)
    {
        vector<double> temp(system.getPhysMom(n));
        if (temp[1]!=0)
        {
            physMomenta.push_back(temp);
        }
    }
	*/
	
	

// Initial conditions
    system.setToZero();
    system.initialConditions();
	
	
// Save observables at initial time
	int next_energy = 0;
	int next_spin = 0;
	
	
	// Energy
    if (energystep>0)
    {
        next_energy += energystep;
		
        data_energy.push_back( system.compute_energy() );
        times_energy.push_back( 0 );
    }
	
	// Spin magnetizations
    if (spinstep>0)
    {
        next_spin += spinstep;
        
		//data_spin.push_back( system.compute_spin() );
		data_spin.push_back( system.compute_populations_and_coherences() );
        times_spin.push_back( 0 );
    }
    
	
// Time Evolution
    for (int t=1; t<Nt; t++)
    {
        cout << t << endl;
        
		// Compute t and set time to t+1
        system.dynamics();
		
        // Save energy
        if (t==next_energy && energystep>0)
        {
            next_energy += energystep;
            
            data_energy.push_back( system.compute_energy() );
            times_energy.push_back( t*dt );
        }
		
		// Save spin magnetizations
	    if (t==next_spin && spinstep>0)
	    {
	        next_spin += spinstep;
        
	        //data_spin.push_back( system.compute_spin() );
			data_spin.push_back( system.compute_populations_and_coherences() );
	        times_spin.push_back( t*dt );
	    }
		
	}
	
	if (times_energy.size() != data_energy.size()) cout << "Sizes of times_energy and output_energy are different.\n";
	if (times_spin.size() != data_spin.size()) cout << "Sizes of times_spin and output_spin are different.\n";
	
	
	
// String for output
	
    char string_Q[1024];       // Include in filename
	vector<double> Qfrac(system.Qvector());
	snprintf(string_Q,1024,"_Q%.2g",Qfrac[0]);
	//for (int i=1; i<system.dimension(); i++) snprintf(string_Q,1024,"-%.2g",Qfrac[i]);
	
	
// Output energy
    
    if (energystep>0)
    {   
        // Output
        char buffer[1024];
        snprintf(buffer,1024,"%s/energy_fermions2PI%s%s_DIM%i_N%i_Nm%i%s%s_J%g_dt%g%s_file%i.txt",folder,discr,string_difmethod,system.dimension(),N,Nm,string_IC,string_Q,J,dt,string_memCut,nfile);
        ofstream outputfile(buffer);
        outputfile.precision(12);
        
        if (outputfile.is_open())
        {
            outputfile << "# Columns: t | Energy chi | Energy DF\n";
            
            #define VAR(aa,bb,cc) << "# " << #aa << "=" << aa << "\n"
            outputfile varlist;
            #undef VAR
            
            for (int t=0; t<times_energy.size(); t++)
            {
                outputfile << times_energy[t];
                
                for (int j=0; j<data_energy[t].size(); j++)
                {
                    outputfile << '\t' << data_energy[t][j];
                }
                
                outputfile << endl;
            }
        }
        
        outputfile.close();
        
    }
	
	
	
// Output spin magnetizations
    
    if (spinstep>0)
    {   
        // Output
        char buffer[1024];
        snprintf(buffer,1024,"%s/spin_fermions2PI%s%s_DIM%i_N%i_Nm%i%s%s_J%g_dt%g%s_file%i.txt",folder,discr,string_difmethod,system.dimension(),N,Nm,string_IC,string_Q,J,dt,string_memCut,nfile);
        ofstream outputfile(buffer);
        outputfile.precision(12);
        
        if (outputfile.is_open())
        {
			//outputfile << "# Columns: t | sx | sy | sz\n";
			outputfile << "# Columns: t | populations (0,...,N-1) | sx & sy (m<n) | sz\n";
            
            #define VAR(aa,bb,cc) << "# " << #aa << "=" << aa << "\n"
            outputfile varlist;
            #undef VAR
            
            for (int t=0; t<times_spin.size(); t++)
            {
                outputfile << times_spin[t];
                
                for (int j=0; j<data_spin[t].size(); j++)
                {
                    outputfile << '\t' << data_spin[t][j];
                }
                
                outputfile << endl;
            }
        }
        
        outputfile.close();
        
    }
	
	
    
	
	///////////////////////////
	///////////////////////////
	///////////////////////////
	/*
	
    
    // For output
    int next_out = 1;
    
    // Save initial spectrum
    if (outstep>=0)
    {
        // Output t=0
        next_out += outstep;
        
        data.push_back( system.read_data() );
        times.push_back(0);     // Save times for first iteration
    }
    
    

    
    //////
    //      Time Evolution
    //////
    for (int t=2; t<Nt; t++)
    {
        cout << t << endl;
        
        // Dynamics
        
        system.dynamics();    // calculates t+1
        
        
        // Output spectrum
        
        if (t==next_out && outstep>=0)
        {
            next_out += outstep;
            
            data.push_back( system.read_data() );
            times.push_back( (t-1) * dt );     // Save times for first iteration
        }
        
        
       
        
        
    }//End of time evolution
    

    if (times.size() != data.size()) cout << "Sizes of times and output are different.\n";
    
    
    */
     
    
	
	/*
    
    // Normalize data and output it
    
    if (outstep>=0)
    {
        int ncolumns = data[0].size() / physMomenta.size();
        
        // Normalize
        double Nfac = 1.0/iter;
        for (int k=0; k<times.size(); k++)
        {
            for (int l=0; l<data[k].size(); l++)
            {
                data[k][l] *= Nfac;
            }
        }
        
        // Output
        for (int k=0; k<times.size(); k++)
        {
            char buffer[bsize];
            snprintf(buffer,1024,"%s/phi42PI%s_DIM%i_N%i%s_loops%i_lambda%g_dt%g%s%s_t%g_file%i.txt",folder,discr,DIM,N,string_IC,loops,lambda,dt,string_runs,string_memCut,times[k],nfile);
            ofstream outputfile(buffer);
            outputfile.precision(12);
            
            
            if (outputfile.is_open())
            {
                if (approx==0) outputfile << "# Columns: Physical Momentum | sqrt(F(t,t,p)K(t,t,p)) \n";
                else if (approx==1 || approx==2) outputfile << "# Columns: Physical Momentum | |phi(t,p)|^2 | |pi(t,p)|^2 | F(t,t,p) | K(t,t,p) \n";
                
                #define VAR(aa,bb,cc) << "# " << #aa << "=" << aa << "\n"
                outputfile varlist;
                #undef VAR
                
                for (int i=0; i<physMomenta.size(); i++)
                {
                    outputfile << physMomenta[i][0];
                    for (int j=0; j<ncolumns; j++)
                    {
                        outputfile << '\t' << data[k][ncolumns*i+j];
                    }
                    outputfile << endl;
                }
            }
            
            outputfile.close();
        }
        
    }
    
    */
	///////////////////////////
	///////////////////////////
	///////////////////////////
	
	
	
    
    runtime = (long)time(NULL) - runtime;       // Execution time in seconds
    print_time ((int)runtime);                   // Print execution time in hours, minutes, seconds
    
	/*
    gsl_rng_free(r);
    fftw_cleanup_threads();
	*/
    
    return 0;
    
    
    

}// MAIN










/*
 //[ Measures the total pseudo-cycles since the processor was powered on. Used to generate random seeds.
 unsigned long long rdtsc(){
 unsigned int lo,hi;
 __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
 return ((unsigned long long)hi << 32) | lo;
 }//]
 */








