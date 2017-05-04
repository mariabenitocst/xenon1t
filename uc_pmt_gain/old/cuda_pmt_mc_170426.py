

cuda_pmt_mc ="""
#include <curand_kernel.h>

extern "C" {


__device__ int gpu_exponential(curandState_t *rand_state, float exp_constant, float exp_offset)
{
    // pdf = 1/const * exp(-(x-offset)/const)
    return -logf(curand_uniform(rand_state)) * exp_constant + exp_offset;
}


__device__ int gpu_discrete_gaussian(curandState_t *rand_state, float mean, float width)
{
    int lower_bound_for_integral = (int)roundf(mean - 3*width);
    int upper_bound_for_integral = (int)roundf(mean + 3*width);

    float integral_of_dist = 0.;
    int k;
    
    // find approximate integral of distribution
    for (int i = 0; i < (upper_bound_for_integral-lower_bound_for_integral+1); i++)
    {
        k = i + lower_bound_for_integral;
        integral_of_dist += expf( -powf(k-mean, 2) / powf(width, 2) / 2.);
    }
    
    // get uniform random number
    float r_uniform = curand_uniform(rand_state);

    float cumulative_dist = 0.;
    
    // add lower bound to CDF
    cumulative_dist += expf( -powf(lower_bound_for_integral-mean, 2) / powf(width, 2) / 2.);
    
    if (r_uniform < cumulative_dist)
        return lower_bound_for_integral;
        
    else
    {
        for (int i = 0; i < (upper_bound_for_integral-lower_bound_for_integral-1); i++)
        {
            k = 1 + lower_bound_for_integral + i;
            cumulative_dist += expf( -powf(k-mean, 2) / powf(width, 2) / 2.) / integral_of_dist;
            
            if ( (r_uniform > cumulative_dist) && (r_uniform < (cumulative_dist + expf( -powf((k+1)-mean, 2) / powf(width, 2) / 2.) / integral_of_dist)) )
                return k+1;
        }
    }
    
    // at this point must return upper bound since all others failed
    return upper_bound_for_integral;

}



__device__ int gpu_binomial(curandState_t *rand_state, int num_trials, float prob_success)
{

    /*
	int x = 0;
	for(int i = 0; i < num_trials; i++) {
    if(curand_uniform(rand_state) < prob_success)
		x += 1;
	}
	return x;
	*/
	
	
	// Rejection Method (from 7.3 of numerical recipes)
	// slower on 970!!
	
	float pi = 3.1415926535;
	int j;
	int nold = -1;
	float am, em, g, angle, p, bnl, sq, t, y;
	float pold = -1.;
	float pc, plog, pclog, en, oldg;
	
	
	p = (prob_success < 0.5 ? prob_success : 1.0 - prob_success);
	
	am = num_trials*p;
	if (num_trials < 25)
	{
		bnl = 0;
		for (j=0; j < num_trials; j++)
		{
			if (curand_uniform(rand_state) < p) bnl += 1;
		}
	}
	else if (am < 1.0)
	{
		g = expf(-am);
		t = 1.;
		for (j=0; j < num_trials; j++)
		{
			t *= curand_uniform(rand_state);
			if (t < g) break;
		}
		bnl = (j <= num_trials ? j : num_trials);
	}
	else
	{
		if (num_trials != nold)
		{
			en = num_trials;
			oldg = lgammaf(en+1.);
			nold = num_trials;
		}
		if (p != pold)
		{
			pc = 1. - p;
			plog = logf(p);
			pclog = logf(pc);
			pold = p;
		}
		sq = powf(2.*am*pc, 0.5);
		do
		{
			do
			{
				angle = pi*curand_uniform(rand_state);
				y = tanf(angle);
				em = sq*y + am;
			} while (em < 0. || em >= (en+1.));
			em = floor(em);
			t = 1.2*sq*(1. + y*y)*expf(oldg - lgammaf(em+1.) - lgammaf(en-em+1.) + em*plog + (en-em)*pclog);
		} while (curand_uniform(rand_state) > t);
		bnl = em;
	}
	if (prob_success != p) bnl = num_trials - bnl;
	return bnl;
	
	
	
	
	// BTRS method (NOT WORKING)
	/*
	
	float p = (prob_success < 0.5 ? prob_success : 1.0 - prob_success);

	float spq = powf(num_trials*p*(1-p), 0.5);
	float b = 1.15 + 2.53 * spq;
	float a = -0.0873 + 0.0248 * b + 0.01 * p;
	float c = num_trials*p + 0.5;
	float v_r = 0.92 - 4.2/b;
	float us = 0.;
	float v = 0;

	int bnl, m;
	float u;
	float alpha, lpq, h;
	int var_break = 0;
	
	if (num_trials*p < 10)
	{
		bnl = 0;
		for (int j=0; j < num_trials; j++)
		{
			if (curand_uniform(rand_state) < p) bnl += 1;
		}
		return bnl;
	}

	while (1)
	{
		bnl = -1;
		while ( bnl < 0 || bnl > num_trials)
		{
			u = curand_uniform(rand_state) - 0.5;
			v = curand_uniform(rand_state);
			us = 0.5 - abs(u);
			bnl = (int)floor((2*a/us + b) * u + c);
			if (us >= 0.07 && v < v_r) var_break = 1;
			if (var_break == 1) break;
		}
		if (var_break == 1) break;

		alpha = (2.83 + 5.1/b)*spq;
		lpq = logf(p/(1-p));
		m = (int)floor((num_trials+1)*p);
		h = lgammaf(m+1) + lgammaf(num_trials-m+1);

		v = v*alpha/(a/(us*us) + b);

		if (v <= h - lgammaf(bnl+1) - lgammaf(num_trials-bnl+1) + (bnl-m)*lpq) var_break = 1;
		if (var_break == 1) break;
	}

	if (prob_success != p) bnl = num_trials - bnl;
	return bnl;
	
	*/

}

// used for finding index for 2d histogram array
// lower bound corresponds to the index
// uses binary search ON SORTED ARRAY
// THIS IS THE TEST WHICH MUST RETURN VOIDS
// AND HAVE POINTER INPUTS
__global__ void test_gpu_find_lower_bound(int *num_elements, float *a_sorted, float *search_value, int *index)
{
	float *first = a_sorted;
	float *iterator = a_sorted;
	int count = *num_elements;
	int step;
	
	if (*search_value < a_sorted[0] || *search_value > a_sorted[*num_elements])
	{
		*index = -1;
		return;
	}
	
	while (count > 0)
	{
		iterator = first;
		step = count / 2;
		iterator += step;
		if (*iterator < *search_value)
		{
			first = ++iterator;
			count -= step + 1;
		}
		else
		{
			count = step;
		}
		// -1 to get lower bound
		*index = iterator - a_sorted - 1;
	}

}


// used for finding index for 2d histogram array
// lower bound corresponds to the index
// uses binary search ON SORTED ARRAY
__device__ int gpu_find_lower_bound(int *num_elements, float *a_sorted, float search_value)
{
	float *first = a_sorted;
	float *iterator = a_sorted;
	int count = *num_elements;
	int step;
	
	if (search_value < a_sorted[0] || search_value > a_sorted[*num_elements])
	{
		return -1;
	}
	
	while (count > 0)
	{
		iterator = first;
		step = count / 2;
		iterator += step;
		if (*iterator < search_value)
		{
			first = ++iterator;
			count -= step + 1;
		}
		else
		{
			count = step;
		}
	}
	// -1 to get lower bound
	return iterator - a_sorted - 1;

}




#define CURAND_CALL ( x ) do { if (( x ) != CURAND_STATUS_SUCCESS ) {\
printf (" Error at % s :% d \ n " , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)

#include <stdio.h>

__global__ void setup_kernel (int nthreads, curandState *state, unsigned long long seed, unsigned long long offset)
{
	int id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//printf("hello\\n");
	if (id >= nthreads)
		return;
	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init (seed, id, offset, &state[id]);
}



__global__ void cascade_pmt_model(curandState *state, int *num_trials, int *num_loops, float *a_hist, float *mean_num_pe, float *prob_hit_first_dynode, float *mean_e_from_dynode, float *width_e_from_dynode, float *probability_electron_ionized, float *underamp_ionization_correction, float *bkg_mean, float *bkg_std, float *bkg_exp, float *prob_exp_bkg, int *num_bins, float *bin_edges)
{
    //printf("hello\\n");
    
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    int num_dynodes = 12;
    float f_tot_num_pe;
    int pe_from_first_dynode;
    int current_num_dynodes;
    
    
    float ionization_correction;
    
    int num_electrons_leaving_dynode;
    
    int repetition_number;
    
    if (iteration < *num_trials)
	{
    
        for (repetition_number=0; repetition_number < *num_loops; repetition_number++)
        {
            //printf("hello\\n");
            
            int i_tot_num_pe = curand_poisson(&s, *mean_num_pe);
            current_num_dynodes = num_dynodes;
            ionization_correction = 1.;
    
            if (*prob_hit_first_dynode < 0 || *prob_hit_first_dynode > 1)
            {	
                state[iteration] = s;
                continue;
            }
        
        
            pe_from_first_dynode = gpu_binomial(&s, i_tot_num_pe, 1-*prob_hit_first_dynode);
            i_tot_num_pe -= pe_from_first_dynode;
            
            // check if all PE are from first dynode
            // if so just pretend all were from cathode
            // but assume one less dynode
            if (i_tot_num_pe == 0)
            {
                i_tot_num_pe = pe_from_first_dynode;
                pe_from_first_dynode = 0;
                current_num_dynodes -= 1;
                ionization_correction = *underamp_ionization_correction;
            }
            
            if (*mean_e_from_dynode < 0)
            {	
                state[iteration] = s;
                continue;
            }
            
            if (i_tot_num_pe > 0)
            {
                for (int i = 0; i < current_num_dynodes; i++)
                {
                    // after first dynode add the PE originating from
                    // first dynode back in
                    if (i == 1)
                        i_tot_num_pe += pe_from_first_dynode;
                        
                
                    if (i_tot_num_pe < 10000)
                    {
                    
                    
                        if (i_tot_num_pe < 15)
                        {
                            num_electrons_leaving_dynode = (int)gpu_discrete_gaussian(&s, *mean_e_from_dynode*i_tot_num_pe, *width_e_from_dynode*powf(i_tot_num_pe, 0.5));
                            if (num_electrons_leaving_dynode < 1)
                                continue;
                        }
                        else
                            num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                        
                        
                        
                        i_tot_num_pe = gpu_binomial(&s, num_electrons_leaving_dynode, *probability_electron_ionized*ionization_correction);
                    }
                    else
                    {
                        num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                    
                        i_tot_num_pe = (curand_normal(&s) * powf(num_electrons_leaving_dynode**probability_electron_ionized*(1-*probability_electron_ionized), 0.5)) + num_electrons_leaving_dynode**probability_electron_ionized;
                    }
                }
            }
            
            
            if (*bkg_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_pe = (curand_normal(&s) * *bkg_std) + *bkg_mean + i_tot_num_pe;
            
            
            if (*bkg_exp < 0)
            {
                state[iteration] = s;
                return;
            }
            if(curand_uniform(&s) < *prob_exp_bkg)
                f_tot_num_pe += gpu_exponential(&s, *bkg_exp, *bkg_mean);
            
            
            bin_number = gpu_find_lower_bound(num_bins, bin_edges, f_tot_num_pe);
            
            if (bin_number == -1)
            {
                state[iteration] = s;
                continue;
                //return;
            }
            
            atomicAdd(&a_hist[bin_number], 1);
            
            state[iteration] = s;
            //printf("hi: %f\\n", f_tot_num_pe);
        }
        
        
		return;
        
        
        
    }
    
}



__global__ void cascade_pmt_model_array(curandState *state, int *num_trials, int *num_loops, float *a_integrals, float *mean_num_pe, float *prob_hit_first_dynode, float *mean_e_from_dynode, float *width_e_from_dynode, float *probability_electron_ionized, float *underamp_ionization_correction, float *bkg_mean, float *bkg_std, float *bkg_exp, float *prob_exp_bkg, int *num_bins, float *bin_edges)
{
    //printf("hello\\n");
    
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    int num_dynodes = 12;
    float f_tot_num_pe;
    int pe_from_first_dynode;
    int current_num_dynodes;
    
    
    float ionization_correction;
    
    int num_electrons_leaving_dynode;
    
    int repetition_number;
    
    if (iteration < *num_trials)
	{
    
        for (repetition_number=0; repetition_number < 1; repetition_number++)
        {
            //printf("hello\\n");
            
            int i_tot_num_pe = curand_poisson(&s, *mean_num_pe);
            current_num_dynodes = num_dynodes;
            ionization_correction = 1.;
    
            if (*prob_hit_first_dynode < 0 || *prob_hit_first_dynode > 1)
            {	
                state[iteration] = s;
                continue;
            }
        
        
            pe_from_first_dynode = gpu_binomial(&s, i_tot_num_pe, 1-*prob_hit_first_dynode);
            i_tot_num_pe -= pe_from_first_dynode;
            
            // check if all PE are from first dynode
            // if so just pretend all were from cathode
            // but assume one less dynode
            if (i_tot_num_pe == 0)
            {
                i_tot_num_pe = pe_from_first_dynode;
                pe_from_first_dynode = 0;
                current_num_dynodes -= 1;
                ionization_correction = *underamp_ionization_correction;
            }
            
            if (*mean_e_from_dynode < 0)
            {	
                state[iteration] = s;
                continue;
            }
            
            if (i_tot_num_pe > 0)
            {
                for (int i = 0; i < current_num_dynodes; i++)
                {
                    // after first dynode add the PE originating from
                    // first dynode back in
                    if (i == 1)
                        i_tot_num_pe += pe_from_first_dynode;
                        
                
                    if (i_tot_num_pe < 10000)
                    {
                    
                    
                        if (i_tot_num_pe < 15)
                        {
                            num_electrons_leaving_dynode = (int)gpu_discrete_gaussian(&s, *mean_e_from_dynode*i_tot_num_pe, *width_e_from_dynode*powf(i_tot_num_pe, 0.5));
                            if (num_electrons_leaving_dynode < 1)
                                continue;
                        }
                        else
                            num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                        
                        
                        
                        i_tot_num_pe = gpu_binomial(&s, num_electrons_leaving_dynode, *probability_electron_ionized*ionization_correction);
                    }
                    else
                    {
                        num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                    
                        i_tot_num_pe = (curand_normal(&s) * powf(num_electrons_leaving_dynode**probability_electron_ionized*(1-*probability_electron_ionized), 0.5)) + num_electrons_leaving_dynode**probability_electron_ionized;
                    }
                }
            }
            
            
            if (*bkg_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_pe = (curand_normal(&s) * *bkg_std) + *bkg_mean + i_tot_num_pe;
            
            
            if (*bkg_exp < 0)
            {
                state[iteration] = s;
                return;
            }
            if(curand_uniform(&s) < *prob_exp_bkg)
                f_tot_num_pe += gpu_exponential(&s, *bkg_exp, *bkg_mean);
            
            a_integrals[iteration] = f_tot_num_pe;
            state[iteration] = s;
            //printf("hi: %f\\n", f_tot_num_pe);
        }
        
        
		return;
        
        
        
    }
    
}





__global__ void pure_cascade_spectrum(curandState *state, int *num_trials, float *a_hist, int *num_pe, float *prob_hit_first_dynode, float *mean_e_from_dynode, float *width_e_from_dynode, float *probability_electron_ionized, int *num_bins, float *bin_edges)
{
    //printf("hello\\n");
    
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    const int num_dynodes = 12;
    float f_tot_num_pe;
    int current_num_dynodes;
    int pe_from_first_dynode;
    
    
    int num_electrons_leaving_dynode;
    
    
    if (iteration < *num_trials)
	{
    
        int i_tot_num_pe = *num_pe;
        current_num_dynodes = num_dynodes;

        if (*prob_hit_first_dynode < 0 || *prob_hit_first_dynode > 1)
        {	
            state[iteration] = s;
            return;
        }
    
        pe_from_first_dynode = gpu_binomial(&s, i_tot_num_pe, 1-*prob_hit_first_dynode);
        i_tot_num_pe -= pe_from_first_dynode;
        
        
        // check if all PE are from first dynode
        // if so just pretend all were from cathode
        // but assume one less dynode
        if (i_tot_num_pe == 0)
        {
            i_tot_num_pe = pe_from_first_dynode;
            pe_from_first_dynode = 0;
            current_num_dynodes -= 1;
        }
        

        if (*mean_e_from_dynode < 0)
		{	
			state[iteration] = s;
			return;
		}
        
        if (i_tot_num_pe > 0)
        {
            for (int i = 0; i < current_num_dynodes; i++)
            {
                if (i_tot_num_pe < 10000)
                {
                
                
                    if (i_tot_num_pe < 15)
                    {
                        num_electrons_leaving_dynode = (int)gpu_discrete_gaussian(&s, *mean_e_from_dynode*i_tot_num_pe, *width_e_from_dynode*powf(i_tot_num_pe, 0.5));
                        if (num_electrons_leaving_dynode < 1)
                            continue;
                    }
                    else
                        num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                    
                    
                    
                    i_tot_num_pe = gpu_binomial(&s, num_electrons_leaving_dynode, *probability_electron_ionized);
                }
                else
                {
                    num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                
                    i_tot_num_pe = (curand_normal(&s) * powf(num_electrons_leaving_dynode**probability_electron_ionized*(1-*probability_electron_ionized), 0.5)) + num_electrons_leaving_dynode**probability_electron_ionized;
                }
            }
        }
        
        // remove zero counts (~5% probability that single electron)
        // frees zero new electrons
        if (i_tot_num_pe == 0)
		{
			state[iteration] = s;
			return;
		}
        
  
        f_tot_num_pe = (float)i_tot_num_pe;
        
        
        
        bin_number = gpu_find_lower_bound(num_bins, bin_edges, f_tot_num_pe);
		
		if (bin_number == -1)
		{
			state[iteration] = s;
			return;
		}
		
		atomicAdd(&a_hist[bin_number], 1);
		
		state[iteration] = s;
        //printf("hi: %f\\n", f_tot_num_pe);
		return;
        
        
        
    }


}




__global__ void fixed_pe_cascade_spectrum(curandState *state, int *num_trials, int *num_loops, float *a_hist, int *num_pe, float *prob_hit_first_dynode, float *mean_e_from_dynode, float *width_e_from_dynode, float *probability_electron_ionized, float *bkg_mean, float *bkg_std, float *bkg_exp, float *prob_exp_bkg, int *num_bins, float *bin_edges)
{

    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    const int num_dynodes = 12;
    const int fixed_num_pe = *num_pe;
    int current_num_dynodes;
    float f_tot_num_pe;
    int i_tot_num_pe;
    int pe_from_first_dynode;
    
    int num_electrons_leaving_dynode;
    
    int repetition_number;
    
    if (iteration < *num_trials)
	{
    
        for (repetition_number=0; repetition_number < *num_loops; repetition_number++)
        {
        
            i_tot_num_pe = fixed_num_pe;
            current_num_dynodes = num_dynodes;


            if (*prob_hit_first_dynode < 0 || *prob_hit_first_dynode > 1)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
        
            
            pe_from_first_dynode = gpu_binomial(&s, i_tot_num_pe, 1-*prob_hit_first_dynode);
            i_tot_num_pe -= pe_from_first_dynode;
            
            // check if all PE are from first dynode
            // if so just pretend all were from cathode
            // but assume one less dynode
            if (i_tot_num_pe == 0)
            {
                i_tot_num_pe = pe_from_first_dynode;
                pe_from_first_dynode = 0;
                current_num_dynodes -= 1;
            }
            
            if (*mean_e_from_dynode < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            
            if (i_tot_num_pe > 0)
            {
                for (int i = 0; i < current_num_dynodes; i++)
                {
                    // after first dynode add the PE originating from
                    // first dynode back in
                    if (i_tot_num_pe < 10000)
                    {
                    
                    
                        if (i_tot_num_pe < 15)
                        {
                            num_electrons_leaving_dynode = (int)gpu_discrete_gaussian(&s, *mean_e_from_dynode*i_tot_num_pe, *width_e_from_dynode*powf(i_tot_num_pe, 0.5));
                            if (num_electrons_leaving_dynode < 1)
                                continue;
                        }
                        else
                            num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                        
                        
                        
                        i_tot_num_pe = gpu_binomial(&s, num_electrons_leaving_dynode, *probability_electron_ionized);
                    }
                    else
                    {
                        num_electrons_leaving_dynode = (int)roundf( (curand_normal(&s) * *width_e_from_dynode*powf(i_tot_num_pe, 0.5)) + *mean_e_from_dynode*i_tot_num_pe );
                    
                        i_tot_num_pe = (curand_normal(&s) * powf(num_electrons_leaving_dynode**probability_electron_ionized*(1-*probability_electron_ionized), 0.5)) + num_electrons_leaving_dynode**probability_electron_ionized;
                    }
                }
            }
            
            
            if (*bkg_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_pe = (curand_normal(&s) * *bkg_std) + *bkg_mean + i_tot_num_pe;
            
            
            if (*bkg_exp < 0)
            {
                state[iteration] = s;
                return;
            }
            if(curand_uniform(&s) < *prob_exp_bkg)
                f_tot_num_pe += gpu_exponential(&s, *bkg_exp, *bkg_mean);
            
            
            
            bin_number = gpu_find_lower_bound(num_bins, bin_edges, f_tot_num_pe);
            
            if (bin_number == -1)
            {
                state[iteration] = s;
                continue;
                //return;
            }
            
            atomicAdd(&a_hist[bin_number], 1);
            
            state[iteration] = s;
            //printf("hi: %f\\n", f_tot_num_pe);
            
        }
        
		return;
        
    }

}


// gaussian functions


__global__ void gaussian_pmt_model(curandState *state, int *num_trials, int *num_loops, float *a_hist, float *mean_num_pe, float *prob_not_under_amp, float *spe_mean, float *spe_std, float *under_amp_mean, float *under_amp_std, float *bkg_mean, float *bkg_std, float *bkg_exp, float *prob_exp_bkg, int *num_bins, float *bin_edges)
{
    //printf("hello\\n");
    
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    float f_tot_num_e;
    int pe_from_first_dynode;
    int num_fully_amplified;
    int num_under_amplified;
    int i_tot_num_pe;
    
    int repetition_number;
    
    if (iteration < *num_trials)
	{
    
        for (repetition_number=0; repetition_number < *num_loops; repetition_number++)
        {
            //printf("hello\\n");
            
            if (*prob_not_under_amp < 0 || *prob_not_under_amp > 1)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            
            i_tot_num_pe = curand_poisson(&s, *mean_num_pe);
            num_under_amplified = gpu_binomial(&s, i_tot_num_pe, 1.-*prob_not_under_amp);
            num_fully_amplified = i_tot_num_pe - num_under_amplified;
    
            if (*spe_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_e = (curand_normal(&s) * *spe_std*powf(num_fully_amplified, 0.5)) + *spe_mean*num_fully_amplified;
            
            if (*under_amp_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_e += (curand_normal(&s) * *under_amp_std*powf(num_under_amplified, 0.5)) + *under_amp_mean*num_under_amplified;
    
            
            if (*bkg_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_e += (curand_normal(&s) * *bkg_std) + *bkg_mean;
            
            
            if (*bkg_exp < 0)
            {
                state[iteration] = s;
                return;
            }
            if(curand_uniform(&s) < *prob_exp_bkg)
                f_tot_num_e += gpu_exponential(&s, *bkg_exp, *bkg_mean);
            
            
            bin_number = gpu_find_lower_bound(num_bins, bin_edges, f_tot_num_e);
            
            if (bin_number == -1)
            {
                state[iteration] = s;
                continue;
                //return;
            }
            
            atomicAdd(&a_hist[bin_number], 1);
            
            state[iteration] = s;
            //printf("hi: %f\\n", f_tot_num_pe);
        }
        
        
		return;
        
        
        
    }
    


}


__global__ void pure_gaussian_spectrum(curandState *state, int *num_trials, float *a_hist, int *num_pe, float *spe_mean, float *spe_std, int *num_bins, float *bin_edges)
{
    //printf("hello\\n");
    
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    float f_tot_num_e;
    int i_tot_num_pe = *num_pe;
    
    
    if (iteration < *num_trials)
	{

        if (*spe_std < 0)
		{	
			state[iteration] = s;
			return;
		}
        
        if (i_tot_num_pe > 0)
        {
            f_tot_num_e = (curand_normal(&s) * *spe_std*powf(i_tot_num_pe, 0.5)) + *spe_mean*i_tot_num_pe;
        }
        
        bin_number = gpu_find_lower_bound(num_bins, bin_edges, f_tot_num_e);
		
		if (bin_number == -1)
		{
			state[iteration] = s;
			return;
		}
		
		atomicAdd(&a_hist[bin_number], 1);
		
		state[iteration] = s;
        //printf("hi: %f\\n", f_tot_num_e);
		return;
        
        
        
    }


}




__global__ void fixed_pe_gaussian_spectrum(curandState *state, int *num_trials, int *num_loops, float *a_hist, int *num_pe, float *prob_not_under_amp, float *spe_mean, float *spe_std, float *under_amp_mean, float *under_amp_std, float *bkg_mean, float *bkg_std, float *bkg_exp, float *prob_exp_bkg, int *num_bins, float *bin_edges)
{

    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s = state[iteration];
    
    int bin_number;
    const int fixed_num_pe = *num_pe;
    float f_tot_num_e;
    int num_fully_amplified;
    int num_under_amplified;
    int i_tot_num_pe;
    
    int num_electrons_leaving_dynode;
    
    int repetition_number;
    
    if (iteration < *num_trials)
	{
    
        for (repetition_number=0; repetition_number < *num_loops; repetition_number++)
        {
            //printf("hello\\n");
            
            if (*prob_not_under_amp < 0 || *prob_not_under_amp > 1)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            
            i_tot_num_pe = fixed_num_pe;
            num_under_amplified = gpu_binomial(&s, i_tot_num_pe, 1.-*prob_not_under_amp);
            num_fully_amplified = i_tot_num_pe - num_under_amplified;
    
            if (*spe_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_e = (curand_normal(&s) * *spe_std*powf(num_fully_amplified, 0.5)) + *spe_mean*num_fully_amplified;
            
            if (*under_amp_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_e += (curand_normal(&s) * *under_amp_std*powf(num_under_amplified, 0.5)) + *under_amp_mean*num_under_amplified;
    
            
            if (*bkg_std < 0)
            {	
                state[iteration] = s;
                continue;
                //return;
            }
            f_tot_num_e += (curand_normal(&s) * *bkg_std) + *bkg_mean;
            
            
            if (*bkg_exp < 0)
            {
                state[iteration] = s;
                return;
            }
            if(curand_uniform(&s) < *prob_exp_bkg)
                f_tot_num_e += gpu_exponential(&s, *bkg_exp, *bkg_mean);
            
            
            bin_number = gpu_find_lower_bound(num_bins, bin_edges, f_tot_num_e);
            
            if (bin_number == -1)
            {
                state[iteration] = s;
                continue;
                //return;
            }
            
            atomicAdd(&a_hist[bin_number], 1);
            
            state[iteration] = s;
            //printf("hi: %f\\n", f_tot_num_e);
        }
        
		return;
        
    }

}




// final close
}

"""