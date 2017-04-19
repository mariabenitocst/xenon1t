#include <vector>
#include <functional>
#include <numeric>
#include "math.h"

// sometimes numpy dot product is not thread safe
// use this in its place if that is the case


/*
float log_likelihood_matching_uncertainty(float data, float scale, int num_mc_events)
{
	data = (int)round(data);
	float ln_likelihood = 0.;

	//ln_likelihood = -(data+1)*log(scale) - log(2) - lgamma(data+1) + log(pow((-1),data)-1+ ( 1 + pow(-1, 1+data) + pow(pow(scale-1,2)/pow(scale,2),0.5) + pow(-1, data)*pow(pow(scale-1,2)/pow(scale,2),0.5) )*scale ) + num_mc_events*(1-2*scale)/(2*pow(scale,2)) + (data+1)/2.*log(num_mc_events) + (data-1)/2.*log(num_mc_events*pow((1-scale),2)/pow(scale,2));
	
	ln_likelihood = (1+num_mc_events)*log(scale) + (-1-data-num_mc_events)*log(1+scale) + lgamma(1+data+num_mc_events) - (data*log(data) - data) - (num_mc_events*log(num_mc_events) - num_mc_events);
	
	return ln_likelihood;

}




float smart_log_likelihood_old(float *a_flat_data, float *a_flat_mc, int num_bins, int num_mc_events, float scale_const, int num_data_pts, float confidence_interval)
{
	// a_flat_mc[i] = mc_events_in_bin[i]*scale_const*num_data_pts/num_mc_events
	// a_flat_mc[i] = mc_events_in_bin[i] / scale
	float scale = num_mc_events / (float(num_data_pts)*scale_const);

	float total_log_likelihood = 0.;
	float binom_prob = 1. - pow((1.-confidence_interval),(1./num_mc_events));
	//printf("Beginning: %f\n", total_log_likelihood);
	//printf("lgamma(3): %f\n", lgamma(1));
	//printf("log(3): %f\n", log(3));
    
    
    
	
	for (int bin_number=0; bin_number < num_bins; bin_number++)
	{
		
		//if (a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events < 800) continue;
		//if (a_flat_data[bin_number] < 5) continue;
		
		if (a_flat_data[bin_number] != 0 && a_flat_mc[bin_number] == 0)
		{
			//continue;
			total_log_likelihood += lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob);
			//printf("No mc events: %f\n", lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob));
		}
		else if (a_flat_data[bin_number] == 0 && a_flat_mc[bin_number] == 0)
			continue;
		else
		{
			
			total_log_likelihood += a_flat_data[bin_number]*log(a_flat_mc[bin_number]) - a_flat_mc[bin_number] - lgamma(a_flat_data[bin_number]+1.0);
			//total_log_likelihood += log_likelihood_matching_uncertainty(a_flat_data[bin_number], scale, a_flat_mc[bin_number]*scale);
			//printf("\n\ndata:%f\nmc:%f\nscale:%f\nscaled mc:%f\n", a_flat_data[bin_number], a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events, num_mc_events/scale_normalized_to_mc_events, a_flat_mc[bin_number]);
			//printf("MC and data (normal likelihood): %f\n", a_flat_data[bin_number]*log(a_flat_mc[bin_number]) - a_flat_mc[bin_number] - lgamma(a_flat_data[bin_number]+1.0));
			//printf("MC and data (matching uncertainty): %f\n", a_flat_data[bin_number]*log(a_flat_mc[bin_number]) - a_flat_mc[bin_number] - lgamma(a_flat_data[bin_number]+1.0));
		}
	}
	//printf("%f", total_log_likelihood);
	return total_log_likelihood;
}



float log_likelihood_normalized_pdf(float *a_flat_data, float *a_flat_mc_normalized, int num_bins, int num_data_pts, float confidence_interval)
{
	float total_log_likelihood = 0.;
	//float binom_prob = 1. - pow((1.-confidence_interval),(1./num_mc_events));
	
	
	for (int bin_number=0; bin_number < num_bins; bin_number++)
	{
		
		//if (a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events < 800) continue;
		if (a_flat_data[bin_number] < 3) continue;
		
		if (a_flat_data[bin_number] != 0 && a_flat_mc_normalized[bin_number] == 0)
		{
			//continue;
			//total_log_likelihood += lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob);
			//printf("No mc events: %f\n", lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob));
		}
		else if (a_flat_data[bin_number] == 0 && a_flat_mc_normalized[bin_number] == 0)
			continue;
		else
		{
			
			total_log_likelihood += a_flat_data[bin_number]*log(a_flat_mc_normalized[bin_number]*num_data_pts) - a_flat_mc_normalized[bin_number]*num_data_pts - lgamma(a_flat_data[bin_number]+1.0);
			printf("\nbin %d:%f\n\n", bin_number, a_flat_data[bin_number]*log(a_flat_mc_normalized[bin_number]*num_data_pts) - a_flat_mc_normalized[bin_number]*num_data_pts - lgamma(a_flat_data[bin_number]+1.0));
		}
	}
	printf("tot ll matching: %f\n\n", total_log_likelihood);
	return total_log_likelihood;



}



//160402
float smart_log_likelihood(float *a_flat_data, float *a_flat_mc, int num_bins, int num_mc_events, float confidence_interval)
{

	float total_log_likelihood = 0.;
	float binom_prob = 1. - pow((1.-confidence_interval),(1./num_mc_events));
    
    // add small number to each mc bin
    // to avoid negative infinity
    float small_number = 0.001;
	
	for (int bin_number=0; bin_number < num_bins; bin_number++)
	{
		
		if (a_flat_data[bin_number] <= 3) continue;
		
		else if (a_flat_data[bin_number] != 0 && a_flat_mc[bin_number] == 0)
		{
            //continue;
			total_log_likelihood += lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob);
		}
        
		if (a_flat_data[bin_number] == 0 && a_flat_mc[bin_number] == 0)
			continue;
		else
		{
			
			total_log_likelihood += a_flat_data[bin_number]*log((a_flat_mc[bin_number]+small_number)) - (a_flat_mc[bin_number]+small_number) - lgamma(a_flat_data[bin_number]+1.0);
			
		}
	}
	return total_log_likelihood;
}

*/


float smart_log_likelihood(float *a_flat_data, float *a_flat_mc, int num_bins, float small_number)
{

	float total_log_likelihood = 0.;
    
    // add small number to each mc bin
    // to avoid negative infinity
    //float small_number = 0.001;
	
	for (int bin_number=0; bin_number < num_bins; bin_number++)
	{
    
        if (a_flat_data[bin_number] <= 1 && a_flat_mc[bin_number] < 0.5) continue;
        //if (a_flat_data[bin_number] <= 1) continue;
		
        total_log_likelihood += a_flat_data[bin_number]*log((a_flat_mc[bin_number]+small_number)) - (a_flat_mc[bin_number]+small_number) - lgamma(a_flat_data[bin_number]+1.0);
			
	}
	return total_log_likelihood;
}


