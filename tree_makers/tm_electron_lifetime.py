import numpy as np
import sys
import hax
from collections import defaultdict


import matplotlib   # Needed for font size spec, color map transformation function bla bla
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
matplotlib.rc('font', size=16)
plt.rcParams['figure.figsize'] = (12.0, 10.0) # resize plots

# relative gains
PMTGains = [4.903, 3.967, 4.579, 1.163, 2.606, 1.001, 4.048, 1.572, 1.426, 2.108,
                        1.754, -1.00, 1.013, 2.254, 1.196, 1.365, 2.144, 1.361, 1.770, -1.00, 
                        1.768, -1.00, 3.258, 2.400, 2.678, 2.327, 1.001, 1.915, 1.526, 1.652, 
                        3.699, 2.638, 2.870, 2.788, 1.726, 1.992, 1.002, 2.790, 2.720, 2.143, 
                        2.493, 2.245, 2.573, 1.286, 1.874, -1.00, 1.224, 2.851, -1.00, 3.257, 
                        2.745, 2.643, 1.984, 1.315, 2.666, 0.989, 3.415, 2.125, -1.00, 2.345, 
                        2.054, 4.305, 2.849, 2.514, 2.251, 3.342, 2.386, 2.753, 3.945, 2.027, 
                        2.714, 2.331, 1.125, -1.00, 3.963, 3.422, 2.308, 2.331, 2.733, 2.614, 
                        2.378, 3.396, 3.058, 1.263, 1.885, 2.915, 1.328, 4.453, 2.249, 1.695, 
                        1.445, 2.141, 2.441, 2.479, 2.068, 2.620, -1.00, 2.312, 1.253, 1.345, 
                        1.494, 2.019, 2.286, 2.023, 3.002, 2.330, 2.720, 1.594, 3.753, 1.374, 
                        1.225, 2.635, 1.225, 1.993, 2.117, 0.998, 2.262, -1.00, -1.00, 1.789, 
                        1.704, 3.326, 1.219, 1.350, 1.470, 1.232, 3.989, 3.260, 1.043, 1.778, 
                        2.592, 2.210, 4.537, 2.265, 1.928, 2.905, 2.397, 2.256, 2.884, 1.422, 
                        4.332, 2.110, 3.048, 2.328, 1.778, 2.437, 2.592, 6.996, -1.00, 1.414, 
                        -1.00, 2.529, 3.554, 2.332, 1.923, 1.423, 2.278, 1.301, 2.836, 2.858, 
                        1.376, 1.522, 1.438, 1.081, 1.125, 1.471, 1.456, 1.062, 1.283, 2.988, 
                        4.634, 2.002, 2.495, 1.413, 0.978, 2.480, 1.448, 1.853, 3.143, 1.599, 
                        3.057, 3.788, 1.983, 4.082, 1.042, 2.966, 2.321, 2.544, 1.041, 1.017, 
                        3.033, 1.813, -1.00, 4.467, 2.127, 1.146, 1.047, 0.948, 3.131, 2.138, 
                        1.038, 2.335, 1.190, 3.206, 2.499, 2.003, 3.793, 2.151, 1.466, -1.00, 
                        2.142, 2.566, 2.927, 1.841, 1.411, 4.033, 2.652, 1.473, 3.466, 3.265, 
                        1.469, 3.662, 2.353, 3.430, 2.070, 2.986, 2.979, 1.861, 1.821, 4.474, 
                        1.137, 2.388, 1.603, 1.844, -1.00, 3.549, 1.520, 1.190, 5.057, 1.687, 
                        1.523, 2.779, 3.623, 2.807, 0.767, 2.860, 2.361, 3.112, 1.125, 1.020, 
                        1.050, 0.842, 0.850, 0.928
                       ]


WidthThreshold = 0.5 # us on the half

# Tell 'hax' that we're analyzing XENON1T data and password for run
# database (see https://xenon1t-daq.lngs.infn.it/runs).
# hax.init(experiment='XENON1T')

# my own builder
from collections import defaultdict

class tm_electron_lifetime(hax.minitrees.TreeMaker):
    """Computing properties of the S1
    
    This TreeMaker will take the event class and turn it into a row
    in a table (e.g. TNtuple or pandas DataFrame).  We define only
    one function, which takes a pax event in.  It returns a dictionary
    of new variables and their values.
    """
    
    extra_branches = ['*']  # Activate all of ROOT file
    __version__ = '0.0.1' 

    # basically at the commissioning stage
    # I just use the peak width to distinguish between S1 and S2
    def find_first_two_largest_s1(self, event):
        peaks = event.peaks
        largest_id = -1
        second_id = -1
        largest_area = 0
        second_area = 0
        for ID, peak in enumerate(peaks):
            if peak.hit_time_std > WidthThreshold*1000.:
                continue
            if peak.area < largest_area and peak.area < second_area:
                continue
            elif peak.area > largest_area:
                second_area = largest_area
                second_id = largest_id
                largest_area = peak.area
                largest_id = ID
            else:
                second_area = peak.area
                second_id = ID
        return (largest_id, second_id)

    def find_first_two_largest_s2(self, event):
        peaks = event.peaks
        largest_id = -1
        second_id = -1
        largest_area = 0
        second_area = 0
        for ID, peak in enumerate(peaks):
            if peak.hit_time_std < WidthThreshold*1000.:
                continue
            if peak.area < largest_area and peak.area < second_area:
                continue
            elif peak.area > largest_area:
                second_area = largest_area
                second_id = largest_id
                largest_area = peak.area
                largest_id = ID
            else:
                second_area = peak.area
                second_id = ID
        return (largest_id, second_id)

    def extract_data(self, event):  # This runs on each event
        # 'values' is returned once filled and each field defaults to zero.
        values = defaultdict(float)

        # Store the start time of the event
        values['time'] = event.start_time

        #total peak numbers
        values['NbPeaks'] = len(event.peaks)
        if len(event.peaks) == 0:
            return values

        # find the largest and second largest peak
        # in regardless of the peak type
        #largest_s1_id, second_s1_id = self.find_first_two_largest_s1(event)        
        #largest_s2_id, second_s2_id = self.find_first_two_largest_s2(event) 
        
        # get the largesst and second largest peak from pax classification
        s1_ids = event.s1s
        s2_ids = event.s2s
        values['NbS1s'] = int(len(s1_ids))
        values['NbS2s'] = int(len(s2_ids))
        largest_s1_id = -1
        largest_s2_id = -1
        second_s1_id = -1
        second_s2_id = -1
        if len(s1_ids)>0:
            largest_s1_id = s1_ids[0]
        elif len(s1_ids)>1:
            second_s1_id = s1_ids[1]
        if len(s2_ids)>0:
            largest_s2_id = s2_ids[0]
        elif len(s2_ids)>1:
            second_s2_id = s2_ids[1]

        # These are the peak properties that I'm interested in looking at.
        # Look here for more info: http://xenon1t.github.io/pax/format.html#peak
        s1_fields = {'S1sTot': 'area',
                        'S1TopFraction': 'area_fraction_top',
                        'S1sPeakTime': 'area_midpoint',
                        'S1sPeakTimeStd': 'hit_time_std',
                        'S1sCoin': 'n_contributing_channels',
                        'S1sHeight': 'height',
                        'S1sNbSaturationChannels': 'n_saturated_channels'
                      }
        s2_fields = {'S2sTot': 'area',
                        'S2TopFraction': 'area_fraction_top',
                        'S2sPeakTime': 'area_midpoint',
                        'S2sPeakTimeStd': 'hit_time_std',
                        'S2sCoin': 'n_contributing_channels',
                        'S2sHeight': 'height',
                        'S2sNbSaturationChannels': 'n_saturated_channels'
                      }

        # Grab the biggest S1&S2 from the list of peaks
        peaks = event.peaks
        s1peak = event.peaks[0]
        if not largest_s1_id==-1:
            s1peak = event.peaks[largest_s1_id]
        s2peak = event.peaks[0]
        if not largest_s2_id==-1:
            s2peak = event.peaks[largest_s2_id]
        # The store each peak field we want in 'values'
        if not largest_s1_id==-1:
            for s1_field in s1_fields:
                values[s1_field] = getattr(s1peak,
                                        s1_fields[s1_field])
        if not largest_s2_id==-1:
            for s2_field in s2_fields:
                values[s2_field] = getattr(s2peak,
                                        s2_fields[s2_field])
        # Grab the second biggest S1 if it exists
        values['S1sTotSecond']=0
        if not second_s1_id==-1:
            values['S1sTotSecond'] = peaks[second_s1_id].area
        values['S2sTotSecond']=0
        if not second_s2_id==-1:
            values['S2sTotSecond'] = peaks[second_s2_id].area
        # get the widths
        values['S1sWidth'] = s1peak.range_area_decile[5]
        values['S1sLowWidth'] = s1peak.range_area_decile[9]
        values['S2sWidth'] = s2peak.range_area_decile[5]
        values['S2sLowWidth'] = s2peak.range_area_decile[9]
        # loop over to get X-Y of the largest
        PosRecAlgorithms = [
                        'WeightedSum.PosRecWeightedSum',
                        'MaxPMT.PosRecMaxPMT',
                        'RobustWeightedMean.PosRecRobustWeightedMean',
                        'NeuralNet.PosRecNeuralNet',
                        'TopPatternFit.PosRecTopPatternFit',
                        'HitpatternSpread.HitpatternSpread'
                     ]
        # get S1 reconstructed position
        ReconstructedPositions = s1peak.reconstructed_positions
        for reconstructed_position in ReconstructedPositions:
            CurrentAlgorithm = reconstructed_position.algorithm
            NameX = "S1_X_"+CurrentAlgorithm
            NameY = "S1_Y_"+CurrentAlgorithm
            values[NameX]=reconstructed_position.x
            values[NameY]=reconstructed_position.y
        # get S2 reconstructed position
        ReconstructedPositions = s2peak.reconstructed_positions
        for reconstructed_position in ReconstructedPositions:
            CurrentAlgorithm = reconstructed_position.algorithm
            NameX = "S2_X_"+CurrentAlgorithm
            NameY = "S2_Y_"+CurrentAlgorithm
            values[NameX]=reconstructed_position.x
            values[NameY]=reconstructed_position.y
        # get the PMT gain balanced S1
        values['S1sTotGained'] = -1
        if not largest_s1_id==-1:
            Sum = 0.
            # largest peak first
            for ID, area in enumerate(s1peak.area_per_channel):
                # if ID>247, not in TPC
                if ID>247:
                    continue
                if PMTGains[ID]==-1 or PMTGains[ID]==0:
                    continue
                Sum = Sum + area/PMTGains[ID]
            if Sum>0:
                values['S1sTotGained'] = Sum
        # then the second largest
        values['S1sTotSecondGained'] = -1
        if not second_s1_id==-1:
            Sum=0
            second_peak = peaks[second_s1_id]
            for ID, area in enumerate(second_peak.area_per_channel):
                # if ID>247, not in TPC
                if ID>247:
                    continue
                if PMTGains[ID]==-1 or PMTGains[ID]==0:
                    continue
                Sum = Sum + area/PMTGains[ID]
            if Sum>0:
                values['S1sTotSecondGained']=Sum

        # get the PMT gain balanced S2
        values['S2sTotGained'] = -1
        if not largest_s2_id==-1:
            Sum = 0.
            # largest peak first
            for ID, area in enumerate(s2peak.area_per_channel):
                # if ID>247, not in TPC
                if ID>247:
                    continue
                if PMTGains[ID]==-1 or PMTGains[ID]==0:
                    continue
                Sum = Sum + area*2.0/PMTGains[ID]
            if Sum>0:
                values['S2sTotGained'] = Sum
        # then the second largest
        values['S2sTotSecondGained'] = -1
        if not second_s2_id==-1:
            Sum=0
            second_peak = peaks[second_s2_id]
            for ID, area in enumerate(second_peak.area_per_channel):
                # if ID>247, not in TPC
                if ID>247:
                    continue
                if PMTGains[ID]==-1 or PMTGains[ID]==0:
                    continue
                Sum = Sum + area*2.0/PMTGains[ID]
            if Sum>0:
                values['S2sTotSecondGained']=Sum
    
 
        return values

     
