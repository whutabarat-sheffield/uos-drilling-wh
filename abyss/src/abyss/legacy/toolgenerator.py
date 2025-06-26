import numpy as np
import warnings

class IdealToolGenerator:
    '''
        David Miller 2022, University of Sheffield
    
        Ideal tool signal generator class

        It's purpose is for generating ideal tool signals to compare against the recorded ones. Several key parameters
        based on program parameters and tool geomoetry.

        Feed rate is how fast the tool is pushed forward. The sample rate is how fast the data is captured. Affects how dense
        the generated signal is.

        Some artificial parameters are used to affect how fast the torque and thrust change. These are not from the tool program
        as they depend on the materials being driven through. The rate of change for thrust and torque is how much the signal changes
        from one sample to the next.

        The settings for torque and thrust generation can be set either two ways; per sample change or target. If sample change,
        the torque and thrust increase at the specified rate.

        If the user gives a single value, it is the abs value the torque/thrust changes by between samples.

        If the user gives a list of values, they are treated as as the target for each tool shoulder.
        It must be the same number as the number of tool shoulders given in update_geometry.

        The torque and thrust alternates between +ve and 0 change. When the slope is 0, the value is held at the last value irrespective
        of the set target.

        Example usage - Constant slope
        ----------------------------
        # create tool class used fixed per-sample change
        tool = IdealToolGenerator(sr = 100.0, # Hz
                                  fr = 5.0,   # mm/s
                                  torque_slope = 0.1, # A/sample
                                  thrust_slope = 0.1) # A/ssample
        # set the tool geometry lengths
        geo = [1.031,2.800,0.572]
        tool.update_geometry(geo)
        # rebuild signal using the settings
        last = tool.rebuild()

        Example usage - Targets
        ----------------------------
        # set the tool geometry lengths
        geo = [1.031,2.800,0.572]
        # torque and thrust targets
        torque_tg = [10,10,20]
        thrust_tg = [5,5,15]
        # create tool class used fixed per-sample change
        tool = IdealToolGenerator(sr = 100.0, # Hz
                                  fr = 5.0,   # mm/s
                                  torque_targets = torque_tg, # A/sample
                                  thrust_targets = thrust_tg) # A/ssample
        tool.update_geometry(geo)
        # rebuild signal using the settings
        last = tool.rebuild()
    '''
    def __init__(self,sr=100.0,fr=5.0,**kwargs):
        '''
            Construct the tool and set the key settings for the generator.

            The sample rate (sr) is rate at which the data would be captured if the signal was real.

            The feed rate (fr) is the rate at which the tool is pushed forward into the material stack.

            If the user specified torque/thrust targets via *_slope keywords, then the values are changed by the set value between
            samples. e.g. if its set to 0.1, then over 100 samples the variable is increased by 10 over the period.

            If the user specified torque/thrust targets via *_targets keywords, then the values are calculated as going from
            last to target value. As the rate of change alternates between +ve and flat, if it's a flat section the value is held
            at the last value and is not set to reach the target.

            Inputs:
                sr : Sampling rate of the signal Hz. Default 100.0.
                fr : Feed rate of the tool in mm/s. Default 5.0.
                **kwargs:
                    torque_slope : Torque change between samples. Can be a single value or a list of values the same length as number of tool lengths.
                                Default 0.1.
                    thrust_slope : Thrust change between samples. Can be a single value or a list of values the same length as number of tool lengths.
                                Default 0.1.
                    torque_targets : Target torque values to aim for.
                    thrust_targets : Target thrust values to aim for.
        '''
        # sampling rate, Hz
        self.__sr = sr
        self.__T = 1/sr
        # feed rate, mm/s
        self.__fr = 5.0
        # tool list of tool length sections
        self.__toolL = []
        # generated signal
        self.__signal = []
        # distance travelled per time sample based on feed rate
        self.__frp = self.__fr * self.__T
        # last tool run
        self.__last = {}
        # set initial mode to constant slope
        self.__mode = 'slope'
        # if the user has specified targets instead
        if ("torque_targets" in kwargs) and ("thrust_targets" in kwargs):
            # save values
            self.__tqtarget = kwargs["torque_targets"]
            self.__tsttarget = kwargs["thrust_targets"]
            # update mode
            self.__mode = 'targets'
        # if the user has NOT specified targets
        # then default to constant slope
        else:
            # check that the target slopes aren't 0 or negative.
            # need +ve torque and thrust to cut through materials
            slope = kwargs.get("torque_slope",0.1)
            if slope <= 0:
                raise ValueError(f"The torque slope cannot be zero or negative! Received {slope}")
            self.__tqslope = slope
            # check thrust
            slope = kwargs.get("thrust_slope",0.1)
            if slope <= 0:
                raise ValueError(f"The thrust slope cannot be zero or negative! Received {slope}")
            self.__tstslope = slope
    # change the tool geometry
    # chronological order of tool length sections
    def update_geomoetry(self,geometry):
        '''
            Update the tool geometry set.

            Inputs:
                geomoetry : List of tool shoulder lengths in mm.
        '''
        if not geometry:
            raise ValueError("List of tool shoulder lengths cannot be empty!")
        elif self.__mode is "targets":
            if len(geometry) != len(self.__tqtarget):
                raise ValueError("The number of tool lengths does not match the number of torque targets!")
            elif len(geometry) != len(self.__tsttarget):
                raise ValueError("The number of tool lengths does not match the number of thrust targets!")
        self.__toolL = geometry
    # rebuild the tool signal from current settings
    def rebuild(self):
        '''
            Construct the ideal tool signals using the current settings

            Returns dictionary of values constructed
        '''
        # empty list of position values
        pos = np.empty(0)
        time = np.empty(0)
        thrust = np.empty(0)
        torque = np.empty(0)
        # last position value
        last = 0.0
        last_tq = 0.0
        # iterate over tool geometry
        for i,ll in enumerate(self.__toolL,start=1):
            # calculate number of samples to cover the distance based on feed rate
            # distance / feed rate => time (s)
            # time (s) * sr (samples/s) => num samples
            L = int(self.__sr*(ll/self.__fr))
            # create dist vector
            dist = np.linspace(last,last+ll,L)
            # create time vector
            tt = dist/self.__fr
            # update last time position value
            last = dist[-1]+self.__frp
            # if the gen mode is slope
            if self.__mode is 'slope':
                tq = []
                # if the user has given a single value
                if type(self.__tqslope,float):
                    slope = ((i%2)>0)*self.__tqslope
                # if the user has given an iterable list of values instead
                else:
                    slope = ((i%2)>0)*self.__tqslope[i-1]
                # iterate over samples
                for _ in range(L):
                    # if not the first value in local list
                    # use last appended value
                    if li>0:
                        tq.append(tq[-1]+slope)
                    # use last value from global list
                    elif len(torque)>0:
                        tq.append(torque[-1]+slope)
                    else:
                        tq.append(0.0) 
                # create thrust vector
                tst = []
                # if the user has given a single value
                if type(self.__tstslope,float):
                    slope = ((i%2)>0)*self.__tstslope
                # if the user has given an iterable list of values instead
                else:
                    slope = ((i%2)>0)*self.__tstslope[i-1]
                # iterate over number of samples
                for _ in range(L):
                    # if not the first value in local list
                    # use last appended value
                    if li>0:
                        tst.append(tst[-1]+slope)
                    # use last value from global list
                    elif len(thrust)>0:
                        tst.append(thrust[-1]+slope)
                    else:
                        tst.append(0.0)
            # if the gen mode is targets
            elif self.__mode is 'targets':
                start = 0.0
                if len(torque)>0:
                    start = torque[-1]
                # if it's a positive slope
                # generate values going from last value to target
                if ((i%2)>0):
                    tq = np.linspace(start,self.__tqtarget[i-1],L)
                # if it's a flat section
                # generate a vector of the current value
                else:
                    tq = start*np.ones(L)
                start = 0.0
                if len(thrust)>0:
                    start = thrust[-1]
                # if it's a positive slope
                # generate values going from last value to target
                if ((i%2)>0):
                    tst = np.linspace(start,self.__tsttarget[i-1],L)
                # if it's a flat section
                # generate a vector of the current value
                else:
                    tst = start*np.ones(L)
            # append to vectors
            pos = np.append(pos,dist)
            time = np.append(time,tt)
            torque = np.append(torque,tq)
            thrust = np.append(thrust,tst)
        # update dict of last values
        self.__last = {"time":time,"pos":pos,"thrust":thrust,"torque":torque}
        return self.__last
    # return results of last run
    def last(self):
        '''
            Get data from last rebuild run
    
            Returns the dictionary of values construct last time rebuild as called.        
        '''
        return self.__last

def getIdealBPS(tool_lengths,**kwargs):
    '''
        Find the breakpoints for the ideal tool based on its geometry.

        These breakpoints are when the tool is going forward through the material

        The breakpoints can be calculated in terms of position or time. If position is wanted,
        then it is calculated by iterating over the tool lengths and setting the breakpoints as the
        current plus the sum of the previous ones. The returned vector is a time series
        of WHERE the changes occur.

        If the user wants it in terms of time, then they need to provide the feed rate. The tool lengths
        are then each converted to the amount of time it would take to drill it. The returned vector is a
        time series of WHEN the changes occur.

        Inputs:
            tool_lengths : Iterable of tool lengths in mm
            as_time : Flag to return breakpoints in terms of time
            fr : Feed rate of the tool in mm/s

        Returns list of breakpoints in chronological order. If as_time is False, then it is as what positions (mm)
        they occur at. If as_time is True, then it is at what time (s) it occurs at based on feed rate. Both have 0.0
        as the start value.
    '''
    if len(tool_lengths)<=2:
        raise ValueError(f"Number of tool lengths has to be at least 2! len={len(tool_lengths)}")
    # if user wants it as time instead of position
    if not kwargs.get('as_time',False):
        # set first breakpoints as 0 and length of first segment
        bps = [0.0,tool_lengths[0]]
        # iterate over remaining segments
        # append next breakpoint as sum of previous segment lengths and current tool length
        return [tl+sum(tool_lengths[:ti]) for ti,tl in enumerate(tool_lengths[1:],start=1)]
    # check for existence of feed rate
    if not ('fr' in kwargs):
        raise ValueError("Feed Rate required to convert position to time")
    # check that feed rate is positive
    fr = kwargs.get(fr,0.0)
    if fr <= 0.0:
        raise ValueError(fr"Feed Rate has to be greater than 0! Got {fr} mm/s")
    # initialize time vector to 0.0
    time = [0.0]
    # iterate over tool lengths
    for tl in tool_lengths:
        # convert tool length to amount of time it would take to cover
        t = fr/tl
        # append time to vector and offset by previous value
        time.append(t+time[-1])
    # return vector
    return time

def getIdealFullBPS(tool_lengths,**kwargs):
    '''
        Find the breakpoints for the ideal tool based on its geometry.

        These breakpoints are when the tool is going forward AND backwards through the material

        The breakpoints can be calculated in terms of position or time. If position is wanted,
        then it is calculated by iterating over the tool lengths and setting the breakpoints as the
        current plus the sum of the previous ones. The returned vector is a time series
        of WHERE the changes occur.

        If the user wants it in terms of time, then they need to provide the feed rate. The tool lengths
        are then each converted to the amount of time it would take to drill it. The returned vector is a
        time series of WHEN the changes occur.

        Inputs:
            tool_lengths : Iterable of tool lengths in mm
            as_time : Flag to return breakpoints in terms of time
            fr : Feed rate of the tool in mm/s

        Returns list of breakpoints in chronological order. If as_time is False, then it is as what positions (mm)
        they occur at. If as_time is True, then it is at what time (s) it occurs at based on feed rate. Both have 0.0
        as the start value.
    '''
    # get forward half
    bps = getIdealBPS(tool_lengths,**kwargs)
    # get backards half supplying the tool lengths in reverse order as the tool is pulling back
    bps_back = getIdealBPS(tool_lengths[::-1],**kwargs)
    # extend forward half with the backwards half offset by final distance
    return bps.extend([bb+dur for bb in bps_back])

if __name__ == "__main__":
    import dataparser as dp
    from glob import glob
    import matplotlib.pyplot as plt
    # Sampling period
    T = 1/1e2
    # load setitec spreadsheet
    data = dp.loadSetitecXls(list(glob("xls/UC*.xls"))[0])[-1]
    # get position data
    xdata = data['Position (mm)'].values
    xdata = np.abs(xdata)
    # get thrust data
    ydata = data['I Thrust (A)'].values.flatten()
    # construct time vector
    time = np.arange(0.0,len(ydata)*T,T,dtype='float16')
    # function to find closest ydata for the given x data
    def find_closest(xfind,xdata,ydata):
        return [ydata[np.abs(xdata-x).argmin()] for x in xfind]
    # distance the drill needs to travel to reach the first material
    AIR = 6.5
    # tool lengths (mm)
    TL_ogs = [1.192,2.9,0.561,15.93]
    # initially set tool lengths to original
    TL = TL_ogs.copy()
    # set current tool length
    TL_curr = TL.pop(0)
    # material thicknesses (mm)
    M = [10.5,10]
    # set current material
    M_curr = M.pop(0)
    # initialize BPS with starting at 0.0
    BPS = [0.0,]
    # amount of current material drilled (mm)
    md = 0.0
    # global amount of material drilled (mm)
    mdg = 0.0
    # feed rate mm/s
    fr = 5
    # create time vector
    time = np.arange(0.0,len(xdata)*T,T)
    # while there are materials to drill
    while len(M):
        # if the amount drilled is greater than or equal to the current material thickness
        if (md+TL_curr) >= M_curr:
            # add breakpoint where the material has been drilled
            BPS.append(abs(M_curr-(md+TL_curr)))
            # if the tool length precisely covers the remaining material
            if (md+TL_curr) == M_curr:
                # increment total material drilled by the tool length
                mdg += TL_curr
            # if the tool length will go into the next material
            else:
                # increment the total material drilled by the distance to drill the currrent material
                mdg += abs(M_curr-(md+TL_curr))
            # reset tool segments for next material
            # as we're starting from the tip of the tool
            TL = TL_ogs.copy()
            TL_curr = TL.pop(0)
            # reset amount of current material drilled
            md = 0.0
            # move onto next material
            M_curr = M.pop(0)
        # if the tool segment won't drill all the material
        else:
            # add a breakpoint for the distance drilled
            BPS.append(BPS[-1]+TL_curr)
            # move onto next tool segment
            if len(TL)>0:
                TL_curr = TL.pop(0)
            else:
                TL = TL_ogs.copy()
                TL_curr = TL.pop(0)
            # increment the amount drilled by the tool length
            md += TL_curr
            # increment global amount drilled by the tool length
            mdg += TL_curr
    TL = TL_ogs.copy()
    # increment over remaining tool lengths
    # append breakpoints
    while len(TL):
        BPS.append(BPS[-1]+TL_curr)
        TL_curr = TL.pop(0)
        
    print(len(BPS))
    # convert to an array and increment by the amount of air
    BPS = np.array(BPS)+6.9
    f,ax = plt.subplots()
    ax.plot(xdata,ydata,'b-',label="Original")
    ax.plot(BPS,find_closest(BPS,xdata,ydata),'rx',markersize=12,label="Synthetic")
    plt.legend()
