##############
# General python modules
import sys
from mpi4py import MPI
#import matplotlib.pyplot as plt
#plt.ion()
import numpy as np
import time
import platform
import os
import sys
from collections import deque
import importlib
import os.path
import lmfit
from scipy.stats import binned_statistic


#############
# LCLS modules
import psana


############
# Custom modules
# Soume found in:
sys.path.append( os.path.dirname(os.path.abspath(__file__)) +  '/aolPyModules')
import cookieBox
import tof
import aolUtil
import lcls
import arguments

# Set up the mpi cpmmunication
world = MPI.COMM_WORLD
rank = world.Get_rank()
worldSize = world.Get_size()

c_0_mm_per_fs = 2.99792458e8 * 1e3 * 1e-15
# http://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=universal_in! 2014-04-21

#############
# Data definitions
s = 0 # Data size tracker

dRank = s
s += 1
dFiducials = s
s += 1
dTime = s
s += 1
dFull = slice(s, s+16)
s += 16
dIntRoi0 = slice(s, s+16)
s += 16
dIntRoi1 = slice(s, s+16)
s += 16
dPol = slice(s, s+8)
s += 8
dEnergy = slice(s, s+2)
s += 2
dEL3 = s
s += 1
dFEE = slice(s, s+4)
s += 4
#dDeltaK = s
#s += 1
#dDeltaEnc = slice(s, s+4)
#s += 4
dDelayStage = s
s += 1
dFsTiming = s
s += 1
dTtTime = s
s += 1
nEvr = 10
dEvr = slice(s, s+nEvr)
s += nEvr

dSize = s


def connectToDataSource(args, config, verbose=False):
    # If online
    if not args.offline:
        # make the shared memory string
        dataSource = 'shmem=AMO.0:stop=no'
    else:
        dataSource = config.offlineSource
        #config.makeTofConfigList(online=False)
        if verbose:
            print config

    if verbose:
        # check the host name
        host = platform.node()
        print 'rank {} (on {}) connecting to datasource: {}'.format(
                rank,
                host,
                dataSource)

    return psana.DataSource(dataSource)


def importConfiguration(args, verbose=False):
    # Import the correct configuration module    
    confPath, confFileName = os.path.split(args.configuration)
    sys.path.append(confPath)
    if verbose:
        print 'Loading configuration from directory "{}"'.format(confPath)
        print 'File name is {}'.format(confFileName)
    confName, _ = os.path.splitext(confFileName)
    if verbose:
        print 'Module name is {}'.format(confName)
    
    return importlib.import_module(confName)
    

def getDetectorCalibration(verbose=False, fileName=''):
    if fileName == '':
        detCalib = aolUtil.struct({'path':'detCalib',
            'name':'calib'})
        # Get the latest detector callibration values from file
        if not os.path.exists(detCalib.path):
            os.makedirs(detCalib.path)
        if not os.path.exists(detCalib.path + '/' + detCalib.name + '0.txt'):
            np.savetxt(detCalib.path + '/' + detCalib.name + '0.txt', [1]*16)
        
        detCalib.fileNumber = np.max([int(f[len(detCalib.name):-4]) for f in
            os.listdir(detCalib.path) if len(f) > len(detCalib.name) and
            f[:len(detCalib.name)]==detCalib.name])
    else:
        detCalib = aolUtil.struct()
        splitPath = fileName.split('/')
        if len(splitPath) > 1:
            detCalib.path = '/'.join(splitPath[:-1])
        else:
            detCalib.path = '.'
        detCalib.name = splitPath[-1]
        detCalib.fileNumber = np.nan


    if args.calibrate == -1:
        detCalib.factors = np.loadtxt(detCalib.path + '/' + detCalib.name +
                '{}.txt'.format( detCalib.fileNumber if
                    np.isfinite(detCalib.fileNumber) else '' ) ) 
    else:
        detCalib.factors = np.ones(16)
    if verbose:
        print 'Detector factors =', detCalib.factors


    return detCalib


def saveDetectorCalibration(masterLoop, detCalib, config, verbose=False, beta=0):
    calibValues = np.concatenate(masterLoop.calibValues, axis=0)
    average = calibValues.mean(axis=0)
    #factors = average[config.boolFitMask].max()/average
    params = cookieBox.initialParams()
    params['A'].value = 1
    params['beta'].value = beta
    params['tilt'].value = 0
    params['linear'].value = 1
    factors = cookieBox.modelFunction(params, np.radians(np.arange(0, 360,
        22.5))) * float(average.max()) / average
    factors[~config.boolFitMask] = np.nan
    
    if verbose:
        print len(calibValues)
        print masterLoop.calibValues[0].shape
        print calibValues.shape
        print average
        print 'Calibration factors:', factors

    calibFile = (detCalib.path + '/' + detCalib.name +
                    '{}.txt'.format( detCalib.fileNumber+1 if
                    np.isfinite(detCalib.fileNumber) else '' ) )

    np.savetxt(calibFile, factors)
          

def masterDataSetup(masterData):
    # Container for the master data
    masterData.energySignal_V = None
    masterData.timeSignals_V = None
    masterData.timeSignalseFiltered_V = None

    
def masterLoopSetup(args):
    # Master loop data collector
    masterLoop = aolUtil.struct()
    # Define the plot interval from the command line input
    masterLoop.tPlot = args.plotInterval

    # set up the buffer size to be able to handle twice the amoun of
    # data that is expected
    masterLoop.bufSize = int(np.round(120 * masterLoop.tPlot * 1.3))


    # Make template of the array that should be sent between the ranks
    masterLoop.bufTemplate = np.empty((1, dSize), dtype=float)
    
    # Make empty queues.
    masterLoop.req = deque()
    masterLoop.buf = deque()
        
    # Initialize the stop time
    masterLoop.tStop = time.time()
    
    # Calibration
    if args.calibrate > -1:
        masterLoop.calibValues = []
    
    return masterLoop


def getScales(env, config, verbose=False):
    scales = aolUtil.struct()
    # Grab the relevant scales
    scales.energy_eV = (config.energyScaleBinLimits[:-1] +
            np.diff(config.energyScaleBinLimits)/2)
    scales.eRoi0S = slice(
            scales.energy_eV.searchsorted(np.min(config.energyRoi0_eV_common)),
            scales.energy_eV.searchsorted(np.max(config.energyRoi0_eV_common)))
    scales.energyRoi0_eV = scales.energy_eV[scales.eRoi0S]
    
    tempTime = tof.getTimeScale_us(env, config.cbSourceString, verbose=verbose)
    scales.baselineSlice = slice( tempTime.searchsorted(config.baselineEnd_us) )
    scales.timeSlice = slice(
            tempTime.searchsorted(config.tMin_us),
            tempTime.searchsorted(config.tMax_us) )
    scales.time_us = tempTime[ scales.timeSlice ]
    scales.tRoi0S = cookieBox.sliceFromRange(scales.time_us,
            [config.timeRoi0_us_common]*16)
    scales.tRoi0BgS = cookieBox.sliceFromRange(scales.time_us,
            [config.timeRoi0Bg_us_common]*16)
    scales.tRoi0BgFactors = np.array([np.float(s.stop-s.start)/(bg.stop-bg.start) for
        s,bg in zip(scales.tRoi0S, scales.tRoi0BgS)])
    scales.tRoi1S = cookieBox.sliceFromRange(scales.time_us,
            [config.timeRoi1_us_common]*16)
    scales.timeRoi0_us = [scales.time_us[s] for s in scales.tRoi0S]
    scales.timeRoiBg0_us = [scales.time_us[s] for s in scales.tRoi0BgS]
    scales.timeRoi1_us = [scales.time_us[s] for s in scales.tRoi1S]
    scales.angles = cookieBox.phiRad
    scales.anglesFit = np.linspace(0, 2*np.pi, 100)

    scales.scaling_VperCh, scales.offset_V = cookieBox.getSignalScaling(env, config.cbSourceString,
            verbose=verbose)

    return scales

def setupRecives(masterLoop, verbose=False): 
    # Set up the requests for recieving data
    while len(masterLoop.buf) < masterLoop.bufSize:
        masterLoop.buf.append(masterLoop.bufTemplate.copy())
        masterLoop.req.append(world.Irecv([masterLoop.buf[-1], MPI.FLOAT],
            source=MPI.ANY_SOURCE))
    if verbose:
        print 'master set up for', len(masterLoop.buf), 'non blocking recives'
    # Counter for the processed evets
    masterLoop.nProcessed = 0

def eventDataContainer(args):
    # Set up some data containers
    event = aolUtil.struct()
    event.sender = []
    event.fiducials = []
    event.times = []
    event.full = []
    event.intRoi0 = []
    event.intRoi0Bg = []
    event.intRoi1 = []
    event.pol = []
    #event.positions = []

    event.ebEnergyL3 = []
    event.gasDet = []

    if args.photonEnergy != 'no':
        event.energy = []

    event.evrCodes = []

    #event.deltaK = []
    #event.deltaEnc = []
    event.delayStage = []
    event.fsTiming = []
    event.ttTime = []

    return event

def appendEventData(evt, evtData, config, scales, detCalib, masterLoop,
        verbose=False):
    evtData.sender.append(rank)

    timeAmplitudeRaw = cookieBox.getRawSignals(evt, config.cbSourceString,
            verbose=False)

    if timeAmplitudeRaw is None:
        timeAmplitudeRaw = np.zeros((16,1))
        evtData.timeSignals_V = np.zeros((16, len(scales.time_us)))
    else:
        evtData.timeSignals_V = -(scales.scaling_VperCh *
                timeAmplitudeRaw[:,scales.timeSlice].T - scales.offset_V).T

    if config.baselineSubtraction == 'early':
        bg = -(scales.scaling_VperCh * 
                timeAmplitudeRaw[:,scales.baselineSlice].T -
                scales.offset_V).mean(axis=0)
    else:
        bg = np.zeros(16)

    evtData.timeSignals_V = (evtData.timeSignals_V.T - bg).T
 
    if rank == 0:
        # Grab the y data
        if 0:
            #evtData.timeSignalsFiltered_V = cb.getTimeAmplitudesFiltered()
            pass
        else:
            evtData.timeSignalsFiltered_V = evtData.timeSignals_V
        evtData.timeAmplitudeRoi0 = [t[s] for t, s in
                zip(evtData.timeSignalsFiltered_V, scales.tRoi0S)]
        evtData.timeAmplitudeRoi1 = [t[s] for t, s in
                zip(evtData.timeSignalsFiltered_V, scales.tRoi1S)]

        #evtData.energyAmplitude = np.average(cb.getEnergyAmplitudes(), axis=0)
        #evtData.energySignal = np.array(cb.getEnergyAmplitudes()[2])
        #evtData.energySignalRoi0 = \
        #        evtData.energySignal[scales.eRoi0S]

        if verbose:
            print 'Rank', rank, '(master) grabbed one event.'
        # Update the event counters
        masterLoop.nProcessed += 1

    evtData.full.append(
            np.array([ sig.sum() for sig in evtData.timeSignals_V ]) *
            detCalib.factors * config.nanFitMask)


    # Get the intensities
    evtData.intRoi0.append(
            np.array([ sig[sl].sum() - sig[slBg].sum() * fact
                for sig, sl, slBg, fact in zip(
                    evtData.timeSignals_V,
                    scales.tRoi0S,
                    scales.tRoi0BgS,
                    scales.tRoi0BgFactors) ]) * detCalib.factors * config.nanFitMask)

    
    evtData.intRoi1.append(
            np.array([ sig[sl].sum() for sig, sl in zip(
                evtData.timeSignals_V,
                scales.tRoi1S) ]) * detCalib.factors)

    # Get the initial fit parameters
    #params = cookieBox.initialParams(evtData.intRoi0[-1])
    params = cookieBox.initialParams()
    params['A'].value, params['linear'].value, params['tilt'].value = \
            cookieBox.proj.solve(evtData.intRoi0[-1], args.beta)
    # Lock the beta parameter. To the command line?
    params['beta'].value = args.beta
    params['beta'].vary = False

    #print params['A'].value, params['linear'].value, params['tilt'].value
    
    # Perform the fit
    #print scales.angles[config.boolFitMask]
    #print evtData.intRoi0[-1][config.boolFitMask]
    res = lmfit.minimize(
            cookieBox.modelFunction,
            params,
            args=(scales.angles[config.boolFitMask],
                evtData.intRoi0[-1][config.boolFitMask]),
            method='leastsq')

    #print params['A'].value, params['linear'].value, params['tilt'].value
    
    #lmfit.report_fit(params)
                
    # Store the values
    evtData.pol.append(np.array( [
        params['A'].value, params['A'].stderr,
        params['beta'].value, params['beta'].stderr,
        params['tilt'].value, params['tilt'].stderr,
        params['linear'].value, params['linear'].stderr
        ]))

    # Beam position
    #evtData.positions.append( cb.getPositions() )
    
    # Get the photon energy center and width
    #if args.photonEnergy != 'no':
    #    evtData.energy.append(
    #            cb.getPhotonEnergy(
    #                energyShift=args.energyShift
    #                )
    #            )

    # Get lcls parameters
    lcls.setEvent(evt)
    evtData.ebEnergyL3.append(lcls.getEBeamEnergyL3_MeV())
    evtData.gasDet.append( np.array(lcls.getPulseEnergy_mJ()) )
    evrCodes = np.array( lcls.getEvrCodes(verbose=False) )
    evrCodes.resize(nEvr)
    evtData.evrCodes.append( evrCodes.copy()  )
                        
    # timing information
    evtData.fiducials.append( lcls.getEventFiducial())
    evtData.times.append( lcls.getEventTime())

def appendEpicsData(epics, evtData):
    #evtData.deltaK.append(epics.value('USEG:UND1:3350:KACT'))
    #evtData.deltaEnc.append( np.array(
    #    [epics.value('USEG:UND1:3350:{}:ENC'.format(i)) for i in range(1,5)]))

    # Continuum delay stage in blue box
    stage =  epics.value('AMO:LAS:DLS:05:MTR.RBV')
    if stage is None:
        stage = np.nan
    evtData.delayStage.append( - stage * 2 / c_0_mm_per_fs )

    # Laster to x-ray locking system timing
    fs = epics.value('LAS:FS1:VIT:FS_TGT_TIME')
    if fs is None:
        fs = np.nan
    evtData.fsTiming.append( fs * 1e6 )

    # Time tool signal
    tt = epics.value('TTSPEC:FLTPOS_PS') 
    width = epics.value('TTSPEC:FLTPOSFWHM') 
    if (tt is None) or (width is None) or (width > 40.):
        tt = np.nan
    evtData.ttTime.append(  tt * 1e3 )

 
def packageAndSendData(evtData, req, verbose=False):
    # Make a data packet
    data = np.zeros(dSize, dtype=float)
    # Inform about the sender
    data[dRank] = rank
    # timing information
    data[dFiducials] = evtData.fiducials[0]
    data[dTime] = evtData.times[0]
    # amplitude 
    data[dFull] = evtData.full[0]
    data[dIntRoi0] = evtData.intRoi0[0]
    data[dIntRoi1] = evtData.intRoi1[0]
    # polarization
    data[dPol] = evtData.pol[0]
    # Photon energy
    if args.photonEnergy != 'no':
        data[dEnergy] = evtData.energy[0]

    # e-beam data
    data[dEL3] = evtData.ebEnergyL3[0]
    #print 'rank {} with gasdets: {}'.format(rank, repr(evtData.gasDet[0]))
    data[dFEE] = evtData.gasDet[0]
    data[dEvr] = evtData.evrCodes[0]

    # DELTA data
    #data[dDeltaK] = evtData.deltaK[0]
    #data[dDeltaEnc] = evtData.deltaEnc[0]
    data[dDelayStage] = evtData.delayStage[0]
    data[dFsTiming] = evtData.fsTiming[0]
    data[dTtTime] = evtData.ttTime[0]

    # wait if there is an active send request
    if req != None:
        req.Wait()
    #copy the data to the send buffer
    evtData.buf = data.copy()
    if verbose:
        print 'rank', rank, 'sending data'
    req = world.Isend([evtData.buf, MPI.FLOAT], dest=0, tag=0)

    return req

 
def mergeMasterAndWorkerData(evtData, masterLoop, args):
    # Unpack the data
    evtData.sender = np.array( evtData.sender + [ d[dRank] for d in
        masterLoop.arrived ])
    evtData.fiducials = np.array( evtData.fiducials + [ d[dFiducials] for d in
        masterLoop.arrived ])
    evtData.times = np.array( evtData.times +[ d[dTime] for d in
        masterLoop.arrived ])
        
    evtData.full = np.array( evtData.full +
            [d[dFull] for d in masterLoop.arrived])
    evtData.intRoi0 = np.array( evtData.intRoi0 +
            [d[dIntRoi0] for d in masterLoop.arrived])
    evtData.intRoi1 = np.array( evtData.intRoi1 +
            [d[dIntRoi1] for d in masterLoop.arrived])
    evtData.pol = np.array( evtData.pol + [d[dPol] for d in masterLoop.arrived])
    if args.photonEnergy != 'no':
        evtData.energy = np.array( evtData.energy + [d[dEnergy] for d in
            masterLoop.arrived] )
    if args.calibrate > -1:
        masterLoop.calibValues.append(evtData.intRoi0 if args.calibrate==0 else
                evtData.intRoi1)

    #evtData.positions = np.array( evtData.positions ) 

    evtData.ebEnergyL3 = np.array( evtData.ebEnergyL3 + [ d[dEL3] for d in
        masterLoop.arrived ])
    evtData.gasDet = np.array( evtData.gasDet + [ d[dFEE] for d in
        masterLoop.arrived ])
    evtData.evrCodes = np.array( evtData.evrCodes + [ d[dEvr] for d in
        masterLoop.arrived ]) 

    #evtData.deltaK = np.array( evtData.deltaK + [ d[dDeltaK] for d in
    #    masterLoop.arrived ])
    #evtData.deltaEnc = np.array( evtData.deltaEnc + [ d[dDeltaEnc] for d in
    #    masterLoop.arrived ])
    evtData.delayStage = np.array( evtData.delayStage + [ d[dDelayStage] for d
        in masterLoop.arrived ])
    evtData.fsTiming = np.array( evtData.fsTiming + [ d[dFsTiming] for d in
        masterLoop.arrived ])
    evtData.ttTime = np.array( evtData.ttTime + [ d[dTtTime] for d in
        masterLoop.arrived ])

    # delete the recived data buffers
    for i in range(masterLoop.nArrived):
        masterLoop.buf.popleft()
        masterLoop.req.popleft()

# lists for the reference information in the timing histogram
nShotsRef = 1000
fullRefBuff = deque(maxlen=nShotsRef)
roi0RefBuff = deque(maxlen=nShotsRef)
roi1RefBuff = deque(maxlen=nShotsRef)

nShotsHistory = 10000
fullBuff = deque(maxlen=nShotsHistory)
roi0Buff = deque(maxlen=nShotsHistory)
roi1Buff = deque(maxlen=nShotsHistory)
tBuff = deque(maxlen=nShotsHistory)


             
def makeTimingHistogram(evtData):
    
    # No alignment reference evr code
    refEvr = 67

    # Calculate signals for all the events
    #Average ove the two last fee gas dets
    fee = evtData.gasDet[:,2:].mean(axis=1)
    full = evtData.full.mean(axis=1) / fee
    roi0 = evtData.intRoi0.mean(axis=1) / fee
    roi1 = evtData.intRoi1.mean(axis=1) / fee

    delay = evtData.delayStage + evtData.ttTime
    
    nanMask = ~ ( np.isnan(fee)
            | np.isnan(full)
            | np.isnan(roi0)
            | np.isnan(roi1)
            | np.isnan(delay) )


    # Update the reference
    refMask = np.array( [(refEvr in codes) for codes in evtData.evrCodes] )
    # Add the events to theref lists
    refFull = None
    refRoi0 = None
    refRoi1 = None
    for refBuff, ref, buff, data in zip(
            [fullRefBuff, roi0RefBuff, roi1RefBuff],
            [refFull, refRoi0, refRoi1],
            [fullBuff, roi0Buff, roi1Buff],
            [full, roi0, roi1]):
        refBuff.extend(data[refMask & nanMask])
        ref = np.mean(refBuff)
        if np.isnan(ref):
            ref = 0
        # extend the buffer with the data minus reference
        buff.extend( (data-ref)[ (~refMask) & nanMask ] )

    # extend the time
    tBuff.extend( delay[ (~refMask) & nanMask ] )

    if len(tBuff) == 0:
        return

    binSize = 20.
    binEdgeMin = np.floor( np.min(tBuff) / binSize )
    binEdgeMax = np.ceil( np.max(tBuff) / binSize )
    binEdges = np.arange(binEdgeMin, binEdgeMax + 1) * binSize
    timeBins = binEdges[:-1] + np.diff( binEdges ) / 2

    fullBinned, _, _ =  binned_statistic(tBuff, fullBuff, statistic='mean',
            bins=binEdges)
    roi0Binned, _, _ =  binned_statistic(tBuff, roi0Buff, statistic='mean',
            bins=binEdges)
    roi1Binned, _, _ =  binned_statistic(tBuff, roi1Buff, statistic='mean',
            bins=binEdges)

    return {'time' : timeBins,
            'full' : fullBinned,
            'roi0' : roi0Binned,
            'roi1' : roi1Binned}


def zmqPlotting(evtData, augerAverage, scales, zmq):
 
    # Averaging of roi 1 
    if augerAverage.fOldRoi1 != 0:
        if augerAverage.plotRoi1 is None:
            augerAverage.plotRoi1 = evtData.intRoi1[0,:]
        else:
            augerAverage.plotRoi1 *= augerAverage.fOldRoi1
            augerAverage.plotRoi1 += augerAverage.fNewRoi1 \
                    * evtData.intRoi1[0,:]
        for r1 in evtData.intRoi1[1:,:]:
            augerAverage.plotRoi1 *= augerAverage.fOldRoi1
            augerAverage.plotRoi1 += augerAverage.fNewRoi1 \
                    * evtData.intRoi1[0,:]
    else:
        augerAverage.plotRoi1 = evtData.intRoi1[-1,:]
    plotData = {}
    plotData['polar'] = {
            'full':evtData.full[-1],
            'roi0':evtData.intRoi0[-1,:],
            'roi1':augerAverage.plotRoi1,
            'A':evtData.pol[-1][0],
            'beta':evtData.pol[-1][2],
            'tilt':evtData.pol[-1][4],
            'linear':evtData.pol[-1][6]}
    plotData['strip'] = [evtData.fiducials, evtData.pol]
    plotData['traces'] = {}
    if evtData.timeSignals_V != None:
        plotData['traces']['timeRaw'] = evtData.timeSignals_V
        plotData['traces']['timeFiltered'] = evtData.timeSignalsFiltered_V
        plotData['traces']['timeRoi0'] = [sig[sl] for sig, sl in
                zip(evtData.timeSignalsFiltered_V, scales.tRoi0S)]
        plotData['traces']['timeRoi1'] = [sig[sl] for sig, sl in
                zip(evtData.timeSignalsFiltered_V, scales.tRoi1S)]
        plotData['traces']['timeScale'] = [scales.time_us]*16
        plotData['traces']['timeScaleRoi0'] = [ scales.time_us[sl] for sl in
                scales.tRoi0S ]
        plotData['traces']['timeScaleRoi1'] = [ scales.time_us[sl] for sl in
                scales.tRoi1S ]
    if args.photonEnergy != 'no':
        plotData['energy'] = np.concatenate(
                [evtData.fiducials.reshape(-1,1), evtData.energy],
                axis=1).reshape(-1)
    #plotData['spectrum'] = {}
    #plotData['spectrum']['energyScale'] = scales.energy_eV
    #plotData['spectrum']['energyScaleRoi0'] = scales.energyRoi0_eV
    #plotData['spectrum']['energyAmplitude'] = evtData.energySignals
    #plotData['spectrum']['energyAmplitudeRoi0'] = \
    #        evtData.energySignals
            

    #position data
    #plotData['positions'] = evtData.positions.mean(axis=0)

    plotData['timeHist'] = makeTimingHistogram(evtData)
    #print  plotData['timeHist']

    zmq.sendObject(plotData)
                

def openSaveFile(format, online=False, config=None):
    fileName = '/reg/neh/home/alindahl/output/amoi0114/'
    if online is True:
        t = time.localtime()
        fileName += 'online{}-{}-{}_{}-{}-{}.{}'.format(t.tm_year, t.tm_mon,
                t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, format)
    else:
        fileCount = 0
        if config is None:
            fileName += 'outfile{}.' + format
        else:
            fileName += 'run' + config.offlineSource.split('=')[-1] + '_{}.' + format
        while os.path.exists(fileName.format(fileCount)):
            fileCount += 1
        fileName = fileName.format(fileCount)
    if format == 'txt':
        file = open(fileName,'w')

        file.write('eventTime\tfiducial')
        file.write('\tebEnergyL3')
        file.write('\tfee11\tfee12\tfee21\tfee22')
        for i in range(16):
            file.write('\tauger_{}'.format(i))
        for i in range(16):
            file.write('\tphoto_{}'.format(i))
        file.write('\tI0\tI0_err\tbeta\tbeta_err\ttilt\ttilt_err\tlinDegree\tlinDegree_err')
        #file.write('\tdeltaK')
        #for i in range(4):
        #    file.write('\tdeltaEnc{}'.format(i+1))

        file.write('\n')
        file.flush()
        return file

def writeDataToFile(file, data, format):
    if format == 'txt':
        for i in range(len(data.sender)):
            line = ( repr( data.times[i] ) + '\t' +
                    repr( data.fiducials[i] ) )
            line += '\t' + repr(data.ebEnergyL3[i])
            for a in data.gasDet[i,:]:
                line += '\t' + repr(a)
            for a in data.intRoi1[i,:]:
                line += '\t' + repr(a)
            for a in data.intRoi0[i,:]:
                line += '\t' + repr(a)
            for a in data.pol[i,:]:
                line += '\t' + repr(a)
            #line += '\t' + repr(data.deltaK[i])
            #for a in data.deltaEnc[i,:]:
            #    line += '\t' + repr(a)

            line += '\n'

            file.write(line)

        file.flush()

def closeSaveFile(file):
    try:
        file.close()
    except:
        pass

def main(args, verbose=False):
    try:
        # Import the configuration file
        config = importConfiguration(args, verbose=verbose)
    
        # Read the detector transmission calibrations
        detCalib = getDetectorCalibration(verbose=verbose,
                fileName=args.gainCalib)

        # Change the configuration fit masks according to the factors
        config.nanFitMask = config.nanFitMask.astype(float)
        config.nanFitMask[np.isnan(detCalib.factors)] = np.nan
        config.boolFitMask[np.isnan(detCalib.factors)] = False
            
        # The master have some extra things to do
        if rank == 0:
            # Set up the plotting in AMO
            from ZmqSender import zmqSender
            zmq = zmqSender()
    
            masterLoop = masterLoopSetup(args) 

    
            # Averaging factor for the augers
            augerAverage = aolUtil.struct()
            augerAverage.fNewRoi1 = args.roi1Average
            augerAverage.fOldRoi1 = 1. - augerAverage.fNewRoi1
            augerAverage.plotRoi1 = None

            if args.saveData != 'no':
                saveFile = openSaveFile(args.saveData, not args.offline, config)
            
        else:
            # set an empty request
            req = None
    
    
    
        # Connect to the correct datasource
        ds = connectToDataSource(args, config, verbose=verbose)
        events = ds.events()

        # Get the epics store
        epics = ds.env().epicsStore()
    
        # Get the next event. The purpouse here is only to make sure the
        # datasource is initialized enough so that the env object is avaliable.
        evt = events.next()
    
        cookieBox.proj.setFitMask(config.boolFitMask)
    
        # Get the scales that we need
        scales = getScales(ds.env(), config)
        print scales
    
        # The main loop that never ends...
        while 1:
            # An event data container
            eventData = eventDataContainer(args)
                
            # The master should set up the recive requests
            if rank == 0:
                #masterDataSetup(eventData)
                setupRecives(masterLoop, verbose=verbose)
               
            # The master should do something usefull while waiting for the time
            # to pass
            while (time.time() < masterLoop.tStop) if rank==0 else 1 :
        
                # Get the next event
                evt = events.next()

                appendEventData(evt, eventData, config, scales, detCalib,
                        masterLoop if rank==0 else None, verbose=verbose)

                appendEpicsData(epics, eventData)

                
                # Everyone but the master goes out of the loop here
                if rank > 0:
                    break
            
            
            # Rank 0 stuff on timed loop exit
            if rank == 0:
                # Shift the stop time
                masterLoop.tStop += masterLoop.tPlot
        
                # Check how many arrived
                masterLoop.nArrived = \
                        [r.Get_status() for r in masterLoop.req].count(True)
                if verbose:
                    print 'master recived data from', masterLoop.nArrived, \
                            'events'
                    print 'with its own events it makes up', \
                            masterLoop.nArrived + masterLoop.nProcessed
                if masterLoop.nArrived == 0 and masterLoop.nProcessed==0:
                    continue
        
                # A temp buffer for the arrived data
                masterLoop.arrived = [b.reshape(-1) for i,b in
                        enumerate(masterLoop.buf) if i < masterLoop.nArrived]
        
                mergeMasterAndWorkerData(eventData, masterLoop, args)

    
                # Send data for plotting
                zmqPlotting(eventData, augerAverage, scales, zmq)

                if args.saveData != 'no':
                    writeDataToFile(saveFile, eventData, args.saveData)
                   
    
            else:
                # The rest of the ranks come here after breaking out of the loop
                # the goal is to send data to the master core.
                req = packageAndSendData(eventData, req)


    except KeyboardInterrupt:
        print "Terminating program."

        if rank == 0 and args.calibrate > -1:
            if args.saveData != 'no':
                closeSaveFile(saveFile)

            saveDetectorCalibration(masterLoop, detCalib, config,
                    verbose=verbose, beta = args.calibBeta)
           

if __name__ == '__main__':
    # Start here
    # parset the command line
    args = arguments.parse()
    if args.verbose:
        print args

    main(args, args.verbose)

