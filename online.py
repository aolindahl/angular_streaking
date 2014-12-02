import sys
sys.path.append('aolPyModules')
from mpi4py import MPI
import psana
import arguments
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import time
import platform
import sys
from collections import deque
import importlib
import os.path
import cookieBox4 as cookieBox
import lmfit
import aolUtil
import lclsData

# Set up the mpi cpmmunication
world = MPI.COMM_WORLD
rank = world.Get_rank()
worldSize = world.Get_size()



#############
# Data definitions
s = 0 # Data size tracker

dRank = s
s += 1
dFiducials = s
s += 1
dTime = s
s += 1
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
dDeltaK = s
s += 1
dDeltaEnc = slice(s, s+4)
s += 4

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
    masterData.energyAmplitude = None
    masterData.energyAmplitudeRoi0 = None
    masterData.timeAmplitude = None
    masterData.timeAmplitudeFiltered = None
    masterData.timeAmplitudeRoi0 = None
    masterData.timeAmplitudeRoi1 = None

    
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


def getScales(cb, config):
    scales = aolUtil.struct()
    # Grab the relevant scales
    scales.energy_eV = cb.getEnergyScales_eV()[0]
    scales.eRoi0S = cookieBox.sliceFromRange([scales.energy_eV],
            config.energyRoi0_eV_common)
    scales.energyRoi0_eV = scales.energy_eV[scales.eRoi0S]
    
    scales.time_us = cb.getTimeScales_us()
    scales.tRoi0S = cookieBox.sliceFromRange(scales.time_us,
            config.timeRoi0_us_common)
    scales.tRoi0BgS = cookieBox.sliceFromRange(scales.time_us,
                        config.timeRoi0Bg_us_common)
    scales.tRoi0BgFactors = np.array([np.float(s.stop-s.start)/(bg.stop-bg.start) for
        s,bg in zip(scales.tRoi0S, scales.tRoi0BgS)])
    scales.tRoi1S = cookieBox.sliceFromRange(scales.time_us,
            config.timeRoi1_us_common)
    scales.timeRoi0_us = [t[s] for t,s in zip(scales.time_us, scales.tRoi0S)]
    scales.timeRoiBg0_us = [t[s] for t,s in zip(scales.time_us,
        scales.tRoi0BgS)]
    scales.timeRoi1_us = [t[s] for t,s in zip(scales.time_us, scales.tRoi1S)]
    scales.angles = cb.getAngles('rad')
    scales.anglesFit = np.linspace(0, 2*np.pi, 100)
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
    event.intRoi0 = []
    event.intRoi0Bg = []
    event.intRoi1 = []
    event.pol = []
    event.positions = []

    event.ebEnergyL3 = []
    event.gasDet = []

    if args.photonEnergy != 'no':
        event.energy = []

    event.deltaK = []
    event.deltaEnc = []

    return event

def appendEventData(evt, evtData, cb, lcls, config, scales, detCalib):
    evtData.sender.append(rank)


    # Get the intensities
    evtData.intRoi0.append(
        (
            cb.getIntensityDistribution(domain='Time',
                roiSlices = scales.tRoi0S,
                detFactors = detCalib.factors)
            - cb.getIntensityDistribution(domain='Time',
                roiSlices = scales.tRoi0BgS,
                detFactors = detCalib.factors) * scales.tRoi0BgFactors
            ) * config.nanFitMask)

    #evtData.intRoi0Bg
    
    evtData.intRoi1.append(cb.getIntensityDistribution(domain='Time',
        roiSlices=scales.tRoi1S, detFactors=detCalib.factors) *
        config.nanFitMask)
        
    # Get the initial fit parameters
    #params = cookieBox.initialParams(evtData.intRoi0[-1])
    params = cookieBox.initialParams()
    params['A'].value, params['linear'].value, params['tilt'].value = \
            cb.proj.solve(evtData.intRoi0[-1], args.beta)
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
    evtData.positions.append( cb.getPositions() )
    
    # Get the photon energy center and width
    if args.photonEnergy != 'no':
        evtData.energy.append(
                cb.getPhotonEnergy(
                    energyShift=args.energyShift
                    )
                )

    # Get lcls parameters
    evtData.ebEnergyL3.append(lcls.getEBeamEnergyL3_MeV())
    evtData.gasDet.append( np.array(lcls.getPulseEnergy_mJ()) )
                        
    # timing information
    evtData.fiducials.append( lcls.getEventFiducial())
    evtData.times.append( lcls.getEventTime())

def appendEpicsData(epics, evtData):
    evtData.deltaK.append(epics.value('USEG:UND1:3350:KACT'))
    evtData.deltaEnc.append( np.array(
        [epics.value('USEG:UND1:3350:{}:ENC'.format(i)) for i in range(1,5)]))
 
def masterEventData(evtData, cb, masterLoop, scales, verbose=False):
    # Grab the y data
    #evtData.energyAmplitude = np.average(cb.getEnergyAmplitudes(), axis=0)
    evtData.energyAmplitude = np.array(cb.getEnergyAmplitudes()[2])
    evtData.energyAmplitudeRoi0 = \
            evtData.energyAmplitude[scales.eRoi0S]
    evtData.timeAmplitude = cb.getTimeAmplitudes()
    evtData.timeAmplitudeFiltered = cb.getTimeAmplitudesFiltered()
    evtData.timeAmplitudeRoi0 = [t[s] for t, s in
            zip(evtData.timeAmplitudeFiltered, scales.tRoi0S)]
    evtData.timeAmplitudeRoi1 = [t[s] for t, s in
            zip(evtData.timeAmplitudeFiltered, scales.tRoi1S)]

    if verbose:
        print 'Rank', rank, '(master) grabbed one event.'
    # Update the event counters
    masterLoop.nProcessed += 1

 
def packageAndSendData(evtData, req, verbose=False):
    # Make a data packet
    data = np.zeros(dSize, dtype=float)
    # Inform about the sender
    data[dRank] = rank
    # timing information
    data[dFiducials] = evtData.fiducials[0]
    data[dTime] = evtData.times[0]
    # amplitude 
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

    # DELTA data
    data[dDeltaK] = evtData.deltaK[0]
    data[dDeltaEnc] = evtData.deltaEnc[0]

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

    evtData.positions = np.array( evtData.positions ) 

    evtData.ebEnergyL3 = np.array( evtData.ebEnergyL3 + [ d[dEL3] for d in
        masterLoop.arrived ])
    evtData.gasDet = np.array( evtData.gasDet + [ d[dFEE] for d in
        masterLoop.arrived ])

    evtData.deltaK = np.array( evtData.deltaK + [ d[dDeltaK] for d in
        masterLoop.arrived ])
    evtData.deltaEnc = np.array( evtData.deltaEnc + [ d[dDeltaEnc] for d in
        masterLoop.arrived ])

    # delete the recived data buffers
    for i in range(masterLoop.nArrived):
        masterLoop.buf.popleft()
        masterLoop.req.popleft()
            

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
            'roi0':evtData.intRoi0[-1,:],
            'roi1':augerAverage.plotRoi1,
            'A':evtData.pol[-1][0],
            'beta':evtData.pol[-1][2],
            'tilt':evtData.pol[-1][4],
            'linear':evtData.pol[-1][6]}
    plotData['strip'] = [evtData.fiducials, evtData.pol]
    plotData['traces'] = {}
    if evtData.timeAmplitude != None:
        plotData['traces']['timeRaw'] = evtData.timeAmplitude
        plotData['traces']['timeFiltered'] = evtData.timeAmplitudeFiltered
        plotData['traces']['timeRoi0'] = evtData.timeAmplitudeRoi0
        plotData['traces']['timeRoi1'] = evtData.timeAmplitudeRoi1
        plotData['traces']['timeScale'] = scales.time_us
        plotData['traces']['timeScaleRoi0'] = scales.timeRoi0_us
        plotData['traces']['timeScaleRoi1'] = scales.timeRoi1_us
    if args.photonEnergy != 'no':
        plotData['energy'] = np.concatenate(
                [evtData.fiducials.reshape(-1,1), evtData.energy],
                axis=1).reshape(-1)
    plotData['spectrum'] = {}
    plotData['spectrum']['energyScale'] = scales.energy_eV
    plotData['spectrum']['energyScaleRoi0'] = scales.energyRoi0_eV
    plotData['spectrum']['energyAmplitude'] = evtData.energyAmplitude
    plotData['spectrum']['energyAmplitudeRoi0'] = \
            evtData.energyAmplitudeRoi0
            

    #position data
    plotData['positions'] = evtData.positions.mean(axis=0)


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
        file.write('\tdeltaK')
        for i in range(4):
            file.write('\tdeltaEnc{}'.format(i+1))

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
            line += '\t' + repr(data.deltaK[i])
            for a in data.deltaEnc[i,:]:
                line += '\t' + repr(a)

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
    
        # Make the cookie box object
        cb = cookieBox.CookieBox(config, verbose=False)
        cb.proj.setFitMask(config.boolFitMask)
        if args.bgAverage != 1:
            cb.setBaselineSubtractionAveraging(args.bgAverage)

        # Make the lcls object
        lcls = lclsData.LCLSdata(config.lclsConfig, quiet=False)
    
        # Set up the scales
        retardation = epics.value(config.retardationPV)
        cb.setupScales(ds.env(), config.energyScaleBinLimits,
                retardation=retardation)
    
        # Get the scales that we need
        scales = getScales(cb, config)
    
        # The main loop that never ends...
        while 1:
            # An event data container
            eventData = eventDataContainer(args)
                
            # The master should set up the recive requests
            if rank == 0:
                masterDataSetup(eventData)
                setupRecives(masterLoop, verbose=verbose)
               
            # The master should do something usefull while waiting for the time
            # to pass
            while (time.time() < masterLoop.tStop) if rank==0 else 1 :
        
                # If offline
                #if  args.offline:
                    # wait for a while
                    # time.sleep(0.00001)
    
    
                # Get the next event
                evt = events.next()

                #Check if the retardation have changed
                tempRet = epics.value(config.retardationPV)
                if tempRet != retardation:
                    # If so the scales have to be updated
                    retardation = tempRet
                    # Set up the scales
                    cb.setupScales(ds.env(), config.energyScaleBinLimits,
                            retardation=retardation)
    
                    # Get the scales that we need
                    scales = getScales(cb, config)

                

    
                # Pass the event to the processing
                cb.setRawData(evt, newDataFactor=args.floatingAverage)
                lcls.setEvent(evt)
    
                # Randomize the amplitudes if this is requested 
                if args.randomize:
                    cb.randomizeAmplitudes()
    
                appendEventData(evt, eventData, cb, lcls, config, scales,
                        detCalib)

                appendEpicsData(epics, eventData)

                # rank 0 grabs some trace data
                if rank == 0:
                    masterEventData(eventData, cb, masterLoop, scales,
                            verbose=verbose)
                
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

