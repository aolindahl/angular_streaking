##############
# General python modules
import sys
from mpi4py import MPI
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
import warnings
warnings.simplefilter('error')

############
# Custom modules
# Soume found in:
sys.path.append( os.path.dirname(os.path.abspath(__file__)) +  '/aolPyModules')
import cookie_box
import tof
import aolUtil
import lcls
import arguments
import simplepsana

# Set up the mpi cpmmunication
world = MPI.COMM_WORLD
rank = world.Get_rank()
worldSize = world.Get_size()

if rank == 0:
    from psmon import publish
    from psmon.plots import XYPlot, MultiPlot, Image
    publish.init()

c_0_mm_per_fs = 2.99792458e8 * 1e3 * 1e-15
# http://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=universal_in! 2014-04-21

#############
# Data definitions
s = 0 # Data size tracker

d_int_e_roi_0 = slice(s, s+16)
s += 16
d_int_e_roi_1 = slice(s, s+16)
s += 16
d_int_e_roi_2 = slice(s, s+16)
s += 16
d_int_e_roi_3 = slice(s, s+16)
s += 16
#dRank = s
#s += 1
#dFiducials = s
#s += 1
#dTime = s
#s += 1
#dFull = slice(s, s+16)
#s += 16
dIntRoi0 = slice(s, s+16)
s += 16
dIntRoi1 = slice(s, s+16)
s += 16
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

dTraces = None
d_energy_traces = None


def connect_to_data_source(args, config, verbose=False):
    # If online
    if not args.offline:
        # make the shared memory string
        dataSource = 'shmem=AMO.0:stop=no'
    else:
        dataSource = config.offline_source
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

    return simplepsana.get_data_source(dataSource)


def import_configuration(args, verbose=False):
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
    params = cookie_box.initial_params()
    params['A'].value = 1
    params['beta'].value = beta
    params['tilt'].value = 0
    params['linear'].value = 1
    factors = cookie_box.model_function(params, np.radians(np.arange(0, 360,
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
          

def master_data_setup(args):
    # Container for the master data
    master_data = aolUtil.struct()
    master_data.energySignal_V = None
    master_data.timeSignalseFiltered_V = None
    N = args.traceAverage
    if N == None:
        N = 1
    master_data.traceBuffer = deque([], N)
    master_data.roi_0_buffer = deque([], N)
    master_data.energy_trace_buffer = deque([], N)
    return master_data

    
def master_loop_setup(args, scales):
    # Master loop data collector
    masterLoop = aolUtil.struct()
    # Define the plot interval from the command line input
    masterLoop.tPlot = args.plotInterval

    # Make template of the array that should be sent between the ranks
    masterLoop.bufTemplate = np.empty(dSize, dtype=float)
    
    # Make empty queues.
    masterLoop.req = deque()
    masterLoop.buf = deque()
        
    # Initialize the stop time
    masterLoop.tStop = time.time()
    
    # Calibration
    if args.calibrate > -1:
        masterLoop.calibValues = []
    
    return masterLoop


def get_scales(env, cb, verbose=False):
    global dSize
    global dTraces 
    global d_energy_traces

    scales = aolUtil.struct()
    # Grab the relevant scales
    scales.energy_eV = cb.get_energy_scales_eV()[0]
    scales.energy_roi_0_eV = cb.get_energy_scales_eV(roi=0)[0]
    #if rank==0:
    #    print scales.energy_eV
    #    print scales.energy_roi_0_eV
    scales.energy_roi_0_slice = slice(
        scales.energy_eV.searchsorted(np.min(scales.energy_roi_0_eV)),
        scales.energy_eV.searchsorted(np.max(scales.energy_roi_0_eV),
                                      side='right'))

    #raw_time_scales = cb.get_time_scales_us()

    #scales.baseline_slices = cookie_box.slice_from_range(
    #        raw_time_scales, [-np.inf, config.baselineEnd_us])

    
    scales.time_us = cb.get_time_scales_us()
    scales.time_roi_0_us = cb.get_time_scales_us(roi=0)
    #scales.timeRoiBg0_us = [scales.time_us[s] for s in scales.tRoi0BgS]
    scales.timeRoi1_us = cb.get_time_scales_us(roi=1)
    scales.angles = cookie_box.phi_rad
    scales.anglesFit = np.linspace(0, 2*np.pi, 100)


    # These are the globals
    #print [len(t) for t in scales.time_us]
    traces_size = 16 * np.max([len(t) for t in scales.time_us])
    #print traces_size
    dTraces = slice(dSize, dSize + traces_size)
    #print dTraces
    #print 'dSize =', dSize
    dSize += traces_size

    energy_traces_size = 16 * len(scales.energy_eV)
    d_energy_traces = slice(dSize, dSize+energy_traces_size)
    dSize += energy_traces_size

    return scales


def get_event_data(config, scales, detCalib,
        cb, args, epics, verbose=False):
    data = np.zeros(dSize, dtype=float)

    #data[dRank](rank)

    time_amplitudes = cb.get_time_amplitudes()
    if None in time_amplitudes:
        return None

    data[dTraces] = np.nan
    length = (dTraces.stop - dTraces.start) / 16
    #print 'length =', length
    #print data.shape, dTraces.start, dTraces.stop
    start = dTraces.start
    for t_sig in time_amplitudes:
        if t_sig is None:
            return req
        #print len(t_sig)
        #print start, start+len(t_sig)
        #print data.shape, data[dTraces].shape
        data[start : start + len(t_sig)] = t_sig
        start += length

    #data[dFull] = [ sig.sum() for sig in time_amplitudes] * \
    #              detCalib.factors * config.nanFitMask

    #print cb.get_time_amplitudes_filtered(roi=0)
    data[dIntRoi0] = np.array([np.sum(trace) for trace
                               in cb.get_time_amplitudes_filtered(roi=0)]) * \
                     detCalib.factors * config.nanFitMask
    #data[dIntRoi0] = np.sum(cb.get_time_amplitudes_filtered(roi=0), axis=1) * \
    #                 detCalib.factors * config.nanFitMask

    #data[dIntRoi1] = np.sum(cb.get_time_amplitudes_filtered(roi=1), axis=1) * \
    #                 detCalib.factors


    #print np.array(cb.get_energy_amplitudes()).reshape(-1).shape
    #print d_energy_traces
    #print data.shape
    data[d_energy_traces] = np.array(cb.get_energy_amplitudes()).reshape(-1)

    data[d_int_e_roi_0] = cb.get_intensity_distribution(rois=0,
                                                        domain='Energy')
    data[d_int_e_roi_1] = cb.get_intensity_distribution(rois=1,
                                                        domain='Energy')
    data[d_int_e_roi_2] = cb.get_intensity_distribution(rois=2,
                                                        domain='Energy')
    data[d_int_e_roi_3] = cb.get_intensity_distribution(rois=3,
                                                        domain='Energy')

    #def get_intensity_distribution(self, rois=[slice(None)]*16,
    #        domain='Time', verbose=None, detFactors=[1]*16):

    # Get the initial fit parameters
    #params = cookie_box.initial_params(evtData.intRoi0[-1])
    #params = cookie_box.initial_params()
    #params['A'].value, params['linear'].value, params['tilt'].value = \
    #        cookie_box.proj.solve(evtData.intRoi0[-1], args.beta)
    # Lock the beta parameter. To the command line?
    #params['beta'].value = args.beta
    #params['beta'].vary = False

    #print params['A'].value, params['linear'].value, params['tilt'].value
    
    # Perform the fit
    #print scales.angles[config.boolFitMask]
    #print evtData.intRoi0[-1][config.boolFitMask]
    #res = lmfit.minimize(
    #        cookie_box.model_function,
    #        params,
    #        args=(scales.angles[config.boolFitMask],
    #            evtData.intRoi0[-1][config.boolFitMask]),
    #        method='leastsq')

    #print params['A'].value, params['linear'].value, params['tilt'].value
    
    #lmfit.report_fit(params)
                
    # Store the values
    #evtData.pol.append(np.array( [
    #    params['A'].value, params['A'].stderr,
    #    params['beta'].value, params['beta'].stderr,
    #    params['tilt'].value, params['tilt'].stderr,
    #    params['linear'].value, params['linear'].stderr
    #    ]))

    
    # Get the photon energy center and width
    #if args.photonEnergy != 'no':
    #    evtData.energy.append(
    #            cb.getPhotonEnergy(
    #                energyShift=args.energyShift
    #                )
    #            )

    # Get lcls parameters
    data[dEL3] = lcls.getEBeamEnergyL3_MeV()
    data[dFEE] = lcls.getPulseEnergy_mJ()
    #evrCodes = np.array(lcls.getEvrCodes(verbose=False))
    #evrCodes.resize(nEvr)
    #evtData.evrCodes.append(evrCodes.copy())
                        
    # timing information
    #evtData.fiducials.append( lcls.getEventFiducial())
    #evtData.times.append( lcls.getEventTime())

    #evtData.deltaK.append(epics.value('USEG:UND1:3350:KACT'))
    #evtData.deltaEnc.append( np.array(
    #    [epics.value('USEG:UND1:3350:{}:ENC'.format(i)) for i in range(1,5)]))

    # Continuum delay stage in blue box
    #stage =  epics.value('AMO:LAS:DLS:05:MTR.RBV')
    #if stage is None:
    #    stage = np.nan
    #evtData.delayStage.append( - stage * 2 / c_0_mm_per_fs )

    ## Laster to x-ray locking system timing
    data[dFsTiming] = epics.value('LAS:FS1:VIT:FS_TGT_TIME')
    #fs = epics.value('LAS:FS1:VIT:FS_TGT_TIME')
    #if fs is None: fs = np.nan
    #evtData.fsTiming.append( fs * 1e6)

    ## Time tool signal
    #tt = epics.value('TTSPEC:FLTPOS_PS') 
    #width = epics.value('TTSPEC:FLTPOSFWHM') 
    #if (tt is None) or (width is None) or (width > 40.):
    #    tt = np.nan
    #evtData.ttTime.append(  tt * 1e3 )
 
    # Inform about the sender
    #data[dRank] = rank
    # timing information
    #data[dFiducials] = evtData.fiducials[0]
    #data[dTime] = evtData.times[0]
    # polarization
    #data[dPol] = evtData.pol[0]
    # Photon energy
    #if args.photonEnergy != 'no':
    #    data[dEnergy] = evtData.energy[0]

    # e-beam data
    #print 'rank {} with gasdets: {}'.format(rank, repr(evtData.gasDet[0]))
    
    #data[dEvr] = evtData.evrCodes[0]

    #data[dDelayStage] = evtData.delayStage[0]
    #data[dFsTiming] = evtData.fsTiming[0]
    #data[dTtTime] = evtData.ttTime[0]

    return data
    
def send_data_to_master(data, req, buffer, verbose=False):
    # wait if there is an active send request
    if req != None:
        req.Wait()
    #copy the data to the send buffer
    buffer = data.copy()
    if verbose and 0:
        print 'rank', rank, 'sending data'
    req = world.Isend([buffer, MPI.FLOAT], dest=0, tag=0)

    return req

 
def merge_arrived_data(data, masterLoop, args, scales, verbose=False):
    if verbose:
        print 'Merging master and worker data.'
        #print 'masterLoop.buf =', masterLoop.buf
    # Unpack the data
    #data.sender = [ d[dRank] for d in masterLoop.buf ]
    #data.fiducials = [ d[dFiducials] for d in masterLoop.buf ]
    #data.times = [ d[dTime] for d in masterLoop.buf ]
        
    #data.full = [d[dFull] for d in masterLoop.buf]
    data.intRoi0 = np.array([d[dIntRoi0] for d in masterLoop.buf])
    data.intRoi1 = np.array([d[dIntRoi1] for d in masterLoop.buf])

    #if args.photonEnergy != 'no':
    #    data.energy = np.array( data.energy + [d[dEnergy] for d in
    #        masterLoop.buf] )
    #if args.calibrate > -1:
    #    masterLoop.calibValues.append(data.intRoi0 if args.calibrate==0 else
    #            data.intRoi1)

    data.timeSignals_V = []
    for event_data in masterLoop.buf:
        data.timeSignals_V.append([d[:len(scale)] for d, scale in
            zip(event_data[dTraces].reshape(16, -1), scales.time_us)])

    data.energy_signals = [d[d_energy_traces].reshape(16, -1) for
                           d in masterLoop.buf]

    data.int_e_roi_0 = np.array([d[d_int_e_roi_0] for d in masterLoop.buf])
    data.int_e_roi_1 = np.array([d[d_int_e_roi_1] for d in masterLoop.buf])
    data.int_e_roi_2 = np.array([d[d_int_e_roi_2] for d in masterLoop.buf])
    data.int_e_roi_3 = np.array([d[d_int_e_roi_3] for d in masterLoop.buf])

    #data.positions = np.array( data.positions ) 

    data.ebEnergyL3 = np.array([d[dEL3] for d in  masterLoop.buf ])
    data.gasDet = np.array([d[dFEE] for d in masterLoop.buf])
    #data.evrCodes = np.array( data.evrCodes + [ d[dEvr] for d in
    #    masterLoop.buf ]) 

    #data.deltaK = np.array( data.deltaK + [ d[dDeltaK] for d in
    #    masterLoop.buf ])
    #data.deltaEnc = np.array( data.deltaEnc + [ d[dDeltaEnc] for d in
    #    masterLoop.buf ])
    #data.delayStage = np.array( data.delayStage + [ d[dDelayStage] for d
    #    in masterLoop.buf ])
    data.fsTiming = np.array([d[dFsTiming] for d in masterLoop.buf])
    #data.ttTime = np.array( data.ttTime + [ d[dTtTime] for
    #    d in masterLoop.buf ] )


    for i in range( len( data.gasDet ) ):
        if data.gasDet[i].mean() > args.tAFee:
            data.traceBuffer.append(data.timeSignals_V[i])
            data.roi_0_buffer.append(data.intRoi0[i])
            data.energy_trace_buffer.append(data.energy_signals[i])

    print 'trace buffer length:', len(data.traceBuffer)
    if len(data.traceBuffer) > 0:
        data.traceAverage = np.mean(data.traceBuffer, axis=0)
        data.roi_0_average = np.mean(data.roi_0_buffer, axis=0)
        data.energy_trace_average = np.mean(data.energy_trace_buffer, axis=0)
    else:
        data.traceAverage = None
        data.roi_0_average = None
        data.energy_trace_aveage = None

        

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


             
def makeTimingHistogram(data):
    
    # No alignment reference evr code
    refEvr = 67

    # Calculate signals for all the events
    #Average ove the two last fee gas dets
    fee = data.gasDet[:,2:].mean(axis=1)
    full = data.full.mean(axis=1) / fee
    roi0 = data.intRoi0.mean(axis=1) / fee
    roi1 = data.intRoi1.mean(axis=1) / fee

    #delay = data.delayStage + data.ttTime
    delay = data.fsTiming
    #delay = data.fsTiming - 4.61802e9
    
    nanMask = ~ ( np.isnan(fee)
            | np.isnan(full)
            | np.isnan(roi0)
            | np.isnan(roi1)
            | np.isnan(delay) )


    # Update the reference
    refMask = np.array( [(refEvr in codes) for codes in data.evrCodes] )
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

    binSize = 10
    #print tBuff
    binEdgeMin = np.floor( float(np.min(tBuff)) / binSize )
    binEdgeMax = np.ceil( float(np.max(tBuff)) / binSize )
    #print binEdgeMin, binEdgeMax
    binEdges = np.arange(binEdgeMin, binEdgeMax + 2) * binSize
    #print binEdges
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


l3BuffLen = 100000
l3Buff = deque([], l3BuffLen)
l3SigBuff = deque([], l3BuffLen)
def l3Plot(data):
    fee = data.gasDet[:,:2].mean(axis=1)
    #full = data.full.mean(axis=1) / fee
    sig = data.intRoi0.mean(axis=1) / fee
    #sig = data.intRoi1.mean(axis=1) / fee

    I = fee > 0.5
    
    if I.sum() == 0:
        return
    #l3Buff.append(data.ebEnergyL3[I].mean())
    #l3SigBuff.append((sig[I]/fee[I]).mean())

    l3Buff.extend(data.ebEnergyL3[I])
    l3SigBuff.extend((sig[I]/fee[I]))
    
    if len(l3Buff) > 1:
        #print len(l3Buff)
        #print len(l3SigBuff)
        l3_plot = XYPlot('', 'L3 plot',
                [np.array(l3Buff)*1e6],
                [np.array(l3SigBuff)],
                xlabel={'axis_title': 'L3 energy', 'axis_units': 'eV'},
                ylabel={'axis_title': 'signal'},
                formats=['g'])
        publish.send('l3', l3_plot)

fsBuffLen = 10000
fsBuff = deque([], fsBuffLen)
fsSigBuff = deque([], fsBuffLen)
def fsPlot(data):
    #print 'fs plotting'
    fee = data.gasDet[:,:2].mean(axis=1)
    #full = data.full.mean(axis=1) / fee
    #sig = data.intRoi0.mean(axis=1) / fee
    #sig = data.intRoi1.mean(axis=1) / fee
    sig = data.int_e_roi_1.mean(axis=1) / fee

    I = fee > 0.05
    #print fee
    #print I.sum() 
    if I.sum() == 0:
        #print 'fs 1'
        return
    #fsBuff.append(data.ebEnergyL3[I].mean())
    #fsSigBuff.append((sig[I]/fee[I]).mean())

    fsBuff.extend(data.fsTiming[I])
    fsSigBuff.extend((sig[I]/fee[I]))

    if len(fsBuff) < 2:
        #print 'fs 2'
        return

    #print fsBuff[0]
    t_min = np.min(fsBuff) - 1e-6
    t_max = np.max(fsBuff) + 1e-6
    t_lims = np.linspace(t_min, t_max, 1024)
    #sig_lims = np.linspace(np.min(fsSigBuff), np.max(fsSigBuff), 8)
    #print 't_lims', t_lims
    #print 'sig_lims', sig_lims
    #image, _, _ = np.histogram2d(fsBuff, fsSigBuff, [t_lims, sig_lims])
    #image /= image.max()
    #print image

    hist, _ = np.histogram(fsBuff, t_lims, weights=fsSigBuff)
    t_hist, _ = np.histogram(fsBuff, t_lims)
    signal = hist / t_hist
    t_ax = (t_lims[:-1] + (t_lims[1]-t_lims[0])/2 - 4581.15705) * 1e-9


    #dt = (t_lims[1] - t_lims[0]) * 1e-9
    #t_start = (t_min - 4581.146) * 1e-9

    #ds = sig_lims[1] - sig_lims[0]
    #s_start = sig_lims[0]
    #print s_start, ds
    
    if len(fsBuff) > 1:
        #print 'fs 3'
        #fs_plot = Image('', 'timing plot', image.T,
        #        aspect_lock=False,
        #        #pos=[t_start, 0],
        #        #scale=[dt, 1],
        #        xlabel={'axis_title': 'fs timing',
        #                'axis_units': 'bin'},
        #        ylabel={'axis_title': 'signal'})
        fs_plot = XYPlot('', 'fs plot',
                [t_ax],
                [signal],
                xlabel={'axis_title': 'fs timing [-4581.15705 ns]',
                        'axis_units': 's'},
                ylabel={'axis_title': 'signal'},
                formats=['b'])
        #3print 'fs 4'
        publish.send('fs', fs_plot)

def angle_energy(data, scales):
    energy_scale_length = len(scales.energy_roi_0_eV)
    image = np.empty((energy_scale_length, 16))
    for i in range(16):
        image[:, i] = data.energy_trace_average[i][scales.energy_roi_0_slice]
        image[:, i] /= image[:, i].sum()
    image[np.isnan(image)] = 0
    nick = Image('', 'Nick plot', image,
                 #aspect_ratio=0.01,
                 aspect_lock=False,
                 pos=[-11.25, scales.energy_roi_0_eV[0]],
                 scale=[22.5,
                     scales.energy_roi_0_eV[1]-scales.energy_roi_0_eV[0]],
                 xlabel={'axis_title': 'angle', 'axis_units': 'degree'},
                 ylabel={'axis_title': 'energy', 'axis_units': 'eV'})
    publish.send('nick', nick)



def psmon_plotting(data, scales, args):
    ##########################################
    # Plot traces
    #print type(data.traceAverage), data.traceAverage.shape
    if not isinstance(data.traceAverage, np.ndarray):
        return
    angles = np.arange(0, 360, 22.5)
    phi = np.deg2rad(angles)
    traces = MultiPlot(('Single shots' if args.traceAverage in [1, None]
                            else 'Average of {} shots'.format(
                                args.traceAverage)),
                       'Traces',
                       ncols=4)
    #print data.traceAverage
    if args.timeAmplitudesSame:
        max = np.max([np.max(trace) for trace in data.traceAverage])
        for trace in data.traceAverage:
            trace[-1] = max

    xfel_names = [16] + range(1,16)
    daq_names = ['1_7', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8',
                 '1_1', '1_2', '1_3', '2_1', '1_4', '2_2', '1_6']

    for i in range(8):
        traces.add(XYPlot(('Single shots' if args.traceAverage in [1, None]
                            else 'Average of {} shots'.format(
                                args.traceAverage)),
            '{} ({}, {}) (w) and {} ({}, {}) (r)'.format(angles[i], xfel_names[i],
                daq_names[i], angles[i+8], xfel_names[i+8], daq_names[i+8]),
            [scales.time_us[i] *  1e-6,
                scales.time_us[i+8] * 1e-6],
            [data.traceAverage[i],
                data.traceAverage[i+8]],
            xlabel={'axis_title': 'time', 'axis_units': 's'},
            ylabel={'axis_title': 'signal', 'axis_units': 'V'},
            formats=['w-', 'r-']))

    #print 'Plotting!'
    publish.send('time_traces', traces)

    ##########################################
    # Polar plot
    r = data.roi_0_average
    x = -r * np.sin(phi)
    y = r * np.cos(phi)
    N = 256
    n_r = 3
    many_phi = np.linspace(0, 2*np.pi, N)
    ring_x = np.empty((n_r, N))
    ring_y = np.empty((n_r, N))
    for i, ring_r in enumerate([np.mean(r)] if n_r==1 else 
                                np.linspace(0, np.max(np.abs(r)),
                                    n_r+1)[1:]):

        ring_x[i,:] = -ring_r * np.sin(many_phi)
        ring_y[i,:] = ring_r * np.cos(many_phi)

    spokes_x = np.zeros(3*8)
    spokes_y = np.zeros(3*8)
    spokes_r = np.max(np.abs(r))
    spokes_x[1::3] = -spokes_r * np.sin(phi[::2])
    spokes_y[1::3] = spokes_r * np.cos(phi[::2])

    ring_x = np.concatenate([ring_x.reshape(-1), spokes_x])
    ring_y = np.concatenate([ring_y.reshape(-1), spokes_y])
    
    polar = XYPlot('roi 0', 'Detector signals',
                   [ring_x, x],
                   [ring_y, y],
                   xlabel={'axis_title': 'signal'},
                   ylabel={'axis_title': 'signal'},
                   formats=['w-', 'r'])
    publish.send('polar', polar)

    #############################################
    # Energy spectra plot
    energy = MultiPlot('', 'Energy traces', ncols=4)

    for i in range(8):
        energy.add(XYPlot(
            '',
            '{} (w) and {} (r) deg'.format(angles[i], angles[i+8]),
            [scales.energy_eV, scales.energy_eV],
            [data.energy_trace_average[i], data.energy_trace_average[i+8]],
            xlabel={'axis_title': 'energy', 'axis_units': 'eV'},
            ylabel={'axis_title': 'signal'},
            formats=['w-', 'r-']))

    publish.send('energy_traces', energy)

    #############################################
    # Angle-Energy image
    angle_energy(data, scales)

    ###########################################
    phi = np.arange(0, 360, 22.5)
    test = XYPlot('', 'test', [phi, phi],
            [data.int_e_roi_1[-1]/data.int_e_roi_3[-1],
                data.int_e_roi_2[-1]/data.int_e_roi_3[-1]])
    publish.send('test', test)

    ###########################################
    # L3 plot
    #l3Plot(data)

    fsPlot(data)

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
        #file.write('\tI0\tI0_err\tbeta\tbeta_err\ttilt\ttilt_err\tlinDegree\tlinDegree_err')
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
            #for a in data.pol[i,:]:
            #    line += '\t' + repr(a)
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
    verbose=args.verbose
    try:
        # Import the configuration file
        config = import_configuration(args, verbose=verbose)

    
        # Make a cookie box object
        cb = cookie_box.CookieBox(config, verbose=False if rank==0 else False)

        # Read the detector transmission calibrations
        detCalib = getDetectorCalibration(verbose=verbose,
                fileName=args.gainCalib)

        # Change the configuration fit masks according to the factors
        config.nanFitMask = config.nanFitMask.astype(float)
        config.nanFitMask[np.isnan(detCalib.factors)] = np.nan
        config.boolFitMask[np.isnan(detCalib.factors)] = False
               
    
        # Connect to the correct datasource
        ds = connect_to_data_source(args, config, verbose=False)
        events = ds.events()

        # Get the epics store
        epics = ds.env().epicsStore()
    
        # Get the next event. The purpouse here is only to make sure the
        # datasource is initialized enough so that the env object is avaliable.
        evt = events.next()
    
    
        # Get the scales that we need
        cb.setup_scales(config.energy_scale_eV, ds.env(),
                retardation=args.retardation)
        scales = get_scales(ds.env(), cb)
        #print scales
     
        # The master have some extra things to do
        if rank == 0:
            masterLoop = master_loop_setup(args, scales) 

            master_data = master_data_setup(args)
    
            # Averaging factor for the augers
            #augerAverage = aolUtil.struct()
            #augerAverage.fNewRoi1 = args.roi1Average
            #augerAverage.fOldRoi1 = 1. - augerAverage.fNewRoi1
            #augerAverage.plotRoi1 = None

            if args.saveData != 'no':
                saveFile = openSaveFile(args.saveData, not args.offline, config)
            
        else:
            # set an empty request
            req = None
            buffer = np.empty(dSize, dtype=float)
            t1 = 0
    

        # The main loop that never ends...
        while 1:
            # The master should set up the receive requests
            if rank == 0:
                #print 'm1'
                masterLoop.buf = []
                while time.time() < masterLoop.tStop:
                    #print 'ml1'
                    masterLoop.buf.append(masterLoop.bufTemplate.copy())
                    #print 'ml2'
                    world.Recv([masterLoop.buf[-1], MPI.FLOAT],
                               source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    #print 'ml3'
                    if args.offline:
                        time.sleep(0.01)
                    #print 'ml4'

                # Shift the stop time
                masterLoop.tStop += masterLoop.tPlot
        
                #print 'm2'
                # Check how many arrived
                masterLoop.nArrived = len(masterLoop.buf)
                if verbose:
                    print 'master received {} events'.format(
                            masterLoop.nArrived)
                if masterLoop.nArrived == 0:
                    continue
        
                merge_arrived_data(master_data, masterLoop, args,
                        scales, verbose=False)

    
                #print 'm3'
                # Send data for plotting
                psmon_plotting(master_data, scales, args)

                if args.saveData != 'no':
                    writeDataToFile(saveFile, eventData, args.saveData)
 
                #print 'm4'
            else:
                #print 'w1'
                # Get the next event
                evt = events.next()
                cb.set_raw_data(evt)
                lcls.setEvent(evt)
                #print 'w3'

                data = get_event_data(config, scales, detCalib,
                        cb, args, epics, verbose=verbose)
                if data is None:
                    continue
                #print 'w4'

                req = send_data_to_master(data, req, buffer,
                                          verbose=verbose)
                #print 'w5'


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

