"""
This file implements following paper:
Video Segmentation by Non-Local Consensus Voting
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
from PIL import Image
import numpy as np
from scipy.misc import imresize
import videoseg.src._init_paths
import videoseg.src.nlc_mod as nlc
import videoseg.src.vid2shots as vid2shots
import videoseg.src.crf as crf

def demo_images(numshards=1, shardid=0, doload=False, dosave=False, crfparams=0, seed=2905, vids=None):

    # For Shot:
    maxShots = 5
    vmax = 0.6
    colBins = 40

    # For NLC:
    redirect = True  # redirecting to output file ? won't print status
    frameGap = 0  # 0 means adjusted automatically per shot (not per video)
    maxSide = 650  # max length of longer side of Im
    minShot = 2  # minimum shot length
    maxShot = 110  # longer shots will be shrinked between [maxShot/2, maxShot]
    binTh = 0.7  # final thresholding to obtain mask
    clearVoteBlobs = True  # remove small blobs in consensus vote; uses binTh
    relEnergy = binTh - 0.1  # relative energy in consensus vote blob removal
    clearFinalBlobs = True  # remove small blobs finally; uses binTh
    maxsp = 400
    iters = 50

    # For CRF:
    gtProb = 0.7
    posTh = binTh
    negTh = 0.4

    # For blob removal post CRF: more like salt-pepper noise removal
    bSize = 25  # 0 means not used, [0,1] relative, >=1 means absolute

    # parse commandline parameters
    np.random.seed(seed)

    masks = []

    for imSeq in vids:

        # First run shot detector
        if not doload:
            shotIdx = vid2shots.vid2shots(imSeq, maxShots=maxShots, vmax=vmax,
                                            colBins=colBins)
        print('Total Shots: ', shotIdx.shape, shotIdx)

        # Adjust frameGap per shot, and then run NLC per shot
        for s in range(shotIdx.shape[0]):
            shotS = shotIdx[s]  # 0-indexed, included
            shotE = imSeq.shape[0] if s == shotIdx.shape[0] - 1 \
                else shotIdx[s + 1]  # 0-indexed, excluded
            shotL = shotE - shotS
            if shotL < minShot:
                continue

            frameGapLocal = frameGap
            if frameGapLocal <= 0 and shotL > maxShot:
                frameGapLocal = int(shotL / maxShot)
            imSeq1 = imSeq[shotS:shotE:frameGapLocal + 1]

            print('\nShot: %d, Shape: ' % (s + 1), imSeq1.shape)
            if not doload:
                maskSeq = nlc.nlc(imSeq1, maxsp=maxsp, iters=iters, #suffix=suffixShot,
                                    clearBlobs=clearVoteBlobs, binTh=binTh,
                                    relEnergy=relEnergy,
                                    redirect=redirect, doload=doload,
                                    dosave=dosave)
                if clearFinalBlobs:
                    maskSeq = nlc.remove_low_energy_blobs(maskSeq, binTh)
            if s == 0:
                mask = (maskSeq > binTh).astype(np.uint8)
            else:
                mask = np.append(mask, (maskSeq > binTh).astype(np.uint8), axis=0)
            print('Mask finished')
        masks.append(mask)

    return masks