# -*- coding: utf-8 -*-
from __future__ import print_function

import pyfits
#from scipy import ndimage
import numpy as np
from scipy import ndimage
from time import clock
import glob
from voronoi_2d_binning import voronoi_2d_binning
#from ppxfcons import ppxfcons
from ppxf import ppxf
import ppxf_util as util
import matplotlib.pyplot as plt


def ppxfrunner():

    ncompfit = 2  # 1 for single component for initial estimates, 2 for both components
    brot = 1	  # 1 for rotation, 0 for fixing to no rotation
    bfract = 0	  # To determine flux fraction of bulge. 1 to determine, 0 otherwise
    mom = 4	  # Number of moments: 2 for only v & sigma, 4 for h3 & h4
    #binning()	  # Have to run the first time to bin the galaxy

    ppxfit(ncompfit,brot,bfract,mom)



#=======================================================================================================================
#========================================================================================================================



def binning():
    file = 'NGC0528-V500.rscube.fits'
    hdu = pyfits.open(file)
    gal_lin = hdu[0].data # axes are wav,y,x
    h1 = hdu[0].header
    gal_err = hdu[1].data
    badpix = hdu[3].data

    xs=70
    ys=70
    ws=gal_lin.shape[0]

#Remove badpixels

    medgal=np.zeros((ys,xs))
    medgalerr=np.zeros((ys,xs))

    xarr=np.zeros(xs*ys)
    yarr=np.zeros(xs*ys)
    medarr=np.zeros(xs*ys)
    mederrarr=np.zeros(xs*ys)

    count=0
    for i in range(xs):
	for j in range(ys):
	    no=np.where(badpix[:,j,i]==1)[0]
	    numbd=no.size
	    numgd=ws-numbd
	    if numgd > 0 and numgd < ws:
		cgal=np.delete(gal_lin[:,j,i],no)
		cgalerr=np.delete(gal_err[:,j,i],no)
		medgal[j,i]=np.median(cgal)
		medgalerr[j,i]=np.median(cgalerr)
	    elif numgd==0:
		medgal[j,i]=0.0
		medgalerr[j,i]=0.0
	    elif numgd==ws:
		medgal[j,i]=np.median(cgal)
		medgalerr[j,i]=np.median(cgalerr)
	    xarr[count]=i
	    yarr[count]=j
	    medarr[count]=medgal[j,i]
	    mederrarr[count]=medgalerr[j,i]
	    count=count+1

    hdu=pyfits.PrimaryHDU(medgal)
    hdu.writeto('medgal.fits',clobber=True)
#Remove low S/N pixels

    x=np.zeros(0)
    y=np.zeros(0)
    signal=np.zeros(0)
    noise=np.zeros(0)
    gd=0

    for i in range(count):
	if medarr[i] > 0.05 and mederrarr[i] < 100:
	    gd=gd+1
	    x=np.append(x,xarr[i])
	    y=np.append(y,yarr[i])
	    signal=np.append(signal,medarr[i])
	    noise=np.append(noise,mederrarr[i])

    output=np.zeros((gd,4))
    output[:,0]=x
    output[:,1]=y
    output[:,2]=signal
    output[:,3]=noise

    np.savetxt('medgalpy.txt',output,fmt="%10.3g")

#Voronoi binning

    targetSN=80
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=1)

    np.savetxt('voronoi_2d_binning_output.txt', np.column_stack([x, y, binNum]),
               fmt=b'%10.6f %10.6f %8i')
    np.savetxt('bins.txt',np.column_stack([xNode,yNode]),fmt="%10.3g")
    nbins=xNode.shape[0]

    avspec=np.zeros((nbins,ws))
    avspecerr=np.zeros((nbins,ws))
    binflux=np.zeros(nbins)
    x=x.astype(int)
    y=y.astype(int)
    for j in range(nbins):
	b=np.where(binNum==j)[0]
	valbin=b.size
	if valbin == 1:
	    for i in range(ws):
		avspec[j,i]=gal_lin[i,y[b],x[b]]
		avspecerr[j,i]=gal_err[i,y[b],x[b]]
	else:
	    for i in range(ws):
		avspec[j,i]=np.sum(gal_lin[i,y[b],x[b]])/valbin
		avspecerr[j,i]=np.sum(gal_err[i,y[b],x[b]])/valbin
	binflux[j]=np.sum(avspec[j,:])

    np.savetxt('binflux.txt',binflux,fmt="%10.3g")

    hdu=pyfits.PrimaryHDU(avspec)
    hdu.writeto('galaxybinspy.fits',clobber=True)



#========================================================================================================================
#========================================================================================================================

def ppxfit(ncompfit,brot,bfract,mom):

    velscale=110.
    file = 'NGC0528-V500.rscube.fits'
    hdu = pyfits.open(file)
    gal_lin = hdu[0].data
    h1 = hdu[0].header

    medfl=np.loadtxt("medgalpy.txt")
    x = medfl[:,0]
    y = medfl[:,1]
    sig = medfl[:,2]
    noise = medfl[:,3]

    bins=np.loadtxt("voronoi_2d_binning_output.txt",skiprows=1)
    x = bins[:,0]
    y = bins[:,1]
    binnum = bins[:,2]

    binco=np.loadtxt("bins.txt")
    xbin = binco[:,0]
    ybin = binco[:,1]

    file = 'galaxybinspy.fits' # spectra arranged horizontally
    hdu = pyfits.open(file)
    gal_bin = hdu[0].data
    gs=gal_bin.shape
    nbins=gs[0]
    xcut=0.0
    ycut=0.0
    delta = h1['CDELT3']
    lamRange1 = h1['CRVAL3'] + np.array([xcut*delta,delta*((h1['NAXIS3']-1)-ycut)])
    FWHM_gal = 6.0 # CALIFA has an instrumental resolution FWHM of 6A.

    galaxyz, logLam1, velscale = util.log_rebin(lamRange1, gal_bin[0,:],velscale=velscale)


    galaxy= np.empty((galaxyz.size,nbins))
    noise= np.empty((galaxyz.size,nbins))

    for j in range(nbins):
	galaxy[:,j], logLam1, velscale = util.log_rebin(lamRange1, gal_bin[j,:],velscale=velscale)
	galaxy[:,j] = galaxy[:,j]/np.median(galaxy[:,j]) # Normalize spectrum to avoid numerical issues
	noise[:,j] = galaxy[:,j]*0 + 0.0049           # Assume constant noise per pixel here

    #dir='/home/ppxmt3/astro/MILES/'
    dir='galspec/'
    miles = glob.glob(dir + 'Mun*.fits')
    miles.sort()
    FWHM_tem = 2.5 # Miles spectra have a resolution FWHM of 1.8A.

    age=np.empty(len(miles))
    met=np.empty(len(miles))
    #age=np.chararray(len(miles),itemsize=7)
    #met=np.chararray(len(miles),itemsize=5)

    for j in range(len(miles)):
	ast=miles[j][22:29]
	mst=miles[j][17:21]
	pm=miles[j][16:17]
	if pm == 'p': pmn='+'
	elif pm =='m': pmn='-'
	mstpm=(pmn,mst)
	#met[j,:]=miles[j][16:19]
	age[j]=float(ast)
	met[j]=float("".join(mstpm))

    #age2,inda=np.unique(age,return_inverse=True)
    #met2,ind=np.unique(met,return_inverse=True)

    #c=1
    #for i in range(len(age2)/2):
	#indout=np.where(age==age2[c])[0]
	##print(indout)
	#miles=np.delete(miles,indout)
	#age=np.delete(age,indout)
	#c=c+2


    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the CALIFA galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.

    hdu = pyfits.open(miles[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0.,h2['CDELT1']*(h2['NAXIS1']-1)])
    sspNew, logLam2,velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)


    # Convolve the whole miles library of spectral templates
    # with the quadratic difference between the CALIFA and the
    # miles instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels miles --> CALIFA
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1'] # Sigma difference in pixels


#==========================================================================================================================
#One component fit - saves veloctiy values in 'NGC528_onecompkin.txt' to be used as initial estimates for two component fit


    if ncompfit == 1:

	templates = np.empty((sspNew.size,len(miles)))
	for j in range(len(miles)):
	    hdu = pyfits.open(miles[j])
	    ssp = hdu[0].data
	    ssp = ndimage.gaussian_filter1d(ssp,sigma)
	    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
	    templates[:,j] = sspNew/np.median(sspNew) # Normalizes templates

	c = 299792.458
	dv = (logLam2[0]-logLam1[0])*c # km/s

	vel = 4750. # Initial estimate of the galaxy velocity in km/s
	goodPixels = util.determine_goodpixels(logLam1,lamRange2,vel)

	start=np.zeros(2)
	output = np.zeros((nbins,5))
	output[:,0] = xbin[:]
	output[:,1] = ybin[:]

	start[:] = [vel, 3.*velscale] # (km/s), starting guess for [V,sigma]

	for j in range(nbins):
	    print('On ',j+1,' out of ',nbins)
	    print(start)
	    pp = ppxf(templates, galaxy[:,j], noise[:,j], velscale, start,
		      goodpixels=goodPixels,
		      degree=4, vsyst=dv, plot=True,moments=mom)
	    kinem=np.loadtxt("ppxfout.txt")
	    if mom==2:
	    	output[j,2]=kinem[0] #vel
	    	output[j,3]=kinem[1] #sigma
	    	output[j,4]=kinem[2] #chisq
	    if mom==4:
	    	output[j,2]=kinem[0] #vel
	    	output[j,3]=kinem[1] #sigma
	    	output[j,4]=kinem[4] #chisq

	np.savetxt('NGC528_onecompkinm2nch.txt',output,fmt="%10.3g")

#=========================================================================================================================
#Two component fit

    elif ncompfit == 2:

# To determine flux fraction of bulge. Set bfract to 0 to disable

# 'bulgediskblock.fits' is created by running GALFIT to get a galfit.01 file of best fit parameters, then using
# '>galfit -o3 galfit.01' to get cube of galaxy image, bulge image and disk image

	if bfract == 1:
	    file = 'bulgediskblock.fits'
	    hdu = pyfits.open(file)
	    galb = hdu[1].data
	    bulge = hdu[2].data
	    disk = hdu[3].data

	    # Bin bulge and disk images into same binning as datacube
	    nbins=xbin.shape[0]
	    avbulge=np.zeros(nbins)
	    avdisk=np.zeros(nbins)
	    avtot=np.zeros(nbins)
	    binflux=np.zeros(nbins)
	    x=x.astype(int)
	    y=y.astype(int)
	    for j in range(nbins):
		b=np.where(binnum==j)[0]
		valbin=b.size
		if valbin == 1:
		    avbulge[j] = bulge[y[b],x[b]]
		    avdisk[j] = disk[y[b],x[b]]
		    avtot[j] = galb[x[b],y[b]]
		else:
		    avbulge[j] = np.median(bulge[y[b],x[b]])
		    avdisk[j] = np.median(disk[y[b],x[b]])
		    avtot[j] = np.median(galb[x[b],y[b]])

	    bulge_fraction=avbulge/(avbulge+avdisk)

	    hdu=pyfits.PrimaryHDU(bulge_fraction)
	    hdu.writeto('bulge_fraction.fits',clobber=True)

#====================================================================================


	templates = np.empty((sspNew.size,2*len(miles)))
	ssparr = np.empty((ssp.size,len(miles)))

	for j in range(len(miles)):
	    hdu = pyfits.open(miles[j])
	    ssparr[:,j] = hdu[0].data
	    ssp = hdu[0].data
	    ssp = ndimage.gaussian_filter1d(ssp,sigma)
	    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
	    templates[:,j] = sspNew/np.median(sspNew) # Normalizes templates
	for j in range(len(miles),2*len(miles)):
	    hdu = pyfits.open(miles[j-len(miles)])
	    ssp = hdu[0].data
	    ssp = ndimage.gaussian_filter1d(ssp,sigma)
	    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
	    templates[:,j] = sspNew/np.median(sspNew) # Normalizes templates


	component = np.zeros((2*len(miles)),dtype=np.int)
	component[0:len(miles)]=0
	component[len(miles):]=1

	c = 299792.458
	dv = (logLam2[0]-logLam1[0])*c # km/s
	vel = 4750. # Initial estimate of the galaxy velocity in km/s
	goodPixels = util.determine_goodpixels(logLam1,lamRange2,vel)

	kin=np.loadtxt('NGC528_onecompkinnch.txt')
	xbin = kin[:,0]
	ybin = kin[:,1]
	vpxf = kin[:,2]
	spxf = kin[:,3]
	occh = kin[:,4]
	velbulge=vel
	sigdisk=50.

	file = 'bulge_fraction.fits' # Read out bulge fraction for each bin
	hdu = pyfits.open(file)
	bulge_fraction = hdu[0].data

	bvel = np.zeros(nbins)
	bsig = np.zeros(nbins)
	bh3 = np.zeros(nbins)
	bh4 = np.zeros(nbins)
	dvel = np.zeros(nbins)
	dsig = np.zeros(nbins)
	dh3 = np.zeros(nbins)
	dh4 = np.zeros(nbins)
	bwt = np.zeros(nbins)
	dwt = np.zeros(nbins)
	output = np.zeros((nbins,10+(2*(mom-2))))
	popoutput = np.zeros((nbins,6))
	output[:,0] = xbin[:]
	output[:,1] = ybin[:]
	popoutput[:,0] = xbin[:]
	popoutput[:,1] = ybin[:]

	bmet = np.zeros(nbins)
	bage = np.zeros(nbins)
	dage = np.zeros(nbins)
	dmet = np.zeros(nbins)
	count=0
	for j in range(2,nbins-1):
	    print('Bin number:',j+1,'out of',nbins)
	    print('Bulge fraction:',bulge_fraction[j])
	    if spxf[j] > 350: spxf[j] = 350.
	    if abs(vpxf[j]-4750.) > 300: vpxf[j] = 4750.

	    #start = np.array([[vpxf[j], spxf[j]],[vpxf[j],sigdisk]]) # (km/s), starting guess for [V,sigma]

	    start = np.array([[velbulge, spxf[j]],[vpxf[j],sigdisk]]) # (km/s), starting guess for [V,sigma]
	    print('Starting velocity estimates:',start[0,0],start[0,1],start[1,0],start[1,1])
	    print('Xbin:',xbin[j],'Ybin:',ybin[j])
	    t = clock()
	    pp = ppxf(templates, galaxy[:,j], noise[:,j], velscale, start, bulge_fraction=bulge_fraction[j],
		      goodpixels=goodPixels, moments=[mom,mom],
		      degree=4, vsyst=dv,component=component,brot=1,plot=True) #brot=0 for nonrotating, brot=1 for rotating
	#Kinematics

	    kinem=np.loadtxt("ppxfout.txt")
	    wts=np.loadtxt("ppxfoutwt.txt")

	    if mom == 2:
		output[j,2]=kinem[0,0] #bvel
		output[j,3]=kinem[0,1] #bsig
		output[j,4]=kinem[1,0] #dvel
		output[j,5]=kinem[1,1] #dsig
		output[j,6]=wts[0] #bulge weight
		output[j,7]=wts[1] #disk weight
		output[j,8]=wts[2] #chisqn
		output[j,9]=wts[3] #chisq

	    if mom == 4:
		output[j,2]=kinem[0,0] #bvel
		output[j,3]=kinem[0,1] #bsig
		output[j,4]=kinem[0,2] #bh3
		output[j,5]=kinem[0,3] #bh4
		output[j,6]=kinem[1,0] #dvel
		output[j,7]=kinem[1,1] #dsig
		output[j,8]=kinem[1,2] #dh3
		output[j,9]=kinem[1,3] #dh4
		output[j,10]=wts[0] #bulge weight
		output[j,11]=wts[1] #disk weight
		output[j,12]=wts[2] #chisqn
		output[j,13]=wts[3] #chisq
		print(wts[0],wts[1])
		print('Chisq difference from one comp fit (pos = improved)',occh[j]-wts[2])
		if occh[j] > wts[2]: count=count+1
	    bwt[j]=wts[0]
	    dwt[j]=wts[1]

	    #Populations

	    #wtsb=np.loadtxt("ppxfoutwtsb.txt")
	    #wtsd=np.loadtxt("ppxfoutwtsd.txt")
	    #shwb=wtsb.shape
	    #shwd=wtsd.shape

	    #if len(shwb) > 1:
		#bulgewt=np.array(wtsb[0,:])
		#bulgewt=bulgewt/bulgewt.sum()
		#bulgewtin=np.array(wtsb[1,:],dtype=int)
	    #else:
		#bulgewt=1.
		#bulgewtin=np.int(wtsb[1])
	    #if len(shwd) > 1:
		#diskwt=np.array(wtsd[0,:])
		#diskwt=diskwt/diskwt.sum()
		#diskwtin=np.array(wtsd[1,:],dtype=int)
	    #else:
		#diskwt=1.
		#diskwtin=np.int(wtsd[1])

	    #bage[j]=np.dot(bulgewt,age[bulgewtin])
	    #bmet[j]=np.dot(bulgewt,met[bulgewtin])
	    #dage[j]=np.dot(diskwt,age[diskwtin])
	    #dmet[j]=np.dot(diskwt,met[diskwtin])

	    #popoutput[j,2]=bage[j]
	    #popoutput[j,3]=bmet[j]
	    #popoutput[j,4]=dage[j]
	    #popoutput[j,5]=dmet[j]


	    #Plots

	    #ssparr=templates
	    #print(gal_bin.shape)
	    #gal_bin=galaxy
	    #print(gal_bin.shape)
	    #bulgespec=ssparr[:,bulgewtin].dot(bulgewt)
	    #diskspec=ssparr[:,diskwtin].dot(diskwt)
	    #diskspec=diskspec/np.median(diskspec)
	    #bulgespec=bulgespec/np.median(bulgespec)
	    #plt.xlabel("Wavelength")
            #plt.ylabel("Counts")
	    #plt.plot(5*(gal_bin[goodPixels,j]/np.median(gal_bin[goodPixels,j])), 'k')
	    #plt.plot(3*(bulgespec[goodPixels]), 'r')
	    #plt.plot(2*(diskspec[goodPixels]), 'b')
	    #plt.plot(5*(bulgespec[goodPixels]+diskspec[goodPixels])/np.median(bulgespec[goodPixels]+diskspec[goodPixels]), 'g')
	    #plt.savefig('outfit')

	np.savetxt('NGC528conskinnchcheckbrot.txt',output,fmt="%10.3g")
	#np.savetxt('NGC528conspopbrotm4nchas.txt',popoutput,fmt="%10.3g")



if __name__ == '__main__':
    ppxfrunner()
