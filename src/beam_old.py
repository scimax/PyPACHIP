# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 11:07:22 2016

@author: Max
"""

#from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import block_diag
from scipy.constants import m_e, m_p, c, e
from multiprocessing import pool

    
#------------------------------------------------------------
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      
    reference: http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    xmin = x_middle - plot_radius
    xmax = x_middle + plot_radius
    ymin = y_middle - plot_radius
    ymax = y_middle + plot_radius
    zmin = z_middle - plot_radius
    zmax = z_middle + plot_radius
    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])
    
    numTicks = 5
    dx = plot_radius/5
    
    # get the max and min values for the ticks at rounded position 
    xticks= np.linspace( float("{0:2.1e}".format(xmin)),
                         float("{0:2.1e}".format(xmax)),
                         numTicks)
    yticks= np.linspace( float("{0:2.1e}".format(ymin)),
                         float("{0:2.1e}".format(ymax)),
                         numTicks)
    zticks= np.linspace( float("{0:2.1e}".format(zmin)),
                         float("{0:2.1e}".format(zmax)),
                         numTicks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
#---------------------------------------------------------------------------

    
    
    


class Beam:
    def __init__(self, mass=1, beta=0.5):
        # this is a list of arrays
        self.particles = []
        self.width= np.ones(6)
        self.position= np.zeros(6)
        self.b0 = beta       # velocity of the reference trajectory/particle
        self.mass = mass       # in units of eV
        self.count=0
        
    def get_p0(self):
        return self.b0 * self.gamma0() * self.mass
    
    def gamma0(self):
        return 1./np.sqrt(1-self.b0**2)  
    def get_Ekin0(self):
        return self.mass*( 1/np.sqrt(1 - self.b0**2) - 1)
    def set_Ekin0(self,Ekin):
        self.b0 = np.sqrt(2* self.mass * Ekin + Ekin**2)/(self.mass + Ekin)
        
    def get_eps_x(self):
        '''
        transverse emittance
        '''
        cov_mat=np.cov(self.particles[:,0:2], rowvar=False)
                
        print("variance x:", cov_mat[0,0])
        print("variance px:", cov_mat[1,1])
        
        return self.width[0]*self.width[1]/self.position[5]
    
    def get_eps_y(self):
        '''
        transverse emittance
        '''
        cov_mat=np.cov(self.particles[:,2:4], rowvar=False)
                
        print("variance y:", cov_mat[0,0])
        print("variance py:", cov_mat[1,1])
        return self.width[2]*self.width[3]/self.position[5]
        
    def get_eps_z(self):
        '''
        long emittance
        '''
        cov_mat=np.cov(self.particles[:,4:6], rowvar=False)
                
        print("variance z:", cov_mat[0,0])
        print("variance pz:", cov_mat[1,1])
        print(np.sqrt(cov_mat[0,0]*cov_mat[1,1] - cov_mat[0,1]**2))
        print(self.width[4]*self.width[5])
#        return np.sqrt(cov_mat[0,0]*cov_mat[1,1] - cov_mat[0,1]**2)/self.position[5]
        return self.width[4]*self.width[5]/self.position[5]
        
    def get_alphax(self):
        pass        
#        return 

        
        
#    def get_EnergyDist(self):
        
    
    def addParticle(self, singleParticle):
        '''
        PRE: 
        singleParticle - 6 dimensional array containing the phase space
                        coordinates of the single particle in the following form
                        [x, px, y, py, z, pz]
        POST:
        adds the single particle vector to the set of all particles, which is called
        beam
        '''
        listTemp= list(self.particles)
        listTemp.append(singleParticle)
        self.particles = np.array(listTemp)
        self.count = self.count +1
        # check number of particles for consistency
        if self.count != len(self.particles):
            print("Number of particles is wrong! Reset the beam particles.")
        
    def removeParticle(self, removingParticle):
        return 0
        
    def removeByMaxTransverse(self, size):
        # boolean list
        particlePasses = np.sum(self.particles[:, 0:3:2]**2, axis=1) < size**2 
        numRemoved = self.count - np.sum(particlePasses)
        if numRemoved != 0:
            self.particles = self.particles[ particlePasses ]
            self.count  = np.sum(particlePasses)
        assert( self.count == len(self.particles))
        return numRemoved
        
    def createGaussian(self, number, *args):
        '''
        PRE:
            if particles already exist, remove them and add particles to the 
            beam which are Gaussian distributed
        number - number of particles
        *args - array containing the widths (sigma) of the distributions in all 6 
                coord. followed by their center (mu)
        POST: array of size number containing the particle's phase space coordinates in the 
            following order
            x,px,y,py,dz,pz
        
        '''
#        print(args)
        #default values for Gaussian parameters
        width = np.ones(6, dtype=float)
        position = np.zeros(6, dtype=float)
        # If only width specified beam is located at origin
        if len(args) == 6:       
            width = np.array(args)
            # TODO
        elif len(args) ==12:
            width= np.array(args[:6])
            position = np.array(args[6:])
        else:
            raise IndexError("Invalid number of arguments when creating Gaussian beam")
        
        # just an empty array
        particles= np.zeros((number, 6))
        # roll the gaussian dice
        # 
        # random.multivariate_normal
        # One of the following lines is sufficient
        # TODO: efficiency test
#        temp = map(random.normal, position, width, number*ones(6))
        temp2 = np.random.multivariate_normal(position, np.diag(width)**2, number)
        
        # temp is a list and ordered the other way round
        self.particles = temp2
        self.width = width
        self.position = position
        
        self.count= number
        #plot both just for testing
        '''        
        plt.figure()
        plt.plot(particles[:,0], particles[:, 1], "ro")
        plt.plot(temp2[:,0], temp2[:, 1], "bo")
        plt.show()
        '''       
        
    def plotxPx(self, boolReturnLimits = False, xlim= None, ylim= None):
        x= np.array(self.particles)[:,0]
        # here, y denotes px such that x,y represent the projection 
        # on the phase space in x
        y= np.array(self.particles)[:,1]
        plt.figure()
        plt.plot(x,y, "ro")
        xLow = self.position[0]- 2*self.width[0]
        xUp = self.position[0] + 2*self.width[0]
        pxLow = self.position[1]- 2*self.width[1]
        pxUp = self.position[1] + 2*self.width[1]
        plt.xlabel("x")
        plt.ylabel(r"$p_x$")
        
        if not (xlim == None or  ylim == None):
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.show()
        
        if boolReturnLimits:
            return plt.xlim(), plt.ylim()
        
    def accelerateWithDLA(self, ebField, numSteps =1000,
                          T_ob= None, numPeriods=10, ellipseParamsAfter= None  ):
        '''
        Acceleration over a given time T within the electromagnetic field 
        ebField is done via leapfrog method.
            
            
        Parameters
        --------------------------
        numSteps
          *int*. number of time steps for the leapfrog algorithm. The 
          timepoint set is a linear spaced range from 0 to the obseration time
          T_ob with numSteps elements.
          
        T_ob
          *float*. Observation time at which the final phase space configuration
          of the beam is considered. The time is considered in units of 1/c
          
        numPeriods
          *float*. If T_ob is not explicitely given, then the observation time
          is numPeriods times one period. The parameter is ignored if the T_ob
          parameter is given. Default value is 10 periods.
        
        ebField
          *EBField*. Object encapsulating the field functions. This gives the
          force acting on a particle at a certain point.
          
        ellipseParamsAfter
          if it's *None* the ellipse parameters alpha_x, beta_x, gamma_x are not
          returned. If it's an *int* it is the number of steps after which the
          ellipse parameters are computed. 
        
        Returns
        --------------------------
        if *ellipseParamsAfter* is None, there is no return value
        
        t_alpha_beta_gamma
          *array* of dimension Mx5, where M is determined by the ellipse Step 
          and the total number of integration steps. It's numSteps//ellipseParamsAfter +2
          because the starting point and the end point are included.
        '''
        
        if T_ob is None:
            T_ob = 2*np.pi/(ebField.omega) *numPeriods*c
        
        def force(x,y,z, beta, t):
            '''
            For usage with an array of particle phase space coordinates, N x 6, take 
            the transponierte
            force(*(beam.particles[:5, ::2]).T)
            
            
            Parameters
            --------------------------
            x,y,z 
              *array*. positions given as arrays all of the same length N. The length
              represents the number of particles to which the force will be applied
            
            beta
              *float*. longitudinal velocity of particles used for computing the 
              acting magnetic force.
              ASSUMPTION: the force due to transversal velocity is neglected. The
              change in velocity is low such that it is used as constant for computing
              the force.
              
            t
              *float*. Time at which the force caused by EM wave is evaluated. 
              In seconds
            
            Returns
            --------------------------
            
            F 
              *array*. array of dimension '(N x 3)' where the ith row is the force
              acting on the ith particle.
            
            '''
            if not np.isscalar(x):
                length= len(x)
                # x,y,z are requirecd to have same length,
                assert((len(x) == len(y)) and (len(x) == len(z)) )
            else:
                length=1        
            Phi = np.arctan2(y,x)
            R= np.sqrt(x**2 + y**2)
        #    print("radius: ",R)
        #    print("angle: ",Phi)
            E_r = ebField.E_r(R,z,t)
            B_phi = ebField.B_phi(R,z,t)
            # e_r x e_phi = e_z  ---- e_z x e_phi= - e_r
            
            #charge set to be q=1
            F_z = np.outer(np.array([0,0,1]), ebField.E_z(R,z,t) )
            F_tr= (E_r - beta*c*B_phi)*np.array([np.cos(Phi), np.sin(Phi), np.zeros(length)])
            # reshaping needs to be done for the case of scalar parameters
            F_tr=F_tr.reshape(F_z.shape)
            return F_z + F_tr
            
        def ellipseParams(x,px, p0):
            cov= np.cov([x, px/p0])
            eps_x = np.sqrt(cov[0,0]*cov[1,1] - cov[0,1]**2 )
            beta_x = cov[0,0]/eps_x
            alpha_x = -cov[0,1]/eps_x
#            gamma_x = (1+alpha_x**2)/beta_x
            rms_x = np.sqrt(cov[0,0])
            return eps_x, alpha_x, beta_x, rms_x
        
        # Acceleration -------------------
        n_t = numSteps       # number of time steps + starting point   
        print("observation time in units of [m/c]:", T_ob)
        print("observation time in units of [s]:", T_ob/c)
        t_set, dt = np.linspace(0, T_ob, n_t, retstep=True)
        
        # transpose the array, such that the shape is 3xN instead of Nx3
        # when passing as an arugment the array can be extracted
        position_i = self.particles[:, ::2 ].T
        momentum_i = self.particles[:, 1::2 ].T
        position_i1 = np.zeros(position_i.shape)
        momentum_i1= np.zeros(momentum_i.shape)
        
#        global pool
#        b0_arr = self.b0*np.ones(self.)
        
        if not ellipseParamsAfter is None:        
            if numSteps % ellipseParamsAfter == 0:
                ellipseEvolution = np.zeros((numSteps//ellipseParamsAfter, 5))
            else:
                ellipseEvolution = np.zeros((numSteps//ellipseParamsAfter +1, 5))
        
        for i,t in enumerate(t_set[:-1]):
            '''
            To use leap frog with momentum instead of velocity in the relativistic case,
            we need the energy of the particle in the half steps
            '''
#            F_i =   pool.map(force, )                      
            F_i = force( *position_i, self.b0, t/c)
            
            E_half_step= np.sqrt( self.mass**2 + \
                np.linalg.norm(momentum_i + F_i * dt/2, axis=0 )**2   )
            # the energy of a particle depends on all it's three momentum coordinates
            # so when applying a leapfrog step the energy is the same for all 3 
            #    directions
            E_half_step = np.outer(np.ones(3),E_half_step)
            
            position_i1 = position_i + dt*momentum_i/E_half_step\
                    + dt**2/2* F_i/ E_half_step
                
            F_i1= force( *position_i1, self.b0, t/c)
            
            momentum_i1 = momentum_i +\
                dt*( F_i1 + F_i)/2
            
            position_i = position_i1
            momentum_i = momentum_i1
            
            if (not ellipseParamsAfter is None) and (i % ellipseParamsAfter  ==0):
                j = i // ellipseParamsAfter
                p0_temp = np.mean(momentum_i[2])
#                print(p0_temp)
#                print(self.position[5])
                ellipseEvolution[j] = np.array([t,
                     *ellipseParams(position_i[0], momentum_i[0], p0_temp)] )
#                ellipseEvolution[j] = np.array([t,
#                     *ellipseParams(position_i[0], momentum_i[0], self.position[5])] )                    
            
        self.particles[:, ::2 ] = position_i.T
        self.particles[:, 1::2 ] = momentum_i.T  
        if not ellipseParamsAfter is None:
#            print(ellipseEvolution)
#            print(ellipseEvolution.shape)
            return ellipseEvolution
        #TODO: adjust beta0, the mean velocity
        
        
    def getEnergyDistribution(self):
        '''
        kinetic energy of the particles
        '''
        E = np.sqrt( self.mass**2 + np.sum(self.particles[:,1::2]**2, axis=1)) - self.mass
        return E
        
    def getMeanEnergyKin(self, boolStd = False):
        E= self.getEnergyDistribution()
        if boolStd:
            return E.mean(), E.std()
        else:
            return E.mean()
    
    def updateBeamParameters(self):
        '''
        updates the mean velocity based on the phase space distribution. 
        Therefore kin. energy is also updated
        '''
        p_absVal_square = sum(self.particles[:,1::2]**2, axis=1)        
        beta = np.sqrt(p_absVal_square/ ( self.mass**2 + p_absVal_square))
        self.b0 = np.mean(beta)
#        sqrt()        
  
def plotEDistribution(beam, x_scaling=1., x_scale_label="", axes=None, label=None):
    E = beam.getEnergyDistribution()
    hist, bins = np.histogram(E, 30)
    bincenters = (bins[:-1] + bins[1:])*0.5
    initial_E = beam.get_Ekin0()
#    print("initial E_kin:", initial_E)
#    print("mean E_kin:", beam.getMeanEnergyKin() )
#    print("bins;",bins)
#    print("bins w/out offset;",bins-initial_E)
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)      
    axes.set_xlabel(r"$\Delta E$ ["+x_scale_label+"eV]")
    axes.set_ylabel("counts")
#    ax.hist(E, 50, color="green", alpha=0.70)
    axes.plot((bincenters - initial_E)/x_scaling, hist, "x", label=label)
    axes.legend()
#    try:
#        fig.show()
#    except NameError:
#        print("'fig' not defined")
    return axes
    
def plotzPz(beam, legendLabel="", axes=None):
    '''
    Plot the longitudinal phase space around the mean momentum such that it's 
    centralized around 0. The position is also centralized at 0
    
    '''
    if axes is None:
        fig = plt.figure()
#        axes = fig.add_axes()
        axes= fig.add_subplot(111)
    axes.set_xlabel("z [m]")
    axes.set_ylabel(r"$p_z$ [eV]")
    z = beam.particles[:,4]
    pz = beam.particles[:,5]
    pz0 = np.mean(pz)
    z0 = np.mean(z)
#    print("z0=",z0)
#    axes.set_title(r"longitudinal, $p_0={0:1.2e}$ eV, $z_0={1:1.2}$ m".format(pz0,z0))
    axes.set_title(r"longitudinal")
    line, = axes.plot(z - z0, pz-pz0, marker="o", linestyle="")
    if legendLabel != "":
        line.set_label(legendLabel)
        axes.legend(loc=0)
    axes.figure.canvas.draw()
    return axes, pz0
    
def plotxPx(beam, legendLabel="", axes=None):
    '''
    Plot the transverse phase space, only horizontal component
    '''
    if axes is None:
        fig = plt.figure()
#        axes = fig.add_axes()
        axes= fig.add_subplot(111)
    axes.set_xlabel("x [m]")
    axes.set_ylabel(r"$p_x$ [eV]")
    x = beam.particles[:,0]
    px = beam.particles[:,1]
#    pz0 = np.mean(pz)
#    z0 = mean(z)
    axes.set_title(r"transverse phasespace")
    line, = axes.plot(x, px, marker="o", linestyle="")
    if legendLabel != "":
        line.set_label(legendLabel)
        axes.legend(loc=0)
    axes.figure.canvas.draw()
    return axes
        
def plotEllipseParams(axesList, paramsArray,):
    '''
    The Function plots the evolution of the beam ellipse parameters during the
    acceleration on the give axeses.
    
    axesList:
        *list of axes* of four elements. These axes have a shared x-axis which
        will be used for the time variable
    
    paramsArray:
        *array* 5xM where the first row contains the time points while the other
        rows are epsilon_x, alpha_x, beta_x the beam ellipse parameters adn finally
        sigma_x, the rms size of the beam.
    '''
#    labels= [r"$\epsilon_x$", r"$\alpha_x$" , r"$\beta_x$", r"$\gamma_x$" ]
    labels= [r"$\epsilon_x$", r"$\alpha_x$" , r"$\beta_x$", r"$\sigma_x$" ]
    colors=['b','g', 'r', 'c']
    for i, axes in  enumerate(axesList):
        axes.plot(paramsArray[:,0], paramsArray[:,i+1], color=colors[i])
        axes.set_ylabel( labels[i] )

    for tl in axesList[1].get_yticklabels():
        tl.set_color(colors[1])
    for tl in axesList[2].get_yticklabels():
        tl.set_color(colors[2])
    
 
       
if __name__=="__main__":
#    import matplotlib as mpl
#    mpl.rc("axes.formatter", limits=(-3,3))    
#    mpl.rc("axes.formatter", use_mathtext=True)
    from matplotlib import rc
    rc("axes.formatter", limits=(-3,3))    
    rc("axes.formatter", use_mathtext=True)

    beam = Beam( mass= m_p/e*c**2, )
    beam.b0= 0.5
    
    p0= beam.b0* beam.gamma0() * beam.mass
    
    from EBFields import EBField
    from dla_TM01 import CylindricDLA
    
    dlaTHZ = CylindricDLA(200e-6, 470e-6, 5)
    k = 1.9e4
    fields = EBField(dlaTHZ, E_0= 1e9, k=k)
    omega = dlaTHZ.dispersion(k)
    
    # phase velocity specified in units of c as well as omega
    omega = omega[0]/c
    v_p= omega/k
    print("Phase velocity in units of c: v_p= {0}".format(v_p))
    beta= v_p              # synchronicity assumed!!!
    beam.b0 = beta
    
    N_p= 500  
    mp = m_p*c**2/e
    p0= beam.b0* beam.gamma0() * mp
    
    s_tr= 0.1*dlaTHZ.a
    s_z = 0.1*dlaTHZ.a
    s_z = 1e-6
    eps_tr = 1e-8      # in mm*mrad
    eps_z = 1e-8    # in mm*mrad

    beam.createGaussian(N_p, 
                    s_tr, eps_tr/s_tr * p0 ,
                    s_tr, eps_tr/s_tr * p0 ,
                    s_z, eps_z/s_z*p0,
                    0, 0, 0, 0, 0, p0 )
#    ''' 
#    
#    print("observation time:", 2*np.pi/(omega*c) *10)    
#
#    gs1=plt.GridSpec(3,2)
#    fig= plt.figure(figsize=plt.figaspect(0.5) )
#    ax = fig.add_subplot( gs1[:,0] )
#              
#    ax= plotEDistribution(beam,  axes=ax, label="initial")
#    ellipseEvolution = beam.accelerateWithDLA(fields, numPeriods=10, ellipseParamsAfter = 15)  
#    ax= plotEDistribution(beam,  axes=ax, label="final")
##    '''        
#    
#    gs2 = plt.GridSpec(3,2, hspace=0.18)
##    fig, axarr = plt.subplots(4, sharex=True)
#    print(ellipseEvolution[:,1])
#    print(ellipseEvolution.shape)
#    # first shared
#    ax1 = fig.add_subplot(gs2[0,1])
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    # add other shared ones
#    ax_twin1= fig.add_subplot(gs2[1,1], sharex=ax1)
#    ax_twin2= ax_twin1.twinx()
#    plt.setp(ax_twin1.get_xticklabels(), visible=False)
##    ax_twin1.xaxis.set_ticklabels([])    
#    ax3 = fig.add_subplot(gs2[2,1], sharex=ax1)
#    ax3.set_xlabel("t [m/c]")    
#    bottom_axes = [ax1, ax_twin1, ax_twin2, ax3]
#
#    # Hide shared x-tick labels
#    for axes in bottom_axes:
#        x_pos=axes.yaxis.get_label().get_position()[0]
#        print(axes.yaxis.get_label().get_position())
#        print(axes.get_yaxis().get_offset_text().get_position())        
#        axes.get_yaxis().get_offset_text().set_x(x_pos)
##        axes.get_yaxis().get_offset_text().set_y(-0.18)
#        axes.get_yaxis().get_offset_text().set_rotation('vertical')
#        axes.locator_params(axis='y', nbins=6)
#    
#    bottom_axes[-1].locator_params(axis='y', nbins=6)

    
#    plotEllipseParams(bottom_axes, ellipseEvolution)
    
#    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
#    '''
    
#    '''
#    ax= plotEDistribution(beam, x_scaling=1e6, x_scale_label="M", label="initial")
#    ax2, pz0= plotzPz(beam, "initial")
#    print("before: pz0= ", pz0)

#    '''
    # PLOT in longitudinal direction with field
    lamb = 2*np.pi/fields.k
    z_range =  np.linspace(0, lamb*10, 100)
    plt.figure()
    plt.plot(z_range, fields.E_z(0, z_range, 0), label=r"$E_z$" )
    z0_pos = beam.particles[:,4]    
    plt.plot(z0_pos, np.zeros(N_p), "bo")
    
    beam.accelerateWithDLA(fields, numPeriods=10)

    z1_pos = beam.particles[:,4]
    plt.plot(z1_pos, np.zeros(N_p), "ro")
#    '''    
    
#    ax2, pz0 = plotzPz(beam, "final", ax2)
#    print("after: pz0= ", pz0)    
    
    '''
    # DLA: length, inner radius, outer radius
    l= 1.0      # in m
    a=sqrt(1.0) # in m
    b= 2.0      # in m
    beam.dla(l,a,b)
    beam.plotxy(False, xlim, ylim) 
    print(len(beam.particles))
    '''
