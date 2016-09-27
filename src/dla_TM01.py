# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 08:57:15 2016

@author: Max
@descriotion: cylindrical DLA structure with the following free parameters:
        a - inner radius of vacuum tube, at interface to dielectric
        b - outer radius of dielectric, at interface to PCM
        E_0 - amplitude of longitudinal electric field
"""
import numpy as np
from numpy import exp, cos, sin, sqrt
from matplotlib import pyplot as plt
from scipy.special import j0, j1, y0, y1, jn_zeros, yn_zeros, jv, jvp, yvp
# bessel functions limited to m= 0
from scipy.optimize import fsolve
# to find roots of implicite dispersion relation
# parameter of particles and constants
from scipy.constants import m_e, m_p, c, e


class CylindricDLA:
    def __init__(self, innerRadius=1.0, outerRadius=2.0 , epsilonR=1):
        '''
        innerRadius: radius of the vacuum area of the cylindrical waveguide
        outerRadius: radius at interface between dielectric and perfect conductor
        epsilonR: relative permittivity of dielectric
        '''
        self.a = innerRadius
        self.b = outerRadius
        self.epsR = epsilonR
    
    def invDispersion(self, omega):
        '''
        For a given frequency omega the wavevector k in longitudinal direction
        is computed
        '''
        if omega < 2.4/self.b * c:
            print("omega too small. Limited by cut-off frequency for vac. waveguide")
            return np.nan
        else:
            gam2 = self.gamma2_of_gamma1_w(omega)
            return sqrt(self.epsR * (omega/c)**2 - gam2**2)
        
    
    def dispersion(self, k ):
        '''
        Gives the frequency of a propagating wave inside the dla for a given k
        k: wave number in longitudinal direction
        '''
        gam2 = self.gamma2_of_gamma1(k)
        return c/sqrt(self.epsR) * sqrt(gam2**2 + k**2)
                
    
    # hidden methods ----------------------------------------------------------    
    # function of gamma2 whose roots are the cutoff freq.
     # TODO: use first besselfunction instead of derivative    
    __gam1= lambda self, gam2, k_: sqrt( abs( gam2**2/ self.epsR + k_**2*(1./self.epsR -1.) ))
    __f = lambda self, gam2, k_ : \
                gam2* jvp(0, self.__gam1(gam2, k_)*self.a  )*\
                ( j0(gam2*self.a) * y0(gam2*self.b) - j0(gam2*self.b) * y0(gam2*self.a) ) \
                - \
                self.__gam1(gam2, k_) * self.epsR * j0( self.__gam1(gam2, k_)*self.a)* \
                ( jvp(0, gam2* self.a) * y0(gam2*self.b) - j0(gam2*self.b)* yvp(0,gam2*self.a) ) 
    
    # use if k is large 
    __f2= lambda self, gam2, k_ : \
            gam2* np.imag(jvp(0, 1j* self.__gam1(gam2, k_)* self.a  )) *\
            ( j0(gam2* self.a) * y0(gam2*self.b) - j0(gam2*self.b) * y0(gam2* self.a) ) \
            - \
            self.__gam1(gam2, k_) * self.epsR * np.real(jv(0, 1j* self.__gam1(gam2, k_)* self.a) ) * \
            ( jvp(0, gam2 * self.a) * y0(gam2*self.b) - j0(gam2*self.b)* yvp(0,gam2* self.a) )\
    
    # forthe usage with givenb omega ------------------------------------------------
    __gam1_w= lambda self, gam2, w_: sqrt( abs( (1- self.epsR) * (w_/c)**2 + gam2**2 ))
    __f_w = lambda self, gam2, w_ : \
                gam2* jvp(0, self.__gam1_w(gam2, w_)*self.a  )*\
                ( j0(gam2*self.a) * y0(gam2*self.b) - j0(gam2*self.b) * y0(gam2*self.a) ) \
                - \
                self.__gam1_w(gam2, w_) * self.epsR * j0( self.__gam1_w(gam2, w_)*self.a)* \
                ( jvp(0, gam2* self.a) * y0(gam2*self.b) - j0(gam2*self.b)* yvp(0,gam2*self.a) ) 
    
    # use if k is large 
    __f2_w= lambda self, gam2, w_ : \
            gam2* np.imag(jvp(0, 1j* self.__gam1_w(gam2, w_)* self.a  )) *\
            ( j0(gam2* self.a) * y0(gam2*self.b) - j0(gam2*self.b) * y0(gam2* self.a) ) \
            - \
            self.__gam1_w(gam2, w_) * self.epsR * np.real(jv(0, 1j* self.__gam1_w(gam2, w_)* self.a) ) * \
            ( jvp(0, gam2 * self.a) * y0(gam2*self.b) - j0(gam2*self.b)* yvp(0,gam2* self.a) )\
    

    def gamma2_of_gamma1(self, k, startpoint= None, returnGamma1=False):
        '''
        k - wave number along direction of propagation    
        # for given gamma1 the boundary condition implicitely gives gamma2
        # the condition is expressed as a function of gamma2 and the roots are looked for    
        # REMARK: if gamma1 should be returned, the returned tuple contains 
            gamma2 first and gamma1 as the second element
        '''
        # for given gamma1 the boundary condition implicitely gives gamma2
        # the condition is expressed as a function of gamma2 and the roots are looked for
        
        # starting point ?
        print("startpoint: ", startpoint)
        if startpoint is None:
            startpoint = 2.4/self.b
        if (startpoint**2/ self.epsR + k**2*(1./self.epsR -1.) ) < 0  :
            # gamma1 is imaginary
            gamma2 = fsolve( self.__f2, startpoint, args=(k) )
        else:
            gamma2 = fsolve( self.__f, startpoint, args=(k) )
        if returnGamma1:
            return gamma2, self.__gam1(gamma2, k)
        else:
            return gamma2
            
    def gamma2_of_gamma1_w(self, w, startpoint= None, returnGamma1=False):
        '''
        w - omega as the angular frequency of the em wave
        # for given gamma1 the boundary condition implicitely gives gamma2
        # the condition is expressed as a function of gamma2 and the roots are looked for    
        # REMARK: if gamma1 should be returned, the returned tuple contains 
            gamma2 first and gamma2 as the second element
        '''
        # for given gamma1 the boundary condition implicitely gives gamma2
        # the condition is expressed as a function of gamma2 and the roots are looked for
        
        # starting point ?
        print("startpoint: ", startpoint)
        if startpoint is None:
            startpoint = 2.4/self.b
            
        if (startpoint**2 + (1-self.epsR) * (w/c)**2 ) < 0  :
            # gamma1 is imaginary
            gamma2 = fsolve( self.__f2_w, startpoint, args=(w) )
        else:
            gamma2 = fsolve( self.__f_w, startpoint, args=(w) )
        if returnGamma1:
            return gamma2, self.__gam1_w(gamma2, w)
        else:
            return gamma2
    
        
def plotDispersion(dla, k= np.linspace(40,1500,100),
                   label="by gamma2", axes=None, line=None,
                   incFreeSpace=False, incFullLoading= False, wOut=False,
                   boolLegend=False, color=None):
    '''
    Plots the Dispersion relation omega(k) of the dla, meaning the dependence of
    the frequency on the longitudinal wave vector.
    
    Parameters
    --------------------------
    dla
      *CylindricDLA*. Object representing the dielectric structure
    
    k 
      *array*. Gives the range of the wavevector for which the dispersion will
      be plotted.
    
    label
      *String*. A text which is used for desribing the plotted line. If 
      it's not specified, the default label will be \"by gamma2 \". This 
      specifies that the transcendental equation is solved for gamma2 for
      each given longitudinal wave vector
      
    axes
      *matplotlib axes object*. An Axes object to which the line will be 
      plotted. If it's not given or **None**, a new figure and axes are 
      created.
     
    line
      *matplotlib line 2d object*. If specified the existing line object will 
      represent the new plot. If not specified, a new line is added. it requires
      the axes parameter. If not given the line parameter will be ignored.
      The color parameter will be ignored
    
    incFreeSpace
      *bool*. Plot free space dispersion for comparison
      
    incFullLoading
      *bool*. Plot fully loaded dispersion additionally
    
    Returns
    --------------------------
    
    axes 
      *matplotlib axes object*. The returned Axes object can be used to plot
      further dispersion relations on the same figure.               
   '''
    gamma2Data= np.zeros(len(k))
    gamma2Data[0] = dla.gamma2_of_gamma1(k[0])
    # unfortunately loop needed since startpoint depends on previous result
    for i in range(1,len(k)):
        gamma2Data[i] = dla.gamma2_of_gamma1(k[i], gamma2Data[i-1] )

    # omega
    yData= c/sqrt(dla.epsR) * sqrt(gamma2Data**2 + k**2)
    # freq. f in units of GHz
    yData = yData/(2*np.pi*1e9)
    if (not line is None) and (not axes is None):
        line.set_data(k, yData)
        ylimMax = np.max(yData) * 1.2
        axes.set_ylim(ymax=ylimMax)
        return axes
    
    #-------the following is only executed if line is not given -------------------
    if axes is None:             
        plt.figure()
        axes= plt.gca()
        print("a={0:1.2e}, b={1:1.2e}, epsR={2}".format(dla.a, dla.b, dla.epsR))
        axes.set_title("a={0:1.2e}, b={1:1.2e}, epsR={2}".format(
            dla.a, dla.b, dla.epsR)
            )        
    axes.set_xlabel("k in 1/m")
    axes.set_ylabel(r"f in GHz")  

    if incFreeSpace:
        axes.plot(k, c*k*1e-9/(2*np.pi), label="free space")
    if incFullLoading:
        axes.plot(k,
              1/(2*np.pi)*c/sqrt(dla.epsR) * sqrt(k**2 + (2.4/dla.b)**2) * 1e-9,
              label="full loading")   
    labelStr=label
    
    if not color is None:
        axes.plot(k, yData, label=labelStr, color=color)
    else:
        axes.plot(k, yData, label=labelStr)

    ylimMax = np.max(yData) * 1.2
    axes.set_ylim(ymax=ylimMax)      
   
    if boolLegend is True:
        axes.legend(loc=0)
    
    if wOut:
        return yData
    else:
        return axes
        

def plotInverseDispersion(dla,  omega= np.linspace(2*np.pi* 26e9, 2*np.pi* 43e9,100), label=None, axes=None):
        '''
        Plots the Dispersion relation omega(k), meaning the dependence of
        the frequency on the longitudinal wave vector.
        
        Parameters
        --------------------------
        omega 
          *array*. Gives the range of the angular frequency for which the 
          inversion will be plotted
        
        label
          *String*. A text which is used for desribing the plotted line. If 
          it's not specified, the default label will be \"by gamma2 \". This 
          specifies that the transcendental equation is solved for gamma2 for
          each given longitudinal wave vector
          
        axes
          *matplotlib axes object*. An Axes object to which the line will be 
          plotted. If it's not given or **None**, a new figure and axes are 
          created.
        
        Returns
        --------------------------
        
        axes 
          *matplotlib axes object*. The returned Axes object can be used to plot
          further dispersion relations on the same figure.               
       '''
        w= omega       
        gamma2Data= np.zeros(len(w))
        gamma2Data[0] = dla.gamma2_of_gamma1_w(w[0])
        # unfortunately loop needed since startpoint depends on previous result
        for i in range(1,len(w)):
            gamma2Data[i] = dla.gamma2_of_gamma1_w(w[i], gamma2Data[i-1] )

        # wavevector k
        yData= sqrt( dla.epsR *(w/c)**2 - gamma2Data**2 )

        if axes is None:             
            plt.figure()
            plt.xlabel(r"$\omega$ in rad*GHz" )
#            plt.ylabel(r"$\omega$ in GHz")
            plt.ylabel("k in 1/m")
            
        if label is None:
            labelStr="by gamma2"
        else:
            labelStr=label
        plt.plot(w *1e-9, yData, label=labelStr)        
#       plt.plot(k, yData1/1e9, label="by gamma1")    
        if axes is None:        
            plt.plot(w*1e-9, w/c, label="free space")
            plt.plot(w* 1e-9, sqrt(dla.epsR * (w/c)**2 - (2.4/dla.b)**2 ), label="full dielec.")            
            plt.plot(w* 1e-9, sqrt( (w/c)**2 - (2.4/dla.b)**2) , label="vacuum")            
            plt.legend()
            plt.title("a={0:1.2e}, b={1:1.2e}, epsR={2}".format(dla.a, dla.b, dla.epsR))
#            plt.ylim(ymax=max(yData/(2*np.pi*1e9)))
            plt.show()
    
    
    
    
if __name__== "__main__":
    dlaX = CylindricDLA(3e-3, 4.57e-3, 4)
#    dlaX.plotDispersion()
#    dlaX.plotInverseDispersion()
#    dlaX.plotDispersion(k=np.linspace(600, 1500, 100))
    # paper "Dresden"
    

    '''
    #### multiple values of a, inner radius

    a_set= np.array([459.99e-6, 450e-6, 400e-6, 350e-6, 300e-6, 250e-6  ])
    a_str= [r"$a=460 \mu m$", r"$a=450 \mu m$", r"$a=400 \mu m$", 
            r"$a=350 \mu m$", r"$a=300 \mu m$", r"$a=250 \mu m$"]
    dlaTHZ = CylindricDLA(a_set[0], 470e-6, 11.6)
    k= np.linspace(10, 1.9e4*1.2 , 200)
    #initial Plot    
    ax= plotDispersion(dlaTHZ, k, label=r"$a=200 \mu m$",
                       incFreeSpace=True, incFullLoading=True, wOut=False)
    for a, label in zip(a_set[1:], a_str[1:]):
        dlaTHZ.a= a
        ax= plotDispersion(dlaTHZ, k, axes=ax, label=label,
                       boolLegend=True)

#    dlaTHZ.plotDispersion(k = np.linspace(10,2.5e4, 200))
#    cax = plt.gca()
   
#    dlaX.invDispersion(24e9 * 2 *np.pi)        
    '''    
    
    
    k=1.9e4
    dlaTHZ = CylindricDLA(200e-6, 470e-6, 11.67)
    
    omega = dlaTHZ.dispersion(k)
    fields = EBField(dlaTHZ, E_0= 1e8, k= k, phi=0.35)
    # phase velocity specified in units of c as well as omega
    omega = omega[0]/c
    v_p= omega/k
    print("Phase velocity in units of c: v_p= {0}".format(v_p))
    beta= v_p               # synchronicity assumed!!!
    mp = m_p*c**2/e         # in units of eV
    plotDispersion(incFreeSpace= True, incFullLoading=True, boolLegend=True)    

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    


    
    
    
    
    
    
    
    
    