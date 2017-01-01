# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:33:45 2016

@author: Max
"""


import numpy as np
from numpy import exp, cos, sin, sqrt
from matplotlib import pyplot as plt
from scipy.special import j0, j1, y0, y1, jn_zeros, yn_zeros, jvp, yvp, jv
# bessel functions limited to m= 0
from scipy.optimize import fsolve
# to find roots of implicite dispersion relation
# parameter of particles and constants
from scipy.constants import m_e, m_p, c, e, epsilon_0
#from dispersion import gamma2_of_gamma1, gam1, omegaByGamma2

from dla_TM01 import CylindricDLA


class EBField:
    def __init__(self, dla, E_0 = 1, k= None, omega=None, phi=0.):
        '''
        Creates the electric and magnetic field within a  cylindrical dielectric
        structure.
        
        Parameters
        --------------------------
        dla 
          *CylindricDLA*. Cylindrical dielectric structure with given inner 
          radius, outer radius and relative permittivity of dielectric loading
        
        E_0
          *float*. field amplitude at cylinder axis (r=0)
          
        k
          *float*. longitudinal wavevector as independent quantity. frequency 
          is given by the dispersion
         
        omega
          *float*. angular frequency as independent variable. Either k or omega
          is given, but 
        
        phi
          *float*. Phase at z=0 compared to cos behavior. This means in 
          longitudinal direction the field has a cos(kz + phi) shape
          
        Returns
        --------------------------
        '''
        #make sure either k or omega is given
        assert( (k is None) != (omega is None) )
        self.E_0 = E_0
        self.dla = dla
        self.phase = phi
        if omega is None:
            self.k = k
            self.gamma_2, self.gamma_1 = dla.gamma2_of_gamma1(k, returnGamma1=True)
            self.omega = (c/sqrt(dla.epsR) * sqrt(self.gamma_2**2 + k**2))[0]
        elif k is None:
            self.omega = omega
            self.gamma_2, self.gamma_1 = dla.gamma2_of_gamma1_w(omega, returnGamma1=True)
            self.k = sqrt( dla.epsR *(omega/c)**2 - self.gamma_2**2 )
        self.phaseVelocity = self.omega/self.k    *1/c            
        # if omega is below the free  space dispersion gamma1 will be imaginary
        self.boolLowOmega = self.omega < c*self.k
        if self.boolLowOmega:
            self.gamma_1 = 1j * self.gamma_1
        #set amplitude such that E_z matches coming from both sides
        self.amp2 = E_0 * jv(0, self.gamma_1 * dla.a) * y0(self.gamma_2* dla.b)/\
            (j0(self.gamma_2 * dla.a) * y0(self.gamma_2 * dla.b) - \
            j0(self.gamma_2*dla.b) * y0(self.gamma_2 * dla.a))
        # period in units of 1/c
        self.period = 2*np.pi/(self.omega) *c 
        
        self.wavelength = 2*np.pi/self.k
        
    def phase_degree(self):
        return self.phase/(2*np.pi) * 360
        
    def __osc(self,z,t):
        return exp(1j* (self.k *z - self.omega*t + self.phase))

    def osc(self,z,t):
        return exp(1j* (self.k *z - self.omega*t + self.phase))
        
    def E_z(self,r,z,t):
        '''
        full dependence on r,z,t
        '''  
        a= self.dla.a
        b= self.dla.b
        E_0 = self.E_0                
        inVac = lambda r: E_0 * jv(0, self.gamma_1 *r)
        inDielec = lambda r:  self.amp2*( j0( self.gamma_2 *r) - 
            jv(0,self.gamma_2 * b)/y0(self.gamma_2*b) * y0( self.gamma_2 *r) )
        
        # make an array. requiered if argument r is not an array but a float
        r=np.reshape(r,-1)
#        assert( (r<= b).all() )  # make sure all provided values are in the waveguide
        datapointsVac = r[np.where( r< a)]
        datapointsDielec= r[np.where( r >= a)]
        radial = np.hstack((inVac(datapointsVac), inDielec(datapointsDielec) ))
        return np.real( radial * self.__osc(z,t))
        
    def E_r(self,r,z,t):
        a= self.dla.a
        b= self.dla.b
        inVac = lambda r: 1j * self.k/ self.gamma_1 * self.E_0 * jvp(0, self.gamma_1 *r)
        inDielec = lambda r:  1j* self.k * self.amp2 / self.gamma_2 *\
            ( jvp(0, self.gamma_2 *r)\
            - j0(self.gamma_2 * b)/y0(self.gamma_2*b) * yvp(0, self.gamma_2 *r) )
        # make an array. requiered if argument r is not an array but a float
        r=np.reshape(r,-1)
#        assert( (r<= b).all() )  # make sure all provided values are in the waveguide
        datapointsVac = r[np.where( r< a)]
        datapointsDielec= r[np.where( r >= a)]
        radial = np.hstack((inVac(datapointsVac), inDielec(datapointsDielec) ))
        return np.real( radial * self.__osc(z,t))
        
    def B_phi(self,r,z,t):
        w = self.omega
        a= self.dla.a
        b= self.dla.b
        epsR = self.dla.epsR
        inVac = lambda r: 1j * w/( c**2 * self.gamma_1) * self.E_0 * jvp(0, self.gamma_1 *r)
        inDielec = lambda r:  1j*w/( c**2 * self.gamma_2) * epsR *  self.amp2 *\
            ( jvp(0, self.gamma_2 *r) - jv(0,self.gamma_2 * b)/y0(self.gamma_2*b) \
            * yvp(0, self.gamma_2 *r) )
        # make an array. requiered if argument r is not an array but a float
        r=np.reshape(r,-1)
#        assert( (r<= b).all() )  # make sure all provided values are in the waveguide
        # TODO: make more efficient
#        np.piecewise(x, [x < 0, x >= 0], [-1, 1])
        datapointsVac = r[np.where( r< a)]
        datapointsDielec= r[np.where( r >= a)]
        radial = np.hstack((inVac(datapointsVac), inDielec(datapointsDielec) )) 
        return np.real( radial * self.__osc(z,t) )
        
    def vectorPot_z(self, r, z, t):
        a= self.dla.a
        b= self.dla.b
        A_z_vac = lambda r: -1j/self.omega* self.E_0 * jv(0, self.gamma_1*r)
        A_z_diel = lambda r: -1j/self.omega* self.amp2 * \
            ( \
                 jv(0, self.gamma_2 * r) - \
                 jv(0, self.gamma_2 * b)/ y0( self.gamma_2 * b) *\
                 y0(self.gamma_2 * r)\
            )
        # make an array. requiered if argument r is not an array but a float
        r=np.reshape(r,-1)
        assert( (r<= b).all() )  # make sure all provided values are in the waveguide
        datapointsVac = r[np.where( r< a)]
        datapointsDielec= r[np.where( r >= a)]
        
        radial = np.hstack(( A_z_vac(datapointsVac), A_z_diel(datapointsDielec) )) 
        #include longitudinal and time dependence
        return np.real(radial * self.__osc(z,t))

        
    def vectorPot_r(self, r,z,t):
        A_r = k/(omega * gamma_1) * E_0 * jvp(0, gamma_1*r) * self.__osc(z,t)
        
    def internalU(self):
        '''
        The internal energy per length is computed by  
        
        *epsilon/2 * integrate | E |^2 dA*

        where the integral is taken over the cross section
        
        Returns
        --------------------------
        u
          *float*. stored energy per longitudinal length in the cylindric
          waveguide. it is returned in units of eV/m
        
        '''
        from scipy.integrate import quad
#        g1 = self.gamma_1
#        g2 = self.gamma_2
        b= self.dla.b
        E_0 = self.E_0
        E_z_Vac = lambda r: E_0 * jv(0, self.gamma_1 *r)
        E_z_Diel = lambda r:  self.amp2*( j0( self.gamma_2 *r) - 
            jv(0,self.gamma_2 * b)/y0(self.gamma_2*b) * y0( self.gamma_2 *r) )
        E_r_Vac = lambda r: 1j * self.k/ self.gamma_1 * self.E_0 * jvp(0, self.gamma_1 *r)
        E_r_Diel = lambda r:  1j* self.k * self.amp2 / self.gamma_2 *( j0( self.gamma_2 *r)\
        - j0(self.gamma_2 * b)/y0(self.gamma_2*b) * yvp(0, self.gamma_2 *r) )
        
        integrand= lambda r: np.real(E_z_Vac(r)**2 + E_r_Vac(r)**2)
        print(integrand(self.dla.a*0.1))        
        quad_vac, err = quad( integrand, 0, self.dla.a)
        print(quad_vac)
        integrand= lambda r: np.real(E_z_Diel(r)**2 + E_r_Diel(r)**2)
        quad_diel, err = quad(integrand, self.dla.a, self.dla.b)
        print("quad vac:", quad_vac)
        print("quad diel:", quad_diel)
        return epsilon_0/2*(quad_vac + self.dla.epsR*quad_diel )
    
    def powerloss(self):
        '''
        
        '''
        
        
        
        
        
        
def plotFieldsRadially(EBField, axes=None):
    rData = np.linspace(0.01*EBField.dla.b,  EBField.dla.b, 200)
    t0 = (2*np.pi)/EBField.omega * 0.25
    z_m = - EBField.phase / EBField.k
    print("z_m:", z_m)
    E_z= lambda r: EBField.E_z(r,z_m,0)
    E_r= lambda r: EBField.E_r(r,z_m,t0)            
    B_phi= lambda r: EBField.B_phi(r,z_m,t0)
    
    if axes is None:
        fig, ax= plt.subplots()
    else:
        ax= axes
    ax.set_xlabel("r in m")
    ax.set_ylabel("field strength [V/m]")
    marker_all = ""
    ax.plot(rData, E_z(rData), marker=marker_all, label=r"$E_z$")
    ax.plot(rData, E_r(rData), marker=marker_all, label=r"$E_r$")
    ax.plot(rData, c*B_phi(rData), marker=marker_all, label=r"$c B_{\phi}$")
    ax.grid()
    ax.legend(loc=0)
    ax.set_title("max. Fields")
    ax.axvline( EBField.dla.a , color="grey")
#    ax.figure.show() 
    

    
    
    
if __name__ =="__main__":
    dlaX = CylindricDLA(3e-3, 4.57e-3, 4)
#    dlaX.plotDispersion(k = np.linspace(10,1000, 200))
#    dlaX.plotDispersion()
    dlaTHZ = CylindricDLA(200e-6, 470e-6, 11.67)
    import matplotlib as mpl
    mpl.rc("text", usetex=False)
    field = EBField(dlaTHZ, E_0= 1e7, k=1.9e4, phi= 0.3)    
    '''    
    ax= plt.gca()
    import matplotlib.text as mtext
    ax.title.set_position((0.5,1.06))
    str_t = ax.get_title()    
    
    fontdict = {
     'verticalalignment': 'baseline'}    
    
    ax.set_title(str_t, fontdict)
    '''    
    lamb = 2*np.pi/field.k
    z_range =  np.linspace(0, lamb, 100)
#   def E_r(self,r,z,t):
    plt.figure()
    r_in_a = 0.3
    plt.plot(z_range, field.E_z(r_in_a*dlaTHZ.a, z_range, 0), label=r"$E_z$" )
    plt.plot(z_range, field.E_r(r_in_a*dlaTHZ.a, z_range, 0), label=r"$E_r$" )
    plt.plot(z_range, c*field.B_phi(r_in_a*dlaTHZ.a, z_range, 0), label=r"$c B_{\phi}$" )
    plt.legend()    
    
    plotFieldsRadially(field)        
    print("gamma1:", field.gamma_1)
    print("evaluate bessel derivative:",jvp(0, field.gamma_1 *0.5* dlaTHZ.a))
#    print("gamma1:", field.gamma_2)
    
    
    
    
#    field = EBField(dlaX, k=900, phi= np.pi/4.)    
#    dlaX.plotDispersion()
#    plotFieldsRadially(field)
#    field = EBField(dlaX, k=900, phi= np.pi/2.)    
#    dlaX.plotDispersion()
#    plotFieldsRadially(field)
#    print(field.gamma_1)
    print(field.internalU())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
       
       
       