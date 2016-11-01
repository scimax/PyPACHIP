# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:01:43 2016

@author: Max
"""

"""
An example of a Model-View-Controller architecture,
using wx.lib.pubsub to handle proper updating of widget (View) values.
"""
# for ui
import wx
#from wx.lib.pubsub import setupkwargs
from wx.lib.pubsub import pub

# for plotting
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   #for nice plots TODO: neccessary??
import numpy as np
from scipy.constants import m_e, m_p, c, e, hbar

from threading import Thread

import traceback   # for appropriate output of errors

# own imports
from dla_TM01 import CylindricDLA, plotDispersion
from EBFields import EBField, plotFieldsRadially
from beam_old import Beam
from beam_old import set_axes_equal, plotEDistribution, plotzPz, plotxPx, plotEllipseParams


class Model: #(threading.Thread): 
    def __init__(self):
#        threading.Thread.__init__(self) 
        self.dlaTHZ = CylindricDLA(200e-6, 470e-6, 11.67)
        k=1.9e4
        omega = self.dlaTHZ.dispersion(k)
        self.fields = EBField(self.dlaTHZ, E_0= 1e8, k= k, phi=0.35)
        # phase velocity specified in units of c as well as omega
        omega = omega[0]/c
        v_p= omega/k
        print("Phase velocity in units of c: v_p= {0}".format(v_p))
        beta= v_p               # synchronicity assumed!!!
        mp = m_p*c**2/e         # in units of eV

#        self.beam = Beam(mp, beta, updateStatus)
        self.beam = Beam(mp, beta)
        N_p= 1000        
        p0= self.beam.get_p0()
        s_tr= 0.1*self.dlaTHZ.a
        s_z = 0.1*self.dlaTHZ.a
        s_z = 1e-6
        eps_tr = 1e-8      # in mm*mrad
        eps_z = 1e-8    # in mm*mrad
        self.beam.createGaussian(N_p, 
                        s_tr, eps_tr/s_tr * p0,
                        s_tr, eps_tr/s_tr * p0,
                        s_z, eps_z/s_z*p0,
                        0, 0, 0, 0, 0, p0 )  
        # used for plotting the evolution of the transverse ellipse parameters
        self.ellipseEvolution = np.zeros((5,1))
        
        self.simulationStatus = 0


        
        #        beta* self.beam.gamma0() * mp
        #create the particle distribution. all quantities are centered at 0 with a
        # spread of 0.1 except for the longitudinal momentum p_z
        # number of particles
        
        # TODO: size in phase space
        
#        self.beam.createGaussian(N_p, 
#                        0.1*self.dlaTHZ.a, 0.1*self.dlaTHZ.a,
#                        0.1*self.dlaTHZ.a, 0.1*self.dlaTHZ.a,
#                        0.1*self.dlaTHZ.a, 0.1,
#                        0, 0, 0, 0, 0, p0 )
        
        '''        
        s_tr= 0.1*self.dlaTHZ.a
        s_z = 0.1*self.dlaTHZ.a
        s_z = 0.1
        eps_tr = 1
        eps_z = 1
        self.beam.createGaussian(N_p, 
                        s_tr, eps_tr/s_tr,
                        s_tr, eps_tr/s_tr,
                        s_z, eps_z/s_z,
                        0, 0, 0, 0, 0, p0 )
        '''
        
    def updateStatus(self):
        pass
        
    def set_dla(self, a, b, epsR):
       self.dlaTHZ.a = a
       self.dlaTHZ.b = b
       self.dlaTHZ.epsR = epsR
#       pub.sendMessage("MSG: DLA changed")
       
    def set_field(self, wavelength, amplitude, phase):
        self.fields = EBField(self.dlaTHZ, E_0= amplitude, 
                              k= 2*np.pi/wavelength, phi= phase*2*np.pi/360 )
#        pub.sendMessage("MSG: Field changed")
    
    def set_beam(self, N_p, E_kin, sx, eps_x, sy, eps_y, sz, eps_z):
        '''
        N_p
          *int*. number of particles
        
        E_kin
          *float*. mean kinetic energy 
          
        sx, eps_x
          *float*. rms width in x coordinate and emittance in x, assuming a
          decoupled beam. The rms in momentum space is computed from these values.

        sy, eps_y
          *flaot*. same as sx and eps_x, but for y component

        sz, eps_z
          *flaot*. same as sx and eps_x, but for z component
        '''
        # TODO: size in phase space
        # TODO: synchronistiy assumed
        # phase velocity specified in units of c as well as omega                
        self.beam.set_Ekin0(E_kin)
        p0= self.beam.b0* self.beam.gamma0() * self.beam.mass
        
        # rms in momentum space
        # before: 0.1*self.dlaTHZ.a,
        spx = eps_x/sx * p0
        spy = eps_y/sy * p0
        spz = eps_z/sz * p0
        
        self.beam.createGaussian(N_p,
                                 sx, spx,
                                 sy, spy,
                                 sz, spz,
                                 0, 0,
                                 0, 0,
                                 0, p0)
#        print("beam created")
        pub.sendMessage("MSG: Beam Changed")

# ui ------------------------------------------------    
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=20, height=4, dpi=100):
        fig = Figure(figsize=(width,height))
        
        gs1 = gridspec.GridSpec(2,3, left=0.07, bottom=0.08, right=0.93, top=0.92,
                    wspace=0.3, hspace=0.3)
        fig.add_subplot(gs1[0,0])
        fig.add_subplot(gs1[0,1])
        fig.add_subplot(gs1[0,2])
        fig.add_subplot(gs1[1,0])
        fig.add_subplot(gs1[1,1])
        
        ### Plots for Coutant-Snyder variables
        gs02 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs1[1,2], hspace=0.0)
        # first shared
        ax1 = fig.add_subplot(gs02[0,0])
        plt.setp(ax1.get_xticklabels(), visible=False)
        # add second one with twin axis
        ax_twin1= fig.add_subplot(gs02[1,0], sharex=ax1)
        ax_twin2= ax_twin1.twinx()
        plt.setp(ax_twin1.get_xticklabels(), visible=False)
        # add third one 
        ax3 = fig.add_subplot(gs02[2,0], sharex=ax1)
        ax3.set_xlabel("t [m/c]")
        # labeling
        labels= [r"$\epsilon_x$", r"$\alpha_x$" , r"$\beta_x$", r"$\gamma_x$" ]
        for i, axes in  enumerate([ax1, ax_twin1, ax_twin2, ax3]):
#            axes.plot(ellipseEvolution[:,0], ellipseEvolution[:,i+1])
            axes.set_ylabel( labels[i] )
            axes.margins(0.05)
            axes.get_yaxis().get_offset_text().set_x(-0.15)
            axes.get_yaxis().get_offset_text().set_rotation('vertical')
            
            

        '''
        Finally, there are 9 figures. 5 of them are distinct while the last four
        are subplots sharing the x axis.
        '''
#        fig.tight_layout(pad= 2.5)
        self.fig=fig
        FigureCanvas.__init__(self, parent, -1, self.fig)
#        gs1.update(left=None, bottom=None, right=None, top=None,
#                    wspace=0.3, hspace=0.3)        
        gs1.tight_layout(fig, pad=0.5)
#        for ax in fig.axes:
#            ax.title.set_position((0.5,1.06))
#        self.setParent(parent)
        
    def initialPlots(self, dla, field, beam):
        ### figure 1: Dispersion
        ax= self.fig.get_axes()[0]
        ax.set_title("Dispersion")
        k= np.linspace(10, field.k*1.2 , 200)
        plotDispersion( dla, k, axes=ax, label=r"$\omega(k)$",
                       incFreeSpace=True)
        self.lineDlaDispersion = ax.lines[1]
        self.lineOpPoint, = ax.plot( field.k, field.omega/(2*np.pi*1e9), "ro")
        self.lineBeamDispersion, = ax.plot( k, beam.b0 * c * k /(2*np.pi*1e9), label=r"$\omega=\beta c k$" )
        ax.legend(loc=0)        
        self.draw()
        
        ### figure 2: fields
        ax= self.fig.get_axes()[1]
        plotFieldsRadially(field, ax)
        ax.relim()
        ax.autoscale_view()
        self.draw()
        
        ### figure 3: long. phasespace
        ax= self.fig.get_axes()[2]
        plotzPz(beam, legendLabel= "initial", axes=ax)
        self.draw()
        
        ### figure 4: tr. phasespace
        ax= self.fig.get_axes()[3]
        plotxPx(beam, legendLabel= "initial", axes=ax)
        self.draw()
        
        ### figure 5: energy plot
        ax= self.fig.get_axes()[4]
#        plotEDistribution(beam, x_scaling=1e6, x_scale_label="M",
#                              label="initial", axes= ax)
        plotEDistribution(beam, label="initial", axes= ax)
        self.draw()      
        
    def update_dispersion_plot_2(self, dla):
        line = self.lineDlaDispersion        
        k= line.get_xdata()    
        plotDispersion(dla, k, label=r"$\omega(k)$",
                       axes= self.fig.get_axes()[0],
                        line= line)
        k_op = self.lineOpPoint.get_xdata()
        self.lineOpPoint.set_ydata( dla.dispersion(k_op)/(2*np.pi*1e9))
        
    def update_field_plot(self, field):
        # Field
        ax= self.fig.get_axes()[1]
        ax.clear()
        plotFieldsRadially(field, ax)
        ax.relim()
        ax.autoscale_view()
        self.draw()
        
    def update_phasespace_plot(self, beam):
        #TODO
        ax= self.fig.get_axes()[2]
        ax.clear()
        plotzPz(beam, legendLabel= "initial", axes=ax)
        ax= self.fig.get_axes()[3]
        ax.clear()
        plotxPx(beam, legendLabel= "initial", axes=ax)
        self.draw()
#        self.fig.tight_layout(pad= 2.5)
                
    def update_energy_plot(self, beam):
        ax= self.fig.get_axes()[4]
        ax.clear()        
        ax= plotEDistribution(beam,
                              label="initial", axes= ax)
        self.draw()    

    # ----------------
    def update_accelerated(self, model):
#        plotEDistribution(beam, x_scaling=1e6, x_scale_label="M",
#                          axes= self.fig.get_axes()[4],
#                            label="final")
        plotEDistribution(model.beam,
                          axes= self.fig.get_axes()[4],
                            label="final")
        plotzPz(model.beam, legendLabel="final",axes=self.fig.get_axes()[2])
        plotxPx(model.beam, legendLabel="final", axes=self.fig.get_axes()[3])
        plotEllipseParams(self.fig.get_axes()[5:], model.ellipseEvolution)
        self.draw()

    def update_OpPoint(self, k, omega):
        '''
        Update the operation point in the dispersion plot without replotting 
        the dla's dispersion
        '''
        self.lineOpPoint.set_data(k, omega/(2*np.pi*1e9))

    def update_BeamDisperion(self, beam):
        k= self.lineBeamDispersion.get_xdata()
        self.lineBeamDispersion.set_ydata(beam.b0 * c * k /(2*np.pi*1e9))
    
    def reset_ellipseParam_plot(self):
        try:
            for ax in self.fig.get_axes()[5:]:
                del ax.lines[0]
            self.draw()
        except IndexError:
            pass
    
class UIFrame(wx.Frame):
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, 
                    title="Proton Accelerator in Cylindric loaded Waveguide")
        panel= wx.Panel(self) 
        self.lineEditElements={}        
        # Main Sizer
        mainSizer = wx.BoxSizer(wx.HORIZONTAL)        
        # Form Sizer
        vboxLeft = wx.BoxSizer(wx.VERTICAL)
        flxGrid = wx.FlexGridSizer(15, 2, 5, 25 )     
        self.initForm(panel, flxGrid)
        flxGrid.AddGrowableCol(1, 1)
        vboxLeft.Add(flxGrid)        
        # buttons
        self.btn = wx.Button(panel, label="OK")
        self.btn_run = wx.Button(panel, label="Run")
        self.btn_animate = wx.Button(panel, label="3d Simulation")        
        self.btn_reset = wx.Button(panel, label="Reset")        
        spacersize = 20        
        vboxLeft.AddSpacer(20)        
        vboxLeft.Add(self.btn, flag= wx.CENTER, border = 5)
        vboxLeft.Add(self.btn_run, flag= wx.CENTER, border = 5)
        vboxLeft.Add(self.btn_reset, flag= wx.CENTER, border = 5)
        vboxLeft.AddSpacer(20)
        vboxLeft.Add(self.btn_animate, flag=wx.CENTER, border = 5 )
        self.btn_animate.Disable()
        vboxLeft.AddSpacer(20)        
        vboxLeft.Add(wx.StaticLine(panel), flag= wx.EXPAND, border= 5)        
        vboxLeft.AddSpacer(20)
        
        # implicite parameters
        flxGrid3 = wx.FlexGridSizer(5, 2, 5, 25 )
        self.initForm_implicite(panel, flxGrid3)
        vboxLeft.Add(flxGrid3)
        vboxLeft.AddSpacer(20)        
        vboxLeft.Add(wx.StaticLine(panel), flag= wx.EXPAND, border= 5)        
        vboxLeft.AddSpacer(20)
        
        # Result Form
        flxGrid2 = wx.FlexGridSizer(4, 2, 5, 25 )
        self.initResultForm(panel, flxGrid2)
        vboxLeft.Add(flxGrid2)
        
        # Sizer for Figure and Toolbar
        vboxRight = wx.BoxSizer(wx.VERTICAL)        
        self.canvas = MyMplCanvas(panel, width=15, height=9)    
        vboxRight.Add(self.canvas, 1, flag=wx.EXPAND | wx.ALL )                
        # All for toolbar
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        # By adding toolbar in sizer, we are able to put it at the bottom
        # of the frame - so appearance is closer to GTK version.
        vboxRight.Add(self.toolbar, 0,  wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()
        
        # Add all to the main sizer
        mainSizer.Add(vboxLeft, proportion=0, flag=wx.ALL, border=15)
        mainSizer.Add(vboxRight, 3, flag=wx.EXPAND | wx.ALL )        
        mainSizer.Fit(panel)
        panel.SetSizer(mainSizer)
        panel.Layout()
        self.sb = self.CreateStatusBar()
        mainSizer.Fit(self)
    
#    @staticmethod    
    def initForm(self, parent, flexGridSizer):
#        flexGridSizer
        # REM: when adding a field, remember to also add it to "short_options"
        # in the controller and a default value
        label_strings= (('inner Radius a [m]', 'a'),
                ('outer Radius b [m]', 'b'),
                ('rel. Permittivity epsR', 'epsR'),
                ('beam kinetic energy [eV]','E_kin'),
                ('number of particles', 'N_p'),
                ('beam size rms x [m]', 'sx'),
                ('emittance in x [m]', 'eps_x' ),
                ('beam size rms y [m]', 'sy'),
                ('emittance in y [m]', 'eps_y' ),
                ('beam size rms z [m]', 'sz'),
                ('emittance in z [m]', 'eps_z' ), 
                ('wavelength lambda [m]','lam'),
                ( 'center amplitude E_0 [V/m]', 'E_0'),
                ('phase shift [deg]', 'phi'),
                 ('time of observation in periods', 'num_obs')
                 )
        for i, (option, shortname) in enumerate(label_strings):
            label = wx.StaticText(parent, label=option)
            input_ = wx.TextCtrl(parent, id=i) 
#            input_.SetValue()
            flexGridSizer.Add(label, border=5)            
            flexGridSizer.Add(input_ , 1, wx.EXPAND) 
            self.lineEditElements[shortname] = input_
#        for key, ctrl in self.lineEditElements.items():
#            print("type of key '",key ,"':", type(ctrl))
        
    def initForm_implicite(self, panel, grid):
        '''
        show implicite information
        '''
        self.txtCtrl_beta= wx.TextCtrl(panel, style=wx.TE_READONLY)
        self.txtCtrl_omega = wx.TextCtrl(panel, style=wx.TE_READONLY)
        self.txtCtrl_phVelocity = wx.TextCtrl(panel,  style=wx.TE_READONLY)
        self.txtCtrl_wavevector = wx.TextCtrl(panel,  style=wx.TE_READONLY)
        self.txtCtrlEnergySpread = wx.TextCtrl(panel,  style=wx.TE_READONLY)
        grid.AddMany(
            [ (wx.StaticText(panel, label= "beta"), 0, 0, 5),
             (self.txtCtrl_beta),
             (wx.StaticText(panel, label= "angular freq. [Hz]"), 0, 0, 5),
             (self.txtCtrl_omega),
             (wx.StaticText(panel, label= "phase velocity [c]"), 0, 0, 5),
             (self.txtCtrl_phVelocity),
             (wx.StaticText(panel, label= "wave number [1/m]"), 0, 0, 5),
             (self.txtCtrl_wavevector),
             (wx.StaticText(panel, label= "energy width initial [eV]"), 0, 0, 5),
             (self.txtCtrlEnergySpread)
             ])
             
    def initResultForm(self,panel, grid):
        self.txtCtrl_energy_change = wx.TextCtrl(panel, style=wx.TE_READONLY)
        self.txtCtrl_loss = wx.TextCtrl(panel,  style=wx.TE_READONLY)
        self.txtCtrlEnergySpread_final = wx.TextCtrl(panel,  style=wx.TE_READONLY)
#        self.txtCtrl_Beta_z  = wx.TextCtrl(panel,  style=wx.TE_READONLY)
        
        grid.AddMany(
            [ (wx.StaticText(panel, label= "energy change [eV]"), 0, 0, 5),
             (self.txtCtrl_energy_change),
             (wx.StaticText(panel, label= "energy width final [eV]"), 0, 0, 5),
             (self.txtCtrlEnergySpread_final),
             (wx.StaticText(panel, label= "particle loss"), 0, 0, 5),
             (self.txtCtrl_loss)
#             ,
#             (wx.StaticText(panel, label= "Twiss beta x"), 0, 0, 5),
#             (self.txtCtrl_Beta_z)
             ])
             
    def showMsgBox(self, text):
        msgbox = wx.MessageDialog(None, text, 'Info', wx.OK)
        msgbox.ShowModal()
        msgbox.Destroy()

class Controller:
    def __init__(self):
        ### Model: 3 objects, dla, ebfield and beam
        self.model = Model()
        # lam = lambda
        self.short_options = ['a', 'b', 'epsR', 
                              'E_kin', 'N_p',
                              'sx', 'eps_x', 'sy', 'eps_y', 'sz', 'eps_z',
                              'lam', 'E_0', 'phi', 'num_obs']        
        # used to keep track which input fields where changed        
        self.txt_changed= np.zeros(len(self.short_options),dtype=bool)
        
        ### subscriber
        pub.subscribe(self.dlaChanged, "MSG: DLA changed")   
        pub.subscribe(self.fieldChanged, "MSG: Field changed")        
        pub.subscribe(self.beamChanged, "MSG: Beam Changed")
        pub.subscribe(self.beamAccelerated,"MSG: Acceleration done")
        
#        pub.subscribe( self. )
       
        ### View        
        self.frame = UIFrame()
                 
        ### Binding of EventHandlers
        self.frame.btn.Bind(wx.EVT_BUTTON, self.onClick)
        for key, txtCtrl in  self.frame.lineEditElements.items():
            txtCtrl.Bind(wx.EVT_TEXT, self.onTextChanged)
        self.frame.btn_run.Bind(wx.EVT_BUTTON, self.onClickRun)
        self.frame.btn_animate.Bind(wx.EVT_BUTTON, self.onClickAnimate)
        self.frame.btn_reset.Bind(wx.EVT_BUTTON, self.onClickReset)
        ### Binding for simple calculation of beta
        self.frame.lineEditElements['E_kin'].Bind(wx.EVT_TEXT, self.onEnergyChanged)
        self.frame.lineEditElements['lam'].Bind(wx.EVT_TEXT, self.onWavelengthChanged)
#        self.frame.lineEditElements['k'].Bind(wx.EVT_TEXT, self.onDispChanged)       

        self.__setDefaults__()
        self.frame.canvas.initialPlots(self.model.dlaTHZ,
                                       self.model.fields, self.model.beam)
       
            
        size = self.frame.GetSize()
        self.frame.SetMinSize((size[0]*0.9, size[1]*0.9))
        self.frame.Show()
        self.frame.Maximize(True)
        
       
    def __setDefaults__(self):
        '''
        Inserts the default values into the text fields for input
        '''
        txtCtrls = self.frame.lineEditElements
        m= self.model
        # TODO: "x emmittance", "y emmittance",
        defaults = (m.dlaTHZ.a, m.dlaTHZ.b, m.dlaTHZ.epsR, m.beam.get_Ekin0(),
                    m.beam.count, 
                    m.beam.width[0], m.beam.get_eps_x(),
                    m.beam.width[2], m.beam.get_eps_y(),
                    m.beam.width[4], m.beam.get_eps_z(),
                    m.fields.wavelength, m.fields.E_0, m.fields.phase_degree(),
                    10
                    )       
        assert(len(defaults) == len(self.short_options))
        for i, key in enumerate(self.short_options):
            if defaults[i] > 1e4 or defaults[i] < 1e-4:
                txtCtrls[key].SetValue("{0:.3e}".format(defaults[i]))
            else:
                txtCtrls[key].SetValue("{0:.3g}".format(defaults[i]))            
        ## just done to trigger EVT_TEXT event to calculate beta
#        txtCtrls['E_kin'].SetValue( txtCtrls['E_kin'].GetValue())  
#        self.txt_changed[3]= False
                
#        self.txt_changed= np.zeros(len(self.short_options),dtype=bool)
        self.txt_changed = [False]*len(self.short_options)
        
#        txtCtrls['lam'].SetValue( txtCtrls['E_kin'].GetValue())  
#        self.txt_changed[11]= False
        
        self.frame.sb.SetStatusText("Initial values set")

#        self.frame.canvas.update_dispersion_plot(m.dlaTHZ, m.fields, m.beam)
#        self.frame.canvas.update_field_plot(m.fields)
#        self.frame.canvas.update_long_phasespace_plot(m.beam)
#        self.frame.canvas.update_energy_plot(m.beam)
#        self.frame.canvas
    
    def getFormParams(self):
        '''
        the return value is a dictionary containing the values from the input
        form. The dict keys are the short_options
        '''        
        params= {}
        for sn in self.short_options:
            input_text = self.frame.lineEditElements[sn].GetValue()
            if input_text != "":
                try:                    
                    params[sn] = float(input_text)
                except ValueError:
                    print("Not a float in option {0}".format(sn))
        return params
       
    # Listener -------------------------------------
    def dlaChanged(self):
        self.frame.canvas.update_dispersion_plot_2(self.model.dlaTHZ)
#                                                 self.model.fields,
#                                                 self.model.beam)
#        pass
    
    def fieldChanged(self):
        self.frame.canvas.update_field_plot(self.model.fields)
        self.frame.canvas.update_OpPoint(self.model.fields.k, self.model.fields.omega)
        
    def beamChanged(self):
        print("MSG to update beam received")
        self.frame.canvas.update_phasespace_plot(self.model.beam)
        self.frame.canvas.update_energy_plot(self.model.beam)
        self.frame.canvas.reset_ellipseParam_plot()
        
    def beamAccelerated(self):
        # Is executed after performingthe acceleration
        self.frame.sb.SetStatusText("Simulation done")            
        self.frame.canvas.update_accelerated(self.model)
        self.frame.sb.SetStatusText("Plots done")     
        self.frame.btn_run.Enable()

        E_after, E_std_after = self.model.beam.getMeanEnergyKin(boolStd=True) 
        delta_E =  E_after - self.model.beam.get_Ekin0()                    
        N_loss=self.model.beam.removeByMaxTransverse(self.model.dlaTHZ.a)
        self.frame.txtCtrl_energy_change.SetValue( "{0:2.3e}".format(delta_E) )             
        self.frame.txtCtrl_loss.SetValue(str(N_loss))
        self.frame.txtCtrlEnergySpread_final.SetValue("{0:2.3e}".format(E_std_after))
#        self.frame.txtCtrl_Beta_z =         
       
    # Event Handler-------------------------------
    def onClick(self, event):
        '''
        update properties of dla, field and beam
        REMARK: 
        '''
        event.GetEventObject().Disable()
        params= self.getFormParams()
        m= self.model
#        print(self.txt_changed)
        try:
            if ( any(self.txt_changed[:3]) ):
    #            print("DLA changed")
                m.set_dla(params['a'], params['b'], params['epsR'])
                m.set_field(params['lam'], params['E_0'], params['phi'])
                pub.sendMessage("MSG: DLA changed")
                pub.sendMessage("MSG: Field changed")
            elif any(self.txt_changed[11:14]):
    #            print("field changed")
                m.set_field(params['lam'], params['E_0'], params['phi'])
                pub.sendMessage("MSG: Field changed")
                print("Field parameters changed")
            if any(self.txt_changed[3:11]):
    #            print("beam changed")            
                m.set_beam( int(params['N_p']), params['E_kin'],
                                params['sx'], params['eps_x'],
                                params['sy'], params['eps_y'],
                                params['sz'], params['eps_z'])
            # output freq. and phase velocity of em wave
            self.frame.txtCtrl_omega.SetValue( "{0:2.2e}".format(m.fields.omega))
            self.frame.txtCtrl_phVelocity.SetValue("{0:1.4f}".format(m.fields.phaseVelocity) )
            self.frame.txtCtrlEnergySpread.SetValue( "{0:2.3e}".format(
                self.model.beam.getMeanEnergyKin(boolStd= True)[1] )  )
        except Exception as ex:
            self.frame.showMsgBox("Error occured:\n"+ str(ex) + "\n"+ traceback.format_exc())
            
        finally:
            #reset bool list for changes
        #        self.txt_changed= np.zeros(len(self.short_options),dtype=bool)
            self.txt_changed = [False]*len(self.short_options)
            event.GetEventObject().Enable()

    
    def onTextChanged(self, event):
        # TODO: make sure the id matches the correct entry in the bool list
        txtCtrlId = event.GetEventObject().GetId()
#        event.
        self.txt_changed[txtCtrlId] = True
    
    def onClickRun(self, event):
        self.frame.btn_run.Disable()
        self.frame.sb.SetStatusText("Simulation Running...")
        numPeriods = int(self.frame.lineEditElements['num_obs'].GetValue())        
        def accelerateAndPlot(periods): 
            self.model.ellipseEvolution = self.model.beam.accelerateWithDLA(
                    self.model.fields, numSteps = 1000,
                    numPeriods=periods, 
                    ellipseParamsAfter= 15) 
            wx.CallAfter( pub.sendMessage, "MSG: Acceleration done" )            
        t = Thread(
            target= accelerateAndPlot,
            args=(numPeriods,)
            )
        t.daemon = True
        t.start()
        
    def onClickReset(self, event):
        params= self.getFormParams()
        self.frame.sb.SetStatusText("Reset Beam")
        self.model.set_beam(
            int(params['N_p']), params['E_kin'],
            params['sx'], params['eps_x'],
            params['sy'], params['eps_y'],
            params['sz'], params['eps_z'])
        
        
    def onClickAnimate(self, event):
        # TODO:        
        self.frame.sb.SetStatusText("3D Animation not supported yet")        
        pass
    
    def onEnergyChanged(self,event):
        E_input = float(event.GetEventObject().GetValue())
        beta= lambda E0, Ekin: np.sqrt(2*E0 * Ekin + Ekin**2)/(E0 + Ekin)
        self.frame.txtCtrl_beta.SetValue( "{0:1.4f}".format(
            beta(self.model.beam.mass, E_input)
            ))
        event.Skip()
    
    def onWavelengthChanged(self, event):
        lam= float(self.frame.lineEditElements['lam'].GetValue())
        self.frame.txtCtrl_wavevector.SetValue(
            "{0:.4g}".format(2*np.pi/lam))
        event.Skip()
#    def onDispChanged(self,event):
        
    

if __name__ == "__main__":
    #axes.formatter.limits : -7, 7
#    mpl.rc("axes.formatter", limits=(-3,4))
    mpl.rc("axes.formatter", limits=(-3,3))    
    mpl.rc("axes.formatter", use_mathtext=True)
    mpl.rc("legend", fontsize= 'medium')
    app = wx.App(False)
    controller = Controller()
    app.MainLoop()
    
    
'''
emittance as parameter
analysis twiss parameters 




'''

    