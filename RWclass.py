# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:46:51 2017

Try classes!

@author: manip

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode

plt.rc('text', usetex=True)
#plt.rc('font', family='arial')


class Efield:
    def __init__(self,CompField,dT):
        self.Sig = CompField
        self.TimeStep = dT
        self.AvgPower = np.mean(abs(CompField)**2)
        self.Span = dT*len(CompField)
        
        
    def PlotSig(self):
        def cm(x,y): return np.sum(x*y)/np.sum(y) # Center Of Mass
        def StDiv(x,y): return np.sqrt(np.sum(y*(x-cm(x,y))**2)/np.sum(y)) # Standard Div
        Time = np.arange(0,self.Span,self.TimeStep)
        f,ax=plt.subplots(2)
        f.suptitle(r' \textless P\textgreater = '+str(self.AvgPower)+'W', fontsize=10)
        f.set_size_inches(8,6)
        ax[0].plot(Time,abs(self.Sig)**2,'r')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Power (W)')
        ax[1].plot(np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep),abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2),'b.')
        ax[1].set_xlim([-1*StDiv(Time,self.Sig)/2,1*StDiv(Time,self.Sig)/2])
        ax[1].set_xlabel('Freq (THz)',color='g')
        ax[1].set_ylabel('Power Spectr. Density', color='r')

                
    def Propagate_SSFM(self, L, betta2=-1,gamma=1, dz=1e-3, param='fin_res'):
        'Propagate Using the split step fourier method'
        uf=self.Sig
        freq = np.fft.fftfreq(len(self.Sig), d=self.TimeStep)
        for ii in np.arange(np.round(L/dz)):
            ii = int(ii)
            ## map array
            if param == 'map' and ii == 0:
                maping = np.zeros((int(np.round(L/dz+1)),len(self.Sig)))
                maping[0] = self.Sig
            elif param == 'map' and ii > 0:
                maping[ii] = uf
            #half step of dispersion
            fftu = np.fft.fft(uf)*np.exp(1j*np.power(np.pi*freq,2)*betta2*dz)
            #step of nonlinearity
            uf = np.fft.ifft(fftu)*np.exp(1j*np.power(abs(uf),2)*gamma*dz) 
            #half step of disp
            fftu = np.fft.fft(uf)*np.exp(1j*np.power(np.pi*freq,2)*betta2*dz)
            #go back
            uf = np.fft.ifft(fftu)
        if param == 'map':
            return maping
        elif param == 'fin_res':
            return uf
        else:
            print 'wrong parameter'
            
            
    def Propagate_SAM(self, L, betta2=-1,gamma=1, dz=1e-3, param='fin_res'):
        'Propagate Using the Step Adaptative  Method'
        def deriv_2(dt,field_in):
        # computes the second-order derivative of field_in
            field_fft=np.fft.fft(field_in)
            freq=1./dt*np.fft.fftfreq(len(field_in))
            #print freq
            omega=2.*np.pi*freq
            #    field_fft*=np.exp(1j*0.5*beta2z*omega**2)
            field_fft*=-omega**2
            out_field=np.fft.ifft(field_fft)
            return out_field 
 
        def NLS_1d(Z,A):
            # time second order derivative
            dAdT2=deriv_2(self.TimeStep,A)
            dAdz = -1j*betta2/2*dAdT2+1j*gamma*abs(A)**2*A
            return dAdz
        
        r = complex_ode(NLS_1d).set_integrator('vode', method='BDF', atol=1e-15, with_jacobian=False)
        r.set_initial_value(self.Sig,0)
        sol=np.ndarray(shape=(int(np.round(L/dz)+1),len(self.Sig)), dtype=complex)
        for it in range(0,int(np.round(L/dz))):
            sol[it]=r.integrate(r.t+dz)
        if param == 'map':
            return sol
        elif param == 'fin_res':
            return sol[-2,:]
        else:
            print 'wrong parameter'

class RandomWave(Efield):
    """
        The ranom wave has the power! spectral width dNu
        The output self.Sig is a complex field
        Pet.__init__(self, name, "Dog")
    """
    def __init__(self,dNu,Pavg,NofP=1e4,dT=0.05):
        self.SpWidth = dNu
        self.AvgPower = Pavg
        self.Span = NofP*dT
        self.TimeStep = dT
        Rf = np.fft.ifft(np.exp(-1*(np.fft.fftfreq(int(NofP),d=dT))**2/4/((dNu/2)**2/2/np.log(2)))*np.exp(1j*np.random.uniform(-1,1,int(NofP))*np.pi))
        # Above 4 is for the sqrt of int        
        A = np.abs(Rf)
        A = np.sqrt(Pavg)*A/np.mean(A)        
        Ph = np.angle(Rf)
        self.Sig = A*np.exp(1j*Ph)
        
#    def __call__(self,dNuNEW,PavgNEW,NofP=1e4,dT=0.05):
#        self.SpWidth = dNuNEW
#        self.AvgPower = PavgNEW
#        self.Span = dNuNEW*dT
#        self.TimeStep = dT
#        Rf = np.fft.ifft(np.exp(-1*(np.fft.fftfreq(int(NofP),d=dT))**2/4/((dNuNEW/2)**2/2/np.log(2)))*np.exp(1j*np.random.uniform(-1,1,int(NofP))*np.pi))
#        # Above 4 is for the sqrt of int        
#        A = np.abs(Rf)
#        A = np.sqrt(PavgNEW)*A/np.mean(A)        
#        Ph = np.angle(Rf)
#        self.Sig = A*np.exp(1j*Ph)
        
    def PlotSig(self):
        f,ax=plt.subplots(2)
        f.suptitle(r' \textless P\textgreater = '+str(self.AvgPower)+' W; \Delta\\nu = '+str(self.SpWidth)+'THz', fontsize=10)
        f.set_size_inches(8,6)
        ax[0].plot(np.arange(0,self.Span,self.TimeStep),abs(self.Sig)**2,'r')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Power (W)')
        ax[1].plot(np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep),abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2),'b.')
        ax[1].set_xlim([-2*self.SpWidth,2*self.SpWidth])
        ax[1].set_xlabel('Freq (THz)',color='g')
        ax[1].set_ylabel('Power Spectr. Density', color='r')
        plt.show()
        
        
class Stand_Func(Efield):
    """
        Stand_Func acsept the Executable func or 'sech' to test the 
        The output self.Sig is Amplitude
    """
    def __init__(self,ExecutFunc,NofP=1e4,dT=0.01):        
        self.Span = NofP*dT
        self.TimeStep = dT
        
        _T = np.arange(-1*np.round(self.Span/2),np.round(self.Span/2),self.TimeStep)
        if  ExecutFunc=='Sech' or ExecutFunc=='sech':
            self.Sig = 1/np.cosh(_T)
        else:
            self.Sig = ExecutFunc(_T)
        self.AvgPower = np.mean(abs(self.Sig)**2)
        
    def PlotSig(self):
        def cm(x,y): return np.sum(x*y)/np.sum(y) # Center Of Mass
        def StDiv(x,y): return np.sqrt(np.sum(y*(x-cm(x,y))**2)/np.sum(y)) # Standard Div
        Time = np.arange(0,self.Span,self.TimeStep)
        f,ax=plt.subplots(2)
        f.suptitle(r' Peak Power = '+str(np.max(abs(self.Sig)**2))+' W' , fontsize=10)
        f.set_size_inches(8,6)
        ax[0].plot(Time,abs(self.Sig)**2,'r')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Power (W)')
        ax[1].plot(np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep),abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2),'b.')
        ax[1].set_xlabel('Freq (THz)',color='g')
        ax[1].set_xlim([-1*StDiv(Time,self.Sig)/2,1*StDiv(Time,self.Sig)/2])
        ax[1].set_ylabel('Power Spectr. Density', color='r')
        plt.show()



def IST(field, dT, Peroidized=250):
    #Time = np.arange(0,len(field),dT)
    coords=[]
    def onclick(event):
        global coords
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(ix, iy)
        #
        if len(coords) == 2:
            f.canvas.mpl_disconnect(cid)
        coords.append((ix, iy))
        return
    
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.suptitle('Choose the working area', fontsize=20)
    f.set_size_inches(8,6)
    ax.plot(abs(field)**2,'b')
    ax.set_xlabel('Number of point')
    ax.set_ylabel('Power (W)')
    plt.ylim(0, max(abs(field)**2)+1)
    plt.xlim(0, len(field))
    cid = f.canvas.mpl_connect('button_press_event', onclick)
    
    ax.plot(abs(field[coords[0][0]:coords[0][1]])**2,'r')
    return 1



    
"""
also good to do things for the wave shaper! line the method wgich could show the dynamics and give the file for ws!
"""
#%%
a = RandomWave(0.1,1.)

IST(a.Sig,a.TimeStep)

#Dist = 0.4
#plt.suptitle('Prop dist:'+str(Dist)+'km')
#plt.plot(np.arange(0,a.Span,a.TmeStep),abs(a.Sig)**2)
#plt.plot(np.arange(0,a.Span,a.TmeStep),abs(a.Propagate_SAM(Dist,betta2=-20,gamma=2.4,param='map')[-2,:])**2)
#plt.plot(np.arange(0,a.Span,a.TmeStep),abs(a.Propagate_SSFM(Dist,betta2=-20,gamma=2.4))**2)
#%%
#def gaussian(x, mu=0, sig=5):
#    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#
#a = Stand_Func(gaussian)
#a.PlotSig()
#plt.figure()
#plt.plot(np.arange(0,a.Span,a.TmeStep),abs(a.Propagate_SSFM(0.2))**2,np.arange(0,a.Span,a.TmeStep),abs(a.Sig)**2)













    
    
    
    
