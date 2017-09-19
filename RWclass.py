# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:46:51 2017

Try classes!

@author: manip

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode
from scipy.linalg import toeplitz, eigvals
#import matplotlib.animation as animation
import matplotlib.ticker as ticker

plt.rc('text', usetex=True)


class Efield:

    def __init__(self, CompField, dT):
        self.Sig = CompField
        self.TimeStep = dT
        self.AvgPower = np.mean(abs(CompField)**2)
        assert isinstance(CompField, object)
        self.Span = dT*len(CompField)

    def SaveTxt(self, FullName):
        np.savetxt(FullName, np.transpose((np.real(self.Sig), np.imag(self.Sig))))
        
    def PlotSig(self):
        def cm(x, y): return np.sum(x*y)/np.sum(y)  # Center Of Mass

        def StDiv(x, y): return np.sqrt(np.sum(y*(x-cm(x, y))**2)/np.sum(y))  # Standard Div
        Time = np.arange(0, self.Span, self.TimeStep)
        f, ax = plt.subplots(2)
        f.suptitle(r' \textless P\textgreater = '+str(self.AvgPower)+'W', fontsize=10)
        f.set_size_inches(8,6)
        ax[0].plot(Time, abs(self.Sig)**2, 'r')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Power (W)')
        ax[1].plot(np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep),abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2),'b.')
        ax[1].set_xlim([-1*StDiv(Time,self.Sig)/2,1*StDiv(Time,self.Sig)/2])
        ax[1].set_xlabel('Freq (THz)',color='g')
        ax[1].set_ylabel('Power Spectr. Density', color='r')
                
    def Propagate_SSFM(self, L, betta2=-1,gamma=1, dz=1e-3, param='fin_res', movie=False, plotmap=False):
        """Propagate Using the split step fourier method"""
        uf = self.Sig
        freq = np.fft.fftfreq(len(self.Sig), d=self.TimeStep)
        for ii in np.arange(np.round(L/dz)):
            ii = int(ii)
            # map array
            if param == 'map' and ii == 0:
                maping = np.zeros((int(np.round(L/dz)), len(self.Sig)))
                maping[0] = self.Sig
            elif param == 'map' and ii > 0:
                maping[ii] = uf
            # half step of dispersion
            fftu = np.fft.fft(uf)*np.exp(1j*np.power(np.pi*freq, 2)*betta2*dz)
            # step of nonlinearity
            uf = np.fft.ifft(fftu)*np.exp(1j*np.power(abs(uf), 2)*gamma*dz)
            # half step of disp
            fftu = np.fft.fft(uf)*np.exp(1j*np.power(np.pi*freq, 2)*betta2*dz)
            # go back
            uf = np.fft.ifft(fftu)
        if param == 'map':
            return maping
        elif param == 'fin_res':
            return uf
        else:
            print 'wrong parameter'

    def Propagate_SAM(self, L, betta2=-1, gamma=1, dz=1e-3, tol=1e-15, param='fin_res'):
        """Propagate Using the Step Adaptative  Method"""
        def deriv_2(dt, field_in):
        # computes the second-order derivative of field_in
            field_fft = np.fft.fft(field_in)
            freq = 1./dt*np.fft.fftfreq(len(field_in))
            # print freq
            omega = 2.*np.pi*freq
            # field_fft*=np.exp(1j*0.5*beta2z*omega**2)
            field_fft *= -omega**2
            out_field = np.fft.ifft(field_fft)
            return out_field 
 
        def NLS_1d(Z, A):
            # time second order derivative
            dAdT2 = deriv_2(self.TimeStep, A)
            dAdz = -1j*betta2/2*dAdT2+1j*gamma*abs(A)**2*A
            return dAdz
        
        r = complex_ode(NLS_1d).set_integrator('vode', method='BDF', atol=tol, with_jacobian=False)
        r.set_initial_value(self.Sig, 0)
        sol=np.ndarray(shape=(int(np.round(L/dz)+1), len(self.Sig)), dtype=complex)
        for it in range(0, int(np.round(L/dz))):
            sol[it] = r.integrate(r.t+dz)
        if param == 'map':
            return sol
        elif param == 'fin_res':
            return sol[-2, :]
        else:
            print 'wrong parameter'


class RandomWave(Efield):
    """
        The ranom wave has the power! spectral width dNu
        The output self.Sig is a complex field
    """
    def __init__(self, dNu, Pavg, Offset=0, NofP=1e4, dT=0.05):
        self.SpWidth = dNu
        self.AvgPower = Pavg
        self.Span = NofP*dT
        self.TimeStep = dT
        Rf = np.fft.ifft(np.exp(-1*(np.fft.fftfreq(int(NofP),d=dT))**2/4/((dNu/2)**2/2/np.log(2)))*np.exp(1j*np.random.uniform(-1,1,int(NofP))*np.pi))
        # Above 4 is for the sqrt of intensity        
        A = np.abs(Rf)**2
        A = Pavg*A/np.mean(A) + Offset
        Ph = np.angle(Rf)
        self.Sig = np.sqrt(A)*np.exp(1j*Ph)
        
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
        f, ax = plt.subplots(2)
        f.suptitle(r' \textless P\textgreater = '+str(self.AvgPower)+' W; \Delta\\nu = '+str(self.SpWidth)+'THz', fontsize=10)
        f.set_size_inches(8, 6)
        ax[0].plot(np.arange(0, self.Span, self.TimeStep), abs(self.Sig)**2, 'r')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Power (W)')
        ax[1].plot(np.fft.fftfreq(int(len(self.Sig)), d=self.TimeStep), abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2), 'b.')
        ax[1].set_xlim([-2*self.SpWidth, 2*self.SpWidth])
        ax[1].set_xlabel('Freq (THz)', color='g')
        ax[1].set_ylabel('Power Spectr. Density', color='r')
        plt.show()
        
        
class Stand_Func(Efield):
    """
        Stand_Func acsept the Executable func or 'sech' to test the 
        The output self.Sig is Amplitude
    """
    def __init__(self, ExecutFunc, NofP=1e4, dT=0.01):
        self.Span = NofP*dT
        self.TimeStep = dT
        
        _T = np.arange(-1*np.round(self.Span/2),np.round(self.Span/2),self.TimeStep)
        if ExecutFunc == 'Sech' or ExecutFunc == 'sech':
            self.Sig = 1/np.cosh(_T)
        else:
            self.Sig = ExecutFunc(_T)
        self.AvgPower = np.mean(abs(self.Sig)**2)
        
    def PlotSig(self):
        def cm(x, y): return np.sum(x*y)/np.sum(y)  # Center Of Mass

        def StDiv(x, y): return np.sqrt(np.sum(y*(x-cm(x,y))**2)/np.sum(y))  # Standard Div
        Time = np.arange(0, self.Span, self.TimeStep)
        f, ax = plt.subplots(2)
        f.suptitle(r' Peak Power = '+str(np.max(abs(self.Sig)**2))+' W', fontsize=10)
        f.set_size_inches(8, 6)
        ax[0].plot(Time, abs(self.Sig)**2, 'r')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Power (W)')
        ax[1].plot(np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep),abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2),'b.')
        ax[1].set_xlabel('Freq (THz)',color='g')
        ax[1].set_xlim([-1*StDiv(Time,self.Sig)/2,1*StDiv(Time,self.Sig)/2])
        ax[1].set_ylabel('Power Spectr. Density', color='r')
        plt.show()


def ISTcompute(field, dT):
    # Fourier collocation method for the Z-S eigenvalue problem
    Nx = len(field)
    if Nx%2:
        field = np.append(field,field[-1])
    Nx = len(field)
    N = Nx/2
    L = dT*Nx
    k0 = 2*np.pi/L
    x = np.arange(-L/2, L/2, dT)
    C = []    
    for n in np.arange(-N, N+1):
        C.append(dT*np.sum(field*np.exp(-1j*k0*n*x))/L)
    B1 = 1j*k0*np.diag(np.arange(-N, N+1))
    B2 = toeplitz(np.append(C[N:], np.zeros(N)), np.append(np.flip(C[:N+1], 0), np.zeros(N)))
    M = np.concatenate((np.concatenate((-B1, B2), axis=1), np.concatenate((B2.conj().T, B1), axis=1)), axis=0)
    return eigvals(-1j*M)

def periodize(dat, period, delay=0):
    loc = dat
    for ii in np.arange(period):
        dat = np.append(dat,loc)
    return dat

def IST(field, dT, periodized=0):     

    def periodize(dat, period, delay=0):
        loc = dat
        for ii in np.arange(period):
            dat = np.append(dat,loc)
        return dat        

    return ISTcompute(periodize(field, periodized), dT)
    
def IST_graf(field, dT, periodized=0):
    coords = []
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d' % (ix, iy)
        coords.append((ix, iy))
        if np.size(coords, 0) == 2:
            f.canvas.mpl_disconnect(cid)
            if coords[0][0] < coords[1][0]:
                st = int(np.floor(coords[0][0]))
                sp = int(np.floor(coords[1][0]))
            elif coords[0][0] > coords[1][0]:
                sp = int(np.floor(coords[0][0]))
                st = int(np.floor(coords[1][0]))
            ax.plot(np.arange(st, sp), abs(field[st:sp])**2, 'r')
            
            ev = ISTcompute(periodize(field[st:sp], periodized), dT)
            axIST = plt.subplot2grid((2, 2), (0, 1), rowspan=2)            
            axIST.plot(ev.real, ev.imag, 'r.')
            
            axPer = plt.subplot2grid((2, 2), (1, 0))
            axPer.plot(abs(periodize(field[st:sp], periodized))**2,'g')
            plt.show()
    f = plt.figure()
    ax = plt.subplot2grid((2, 2), (0, 0))
    plt.suptitle('Choose the working area', fontsize=20)
    f.set_size_inches(10,6)
    ax.plot(abs(field)**2,'b')
    ax.set_xlabel('Number of point')
    ax.set_ylabel('Power (W)')
    plt.ylim(0, max(abs(field)**2)+0.1)
    plt.xlim(0, len(field))
    cid = f.canvas.mpl_connect('button_press_event', onclick)
    
    
def Plot_Map(map_data,dt,dz):

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        x = int(np.floor(ix/dz))
        plt.suptitle('Chosen distance z = %f km'%ix, fontsize=20)
        ax.lines.pop(0)
        ax.plot([ix,ix], [0, dt*np.size(map_data,1)],'r')

        ax2 = plt.subplot2grid((4, 1), (2, 0))            
        ax2.plot(np.arange(0,dt*np.size(map_data,1),dt), abs(map_data[x,:])**2, 'r')
        ax2.set_ylabel('Power (W)')
        ax2.set_xlim(0, dt*np.size(map_data,1))        
        ax3 = plt.subplot2grid((4, 1), (3, 0))
        ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.angle(map_data[x,:])/(np.pi),'b')
        if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
            ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_xlim(0, dt*np.size(map_data,1))
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
        ax3.grid(True)
        
        plt.show()
        
    f = plt.figure()
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    plt.suptitle('Choose the coordinate', fontsize=20)
    f.set_size_inches(10,8)
    Z,T = np.meshgrid( np.arange(0,dz*np.size(map_data,0),dz), np.arange(0, dt*np.size(map_data,1),dt))
    pc = ax.pcolor(Z, T, abs(np.transpose(map_data))**2, cmap=plt.get_cmap('viridis'))
    ax.plot([0, 0], [0, dt*np.size(map_data,1)-dt], 'r')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Time (ps)')
    ax.set_ylim(0, dt*np.size(map_data,1))
    ax.set_xlim(0, dz*np.size(map_data,0)-5*dz)
#    f.colorbar(pc)
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.93, wspace=None, hspace=0.4)
    f.canvas.mpl_connect('button_press_event', onclick)
"""
here is a set of useful standard functions
"""
def SuperGauss(x, p=3, x0=0, sigma=3, a=1) : return a*np.exp(-1*np.power((x-x0)**2/(2*sigma**2),p))
    
def Sech(x,a=1.,x0=0,t=1.) : return a/np.cosh(x/t-x0)

def SolitonNLS(x,Pm=1., x0=0, gamma=1., betta2=-1) : return Pm/np.cosh(x*np.sqrt(gamma*Pm/abs(betta2))-x0)

def KMsr (t, x, phi):
    Om = 2*np.sinh(2*phi)
    q = 2*np.sinh(phi)
    ii = 0
    dat_array = np.zeros([np.size(t),np.size(x)])
    if np.size(x)>1 and np.size(t)>1:
        for xx in x:
            dat_array[:,ii] = (np.cos(Om*t-2*1j*phi)-np.cosh(phi)*np.cosh(q*xx))/(np.cos(Om*t)-np.cosh(phi)*np.cosh(q*xx))*np.exp(2*1j*t) 
            ii += 1
        return dat_array
    else:
        return (np.cos(Om*t-2*1j*phi)-np.cosh(phi)*np.cosh(q*x))/(np.cos(Om*t)-np.cosh(phi)*np.cosh(q*x))*np.exp(2*1j*t) 
        
def ABsr (t, x, phi):
    Om = 2*np.sin(2*phi)
    q = 2*np.sin(phi)
    ii = 0
    dat_array = np.zeros([np.size(t),np.size(x)])
    if np.size(x)>1 and np.size(t)>1:
        for xx in x:
            dat_array[:,ii] = (np.cosh(Om*t-2*1j*phi)-np.cos(phi)*np.cos(q*xx))/(np.cosh(Om*t)-np.cos(phi)*np.cos(q*xx))*np.exp(2*1j*t) 
            ii += 1
        return dat_array
    else:
        return (np.cosh(Om*t-2*1j*phi)-np.cos(phi)*np.cos(q*x))/(np.cosh(Om*t)-np.cos(phi)*np.cos(q*x))*np.exp(2*1j*t) 
"""
also good to do things for the wave shaper! line the method wgich could show the dynamics and give the file for ws!
"""

#%%
#a = RandomWave(0.05, 4.5, NofP=2048, dT=0.08, Offset=0)
##ff = IST_graf(a.Propagate_SSFM(0.5), a.TimeStep, periodized=2)
##plt.figure()
##plt.plot(ff.real,ff.imag,'r.')
##%%
##T = np.arange(-10,10,0.05)
##dsw = Efield(5-5/np.cosh(T), dT=0.05)
##plt.figure()
##plt.pcolor(abs(dsw.Propagate_SSFM(.5,1,5,param='map', Movie=0))**2,cmap=plt.get_cmap('coolwarm'))
##plt.colorbar()
##plt.plot(dsw.Propagate_SSFM(0.1,20,1.3,))
##dsw.PlotSig()
##%%
#
#plt.figure()
#plt.plot(TT,)
#TT = np.arange(-50,50, 0.1)
#dat =  SolitonNLS(TT, betta2=-20,gamma=2,Pm=1.)
#a = Efield(dat, dT=.1)
#a.PlotSig()
#%% test map
#dZ = 0.01
#M = a.Propagate_SAM(0.9, betta2=20., gamma=3., dz=dZ, param='map')
#Plot_Map(M, a.TimeStep, dZ)
##%%
#IST_graf(M[int(np.floor(0./dZ)),:], a.TimeStep, periodized=0)
#%% breather try
#xx = np.arange(-300,300,0.2)
#yy = np.arange(-100,100,0.4)
#a = 0.4999
#M = KMsoliton(yy,yy,a)
#plt.figure()
##plt.plot(abs(M[2500:2730])**2)
#plt.pcolor(M)
##%%
#ff = IST(M[2500:2730], 0.2, periodized=10)
#plt.figure()
#plt.plot(np.real(ff), np.imag(ff),'r.')
#%% propagation tests 
#TT = np.arange(-50,50, 0.1)
#dZ = 0.01
#dat =  SolitonNLS(TT, betta2=-20,gamma=2,Pm=1.)
#a = Efield(dat, dT=.1)
##a.PlotSig() 
#ee = a.Propagate_SAM(0.5, betta2=-20., gamma=30., dz=dZ, param='map')
#Plot_Map(ee, a.TimeStep, dZ)
#b = Efield(ee[int(np.floor(0.117/dZ)),:], dT=.1)
##%%
#b.PlotSig()
#ee = b.Propagate_SAM(-0.5, betta2=-20., gamma=30., dz=-dZ, param='map')
#Plot_Map(ee, b.TimeStep, dZ)
#%% ist check km 
#
#tt = np.arange(-10,10,1./7)
#
#M = ABsr(0., tt, np.pi/4)
##Plot_Map(M, 0.05, 0.05)
#km = Efield(M,tt[1]-tt[0])
##km.PlotSig()
##%%
#kmp = IST_graf(M, tt[1]-tt[0], periodized=20)
#
#plt.figure()
#plt.plot(np.real(kmp), np.imag(kmp),'r.')
#%% one dam
#TT = np.arange(-50,50, 0.1)

#    
#SG = SuperGauss(TT,10, 0, 10, 1)*np.exp(-1j*np.random.uniform(-1,1,int(len(TT)))*np.pi)#+SuperGauss(TT, 1, -4, 2, 5)
#sg = Efield(SG,0.1)
#sg.PlotSig()
#dZ=0.002
#M = sg.Propagate_SAM(0.3, betta2=-20., gamma=2.4, dz=dZ, param='map')
#Plot_Map(M, sg.TimeStep, dZ)
#%% 












    
    
