# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:46:51 2017

Try classes!

@author: tikan

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode
from scipy.linalg import toeplitz, eigvals,eigh
#import matplotlib.animation as animation
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
plt.rc('text', usetex=True)


class Efield:

    def __init__(self, CompField, dT):
        self.Sig = CompField
        self.TimeStep = dT
        self.AvgPower = np.mean(abs(CompField)**2)
        assert isinstance(CompField, object)
        self.Span = dT*len(CompField)
        
        
    def Spectrum (self):
        fr = np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep)
        p = np.fft.fft(self.Sig)
        p = abs(p)**2/np.max(abs(p)**2)       
        return fr,p
        
    def SaveTxt(self, FullName):
        np.savetxt(FullName, np.transpose((np.real(self.Sig), np.imag(self.Sig))))
        
    def SaveBin(self, FullName):
        np.save(FullName, zip(np.arange(0, self.Span, self.TimeStep), self.Sig))   
         
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
        fr = np.fft.fftfreq(int(len(self.Sig)),d=self.TimeStep)
        ax[1].plot(fr,abs(np.fft.fft(self.Sig))**2/np.max(abs(np.fft.fft(self.Sig))**2),'b.')
        ax[1].set_xlim([min(fr),max(fr)])
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
                maping = np.zeros((int(np.round(L/dz)), len(self.Sig)))+1j*np.zeros((int(np.round(L/dz)), len(self.Sig)))
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

    def Propagate_SAM(self, L, betta2=-1, gamma=1, Tr=0, n=50, abtol=1e-10, reltol=1e-9, param='fin_res'):
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
            
        def deriv_1(dt, field_in):
        # computes the second-order derivative of field_in
            field_fft = np.fft.fft(field_in)
            freq = 1./dt*np.fft.fftfreq(len(field_in))
            # print freq
            omega = 2.*np.pi*freq
            # field_fft*=np.exp(1j*0.5*beta2z*omega**2)
            field_fft *= 1j*omega
            out_field = np.fft.ifft(field_fft)
            return out_field
        if Tr==0:
            def NLS_1d(Z, A):
                # time second order derivative
                dAdT2 = deriv_2(self.TimeStep, A)
#                dAAdT = deriv_1(self.TimeStep,abs(A)**2)
                dAdz = -1j*betta2/2*dAdT2+1j*gamma*abs(A)**2*A#-1j*gamma*Tr*dAAdT
                return dAdz        
        else:
            def NLS_1d(Z, A):
                # time second order derivative
                dAdT2 = deriv_2(self.TimeStep, A)
                dAAdT = deriv_1(self.TimeStep,abs(A)**2)
                dAdz = -1j*betta2/2*dAdT2+1j*gamma*abs(A)**2*A-1j*gamma*Tr*dAAdT*A
                return dAdz

        dz =float(L)/n
#        r = complex_ode(NLS_1d).set_integrator('zvode', method='bdf', with_jacobian=False, atol=abtol, rtol=reltol)
        r = complex_ode(NLS_1d).set_integrator('dopri5', atol=abtol, rtol=reltol)
#        r = complex_ode(NLS_1d).set_integrator('lsoda', method='BDF', atol=abtol, rtol=reltol, with_jacobian=False)
        r.set_initial_value(self.Sig, 0)
        sol=np.ndarray(shape=(n+1, len(self.Sig)), dtype=complex)
        sol[0] = self.Sig
        for it in range(1, n+1):
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
        if Offset == 0:
            A = np.abs(Rf)**2
            A = Pavg*A/np.mean(A)
            Ph = np.angle(Rf)
            self.Sig = np.sqrt(A)*np.exp(1j*Ph)
        else:
            A = Rf*Pavg**.5+Offset
            self.Sig = A
        
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


class RandomWaveFromSpec(Efield):
    """
    The ranom wave has the power! based on the knowk spectra
    The output self.Sig is a complex field
    """
    def __init__(self, FreqArray, AbsSpecSqr, Pavg, Offset=0, NofP=1e4):
        
        def Com(x,y): return np.sum(x*y)/np.sum(y)
            
        self.AvgPower = Pavg
        NewFreq = np.linspace(min(FreqArray), max(FreqArray), NofP)
        AbsSpecSqrInterp = np.interp(NewFreq, FreqArray, AbsSpecSqr)
        
        indMax = int(np.where(NewFreq>Com(NewFreq,AbsSpecSqrInterp))[0][0])
        if indMax!= np.round(NofP/2):
            diff = int(indMax - np.round(NofP/2))
        else:
            diff = 0
        if diff>0:
            AbsSpecSqrInterp = np.concatenate((np.delete(AbsSpecSqrInterp, np.s_[:diff]), np.zeros(diff)), axis=0)
        else: 
            diff = abs(diff)
            AbsSpecSqrInterp = np.concatenate((np.zeros(diff), np.delete(AbsSpecSqrInterp, np.s_[-diff:])), axis=0)
        NewFreq = NewFreq - NewFreq[indMax]
        
        Rf = np.fft.ifft(np.fft.fftshift(np.sqrt(AbsSpecSqrInterp)*np.exp(1j*np.random.uniform(-1,1,int(NofP))*np.pi)))##fft or ifft shift
        A = np.abs(Rf)**2
        A = Pavg*A/np.mean(A) + Offset
        Ph = np.angle(Rf)
        self.Sig = np.sqrt(A)*np.exp(1j*Ph)
        
        dT = 1./(NofP*(NewFreq[2]-NewFreq[1]))
        self.Span = NofP*dT
        self.TimeStep = dT

        
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



def ISTcompute_f(field, dT):
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

def ISTcompute_d(field, dT):
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
    M = np.concatenate((np.concatenate((1j*B1, B2), axis=1), np.concatenate((B2.conj().T, -1j*B1), axis=1)), axis=0)
    return eigh(M)

def periodize(dat, period, delay=0):
    loc = dat
    for ii in np.arange(period):
        dat = np.append(dat,loc)
    return dat

def IST(field, dT, periodized=0,param='foc'):
    def periodize(dat, period, delay=0):
        loc = dat
        for ii in np.arange(period):
            dat = np.append(dat,loc)
        return dat        
    if param=='foc':
        return ISTcompute_f(periodize(field, periodized), dT)
    elif param=='def':
        return ISTcompute_d(periodize(field, periodized), dT)


def plotEV(MM,xl=-5,xr=5, yb=-1, yt=1, alph = 0.2):
    col = ['r','g','b','y']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if np.size(MM) ==  np.size(MM,0): 
        plt.plot(np.real(MM), np.imag(MM), ls='none', alpha=alph, marker='o',color=col[0])
    else:
        for kk in range(np.size(MM,1)):
            plt.plot(np.real(MM[:,kk]), np.imag(MM[:,kk]), ls='none', alpha=alph, marker='o',color=col[kk])    
    plt.xlim(xl,xr)
    plt.ylim(yb,yt)
            
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
    
    
def Plot_Map(map_data,dt,dz,colormap = 'cubehelix'):
    def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero
    
        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to 
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }
    
        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)
    
        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False), 
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])
    
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)
    
            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))
    
        newcmap = mcolors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)
    
        return newcmap


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
        f.canvas.draw()
        
    f = plt.figure()
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    plt.suptitle('Choose the coordinate', fontsize=20)
    f.set_size_inches(10,8)
    Z,T = np.meshgrid( np.arange(0,dz*np.size(map_data,0),dz), np.arange(0, dt*np.size(map_data,1),dt))
#    orig_cmap = plt.get_cmap('viridis')
#    colormap = shiftedColorMap(orig_cmap, start=0., midpoint=.5, stop=1., name='shrunk')
    pc = ax.pcolormesh(Z, T, abs(np.transpose(map_data))**2, cmap=colormap)
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

def DarkSoliton(t,a=1,B=1,t0=0):
    xi=a*B*(t-t0-a*B*np.sqrt(1-B**2))
    return a*(B*np.tanh(xi)-1j*np.sqrt(1-B**2))

def N_DarkSoliton(t,N):return N*np.tanh(t)

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
Think about how to save class and reload it again
"""

