import numpy as np
import math
from scipy.integrate import odeint
from matplotlib import animation, rc
rc('animation', html='html5')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Magnetic field, electric field
#B = np.array((0,0,1))
E = np.array((0,0,0))
dipole_moment = np.array((0,0,7.788*(10**22)))
geometry = np.array((0,0,0,1e3)) # three first are position, four is radius of the diamagnetic zone
susceptibility = -1 # perfect diamagnetic, i.e. superconducting

Frames=7000

Re=6371000 #Earth radius in meter


def dipole_field(X, dipole_moment):
    length = np.linalg.norm(X[:3])
    if length == 0.0:
        return np.zeros(3)
    else:
        return (3.0*X[:3]*(dipole_moment.dot(X[:3]))/(length**5)-dipole_moment/(length**3))

def B_calc(X,dipole_moment,geometry,susceptibility):
    if np.linalg.norm(geometry[:3]-X[:3])<geometry[3]:    
        return dipole_field(X, dipole_moment) 
    else:
        return dipole_field(X, dipole_moment) 
        

def lorentz(X, t, q_over_m):
        """
        The equations of motion for the Lorentz force on a particle with
        q/m given by q_over_m. X=[x,y,z,vx,vy,vz] defines the particle's
        position and velocity at time t: F = ma = (q/m)[E + v×B].
        
        """
        B=dipole_field(X, dipole_moment)       
        v = X[3:]
        drdt = v
        dvdt = q_over_m * (E + np.cross(v, B))
        return np.hstack((drdt, dvdt))


def calc_trajectory(q, m, r0=None, energy=None):
    """Calculate the particle's trajectory.
    
    q, m are the particle charge and mass;
    r0 and v0 are its initial position and velocity vectors.
    If r0 is not specified, it is calculated to be the Larmor
    radius of the particle, and particles with different q, m
    are placed so as have a common guiding centre (for E=0).
    
    """
    if energy is None:
        "energy in MeV"    
        energy=10

    theta=np.pi/180*30
    c=299792458
    E_rest_e=0.511
    m_e=9.10938356*(10**(-31))
    gamma=energy/(m*E_rest_e)+1
    beta=math.sqrt(1-(1/gamma)**2)
    v= c*beta
    v0=np.array((v*math.sin(theta),0,v*math.cos(theta)))
    print('Energy is %0.3f MeV. gamma is %0.3f. beta is %0.3f and v=%0.3f km/s'%(energy, gamma ,beta,v/1000))


    
    if r0 is None:
        #rho = larmor(q, m, v0)
        #vp = np.array((v0[1],-v0[0],0))
        r0 = np.array((2*Re,0,0)) 
        #r0=-np.sign(q) * vp * rho
    # Final time, number of time steps, time grid.
    tf = 7
    N = Frames
    t = np.linspace(0, tf, N)
    # Initial positon and velocity components.
    X0 = np.hstack((r0, v0))
    # Do the numerical integration of the equation of motion.
    X = odeint(lorentz, X0, t, args=(q/m/gamma*1.60217662/9.10938356*(10**(-19+31-7)),))
    return X

def setup_axes(ax):
    """Style the 3D axes the way we want them."""
    
    # Gotta love Matplotlib: this is how to indicate a right-handed
    # coordinate system with the x, y, and z axes meeting at a point.
    #ax.yaxis._axinfo['juggled'] = (1,1,2)
    #ax.zaxis._axinfo['juggled'] = (1,2,0)
    # Remove axes ticks and labels, the grey panels and the gridlines.
    #for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis:
        #for e in axis.get_ticklines() + axis.get_ticklabels():
        #    e.set_visible(False) 
        #axis.pane.set_visible(False)
        #axis.gridlines.set_visible(False)
    # Label the x and z axes only.
    ax.set_xlabel('x[$R_e$]', labelpad=10, size=12)
    ax.set_ylabel('y[$R_e$]', labelpad=10, size=12)
    ax.set_zlabel('z[$R_e$]', labelpad=10, size=12)
    

    ax.set_xlim(-5*Re, 5*Re)
    ax.set_xticks(np.arange(-4*Re, (4+1)*Re, 2*Re))
    ax.set_xticklabels(["-4", "-2", "0", "2", "4"])

    ax.set_ylim(-5*Re, 5*Re)
    ax.set_yticks(np.arange(-4*Re, (4+1)*Re, 2*Re))
    ax.set_yticklabels(["-4", "-2", "0", "2", "4"])

    ax.set_zlim(-2*Re, 2*Re)
    ax.set_zticks(np.arange(-2*Re, (2+1)*Re, 1*Re))
    ax.set_zticklabels(["-2", "-1", "0", "1", "2"])


def plot_trajectories(trajectories):
    """Produce a static plot of the trajectories.
    
    trajectories should be a sequence of (n,3) arrays representing
    the n timepoints over which the (x,y,z) coordinates of the
    particle are given.
    
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    for X in trajectories:
        ax.plot(*X.T[:3])
    setup_axes(ax)
    # Plot a vertical line through the origin parallel to the
    # magnetic field.
    #zmax = np.max(np.max([X.T[2] for X in trajectories]))
    #ax.plot([0,0],[0,0],[0,zmax*Re],lw=2,c='gray')

    

    plt.show()

# charges and masses of the electron (e) and ion (i)
qe, me = 1, 1836
qi, mi = 1, 1836
# Calculate the trajectories for the particles (which don't interact) by
# numerical integration of their equations of motion.
Xe = calc_trajectory(qe, me, np.array((4*Re,0,0)))
Xi = calc_trajectory(qi, mi, np.array((2*Re,0,0)))

def init():
    """Initialize the trajectory animation."""
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    # The axis of the magentic field
    #ax.plot([0,0],[0,0],[0,500],lw=2,c='gray')
    # Plot a Sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x_s = Re * np.outer(np.cos(u), np.sin(v))
    y_s = Re * np.outer(np.sin(u), np.sin(v))
    z_s = Re * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_s, y_s, z_s, rstride=4, cstride=4, color='grey')

    # Electron motion, ion motion
    lne, = ax.plot(*Xe.T[:3],color='green')
    lni, = ax.plot(*Xi.T[:3],color='orange')
    # The particles instantaneous positions are indicated by circles from 
    # a scatter plot, scaled according to particle mass. depthshade=0
    # ensures that the colour doesn't fade as the the particle's distance
    # from the observer increases.
    SCALE = 4
    particles = ax.scatter(*np.vstack((Xe[0][:3], Xi[0][:3])).T,
                           s=(SCALE, SCALE),
                           c=('tab:green', 'tab:orange'), depthshade=0)
    # Some tidying up and labelling of the axes.
    setup_axes(ax)
    return fig, ax, lne, lni, particles

def animate(i):
    """The main animation function, called for each frame."""
    
    def animate_particle(X, ln, i):
        """Plot the trajectory of a particle up to step i."""
        
        ln.set_data(X[:i,0], X[:i,1])
        ln.set_3d_properties(X[:i,2])
        particles._offsets3d = np.vstack((Xe[i][:3], Xi[i][:3])).T
    animate_particle(Xe, lne, i)
    animate_particle(Xi, lni, i)

fig, ax, lne, lni, particles = init()
anim = animation.FuncAnimation(fig, animate, frames=Frames, interval=1, blit=False)

plt.show()
