import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	m, k, req, xp, yp = p
	r, v, theta, omega = ic

	sub = {M:m, R:r, Rdot:v, THETA:theta, THETAdot:omega}
	for i in range(4):
		sub[X[i]] = xp[i]
		sub[Y[i]] = yp[i]
		sub[Req[i]] = req[i]
		sub[K[i]] = k[i]
	
	diff_eq = [v,A.subs(sub),omega,ALPHA.subs(sub)]

	print(ti)

	return diff_eq

#-----------------------------------------------------

M, t = sp.symbols('M t')
K = sp.symbols('K0:4')
Req = sp.symbols('Req0:4')
X = sp.symbols('X0:4')
Y = sp.symbols('Y0:4')
R, THETA = dynamicsymbols('R THETA')

XR = R * sp.cos(THETA)
YR = R * sp.sin(THETA)

XRdot = XR.diff(t,1)
YRdot = YR.diff(t,1)
Rdot = R.diff(t,1)
THETAdot = THETA.diff(t,1)

dR = [sp.simplify(sp.sqrt((XR - X[i])**2 + (YR - Y[i])**2) - Req[i]) for i in range(4)]

T = sp.Rational(1,2) * M * (XRdot**2 + YRdot**2)

V = 0
for i in range(4):
	V += K[i] * dR[i]**2
V *= sp.Rational(1, 2)

L = T - sp.simplify(V)

dLdR = L.diff(R,1)
dLdRdot = L.diff(Rdot,1)
ddtdLdRdot = dLdRdot.diff(t,1)

dLdTHETA = L.diff(THETA,1)
dLdTHETAdot = L.diff(THETAdot,1)
ddtdLdTHETAdot = dLdTHETAdot.diff(t,1)

dLR = sp.simplify(ddtdLdRdot - dLdR)
dLTHETA = sp.simplify(ddtdLdTHETAdot - dLdTHETA)

Rddot = R.diff(t,2)
THETAddot = THETA.diff(t,2)

sol = sp.solve([dLR,dLTHETA],(Rddot,THETAddot))

A = sp.simplify(sol[Rddot])
ALPHA = sol[THETAddot]

#--------------------------------------------------

m = 1
k = np.array([25, 25, 25, 25])
req = np.array([2.5, 2.5, 2.5, 2.5])
xp = np.array([2.5, 0, 5, 2.5])
yp = np.array([0, 2.5, 2.5, 5])
ro = 5
vo = 0
thetao = 20
omegao = 0
cnvrt = np.pi/180
thetao *= cnvrt
omegao *= cnvrt
mr = 0.25
tf = 1 

p = m, k, req, xp, yp

ic = ro, vo, thetao, omegao


nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args=(p,))

x = np.asarray([XR.subs({R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)],dtype=float)
y = np.asarray([YR.subs({R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)],dtype=float)

ke = np.asarray([T.subs({M:m, R:rth[i,0], Rdot:rth[i,1], THETA:rth[i,2], THETAdot:rth[i,3]}) for i in range(nframes)])
pe = np.asarray([V.subs({K[0]:k[0], K[1]:k[1], K[2]:k[2], K[3]:k[3], Req[0]:req[0], Req[1]:req[1], Req[2]:req[2], Req[3]:req[3],\
		 X[0]:xp[0], Y[0]:yp[0], X[1]:xp[1], Y[1]:yp[1], X[2]:xp[2], Y[2]:yp[2], X[3]:xp[3], Y[3]:yp[3],\
		 R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)])
E = ke + pe

#-------------------------------------------------------------

xmax = max(x) + 2 * mr if max(x) > max(xp) else max(xp) + 2 * mr
xmin = min(x) - 2 * mr if min(x) < min(xp) else min(xp) - 2 * mr
ymax = max(y) + 2 * mr if max(y) > max(yp) else max(yp) + 2 * mr
ymin = min(y) - 2 * mr if min(y) < min(yp) else min(yp) - 2 * mr

r = np.zeros((4,nframes))
r = np.asarray([np.sqrt((xp[i] - x)**2 + (yp[i] - y)**2) for i in range(4)])
theta = np.zeros((4,nframes))
theta = np.asarray([np.arccos((yp[i] - y)/r[i]) for i in range(4)])
rmax = np.asarray([max(r[i]) for i in range(4)])
nl = np.asarray([int(np.ceil(i / (2 * mr))) for i in rmax])
l = np.zeros((4,nframes))
l = np.asarray([(r[i] - mr)/nl[i] for i in range(4)])
h = np.sqrt(mr**2 - (0.5 * l)**2)
flipa = np.zeros((4,nframes))
flipb = np.zeros((4,nframes))
flipc = np.zeros((4,nframes))
for i in range(4):
	flipa[i] = np.asarray([-1 if x[j]>xp[i] and y[j]<yp[i] else 1 for j in range(nframes)])
	flipb[i] = np.asarray([-1 if x[j]<xp[i] and y[j]>yp[i] else 1 for j in range(nframes)])
	flipc[i] = np.asarray([-1 if x[j]<xp[i] else 1 for j in range(nframes)])
xlo = np.zeros((4,nframes))
ylo = np.zeros((4,nframes))
for i in range(4):
	xlo[i] = x + np.sign((yp[i] - y) * flipa[i] * flipb[i]) * mr * np.sin(theta[i])
	ylo[i] = y + mr * np.cos(theta[i])
xl = np.zeros((4,max(nl),nframes))
yl = np.zeros((4,max(nl),nframes))
for i in range(4):
	xl[i][0] = xlo[i] + np.sign((yp[i]-y)*flipa[i]*flipb[i]) * 0.5 * l[i] * np.sin(theta[i]) - np.sign((yp[i]-y)*flipa[i]*flipb[i]) * flipc[i] * h[i] * np.sin(np.pi/2 - theta[i])
	yl[i][0] = ylo[i] + 0.5 * l[i] * np.cos(theta[i]) + flipc[i] * h[i] * np.cos(np.pi/2 - theta[i])
	for j in range(1,nl[i]):
		xl[i][j] = xlo[i] + np.sign((yp[i]-y)*flipa[i]*flipb[i]) * (0.5 + j) * l[i] * np.sin(theta[i]) - np.sign((yp[i]-y)*flipa[i]*flipb[i]) * flipc[i] * (-1)**j * h[i] * np.sin(np.pi/2 - theta[i])
		yl[i][j] = ylo[i] + (0.5 + j) * l[i] * np.cos(theta[i]) + flipc[i] * (-1)**j * h[i] * np.cos(np.pi/2 - theta[i])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x[frame],y[frame]),radius=mr,fc='xkcd:red')
	plt.gca().add_patch(circle)
	for i in range(4):
		circle=plt.Circle((xp[i],yp[i]),radius=0.5*mr,fc='xkcd:cerulean')
		plt.gca().add_patch(circle)
		plt.plot([xlo[i][frame],xl[i][0][frame]],[ylo[i][frame],yl[i][0][frame]],'xkcd:cerulean')
		plt.plot([xl[i][nl[i]-1][frame],xp[i]],[yl[i][nl[i]-1][frame],yp[i]],'xkcd:cerulean')
		for j in range(nl[i]-1):
			plt.plot([xl[i][j][frame],xl[i][j+1][frame]],[yl[i][j][frame],yl[i][j+1][frame]],'xkcd:cerulean')
	plt.title("Another Springy Thingy")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
#writervideo = animation.FFMpegWriter(fps=nfps)
#ani.save('another_springy_thingy.mp4', writer=writervideo)
plt.show()









