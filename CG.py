from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import ipywidgets as ipw


def f(x, A, b, c):
    return float(0.5 * x.T * A * x - b.T * x + c)

def SD(A, b, x, imax=10, epsilon = 0.01):
    steps=np.asarray(x)
    i = 0
    r = b - A * x
    delta = r.T * r
    delta0 = delta
    while i < imax and delta > epsilon**2 * delta0:
        q = A * r
        alpha = float(delta / (r.T * q))
        x = x + alpha * r
        if i%50 == 0:
            r = b - A * x
        else:
            r = r - alpha * q
        delta = r.T * r
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def GD(A, b, x, alpha = 0.1, imax=30, epsilon = 0.01):
    steps=np.asarray(x)
    i = 0
    r = b - A * x
    delta = r.T * r
    delta0 = delta
    while i < imax and delta > epsilon**2 * delta0:
        q = A * r
        x = x + alpha * r
        if i%50 == 0:
            r = b - A * x
        else:
            r = r - alpha * q
        delta = r.T * r
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def CG(A, b, x, imax=10, epsilon = 0.01):
    steps=np.asarray(x)
    i = 0
    r = b - A * x
    d = r.copy()
    delta_new = r.T * r
    delta_0 = delta_new
    while i < imax and delta_new > epsilon**2 * delta_0:
        q = A * d
        alpha = float(delta_new / (d.T * q))
        x = x + alpha * d
        if i%50 == 0:
            r = b - A * x
        else:
            r = r - alpha * q
        delta_old = delta_new
        delta_new = r.T * r
        beta = float(delta_new / delta_old)
        d = r + beta * d
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def PCG(A, b, x, M_inv, imax=10, epsilon = 0.01):
    steps=np.asarray(x)
    i = 0
    r = b - A * x
    d = M_inv * r
    delta_new = r.T * d
    delta_0 = delta_new
    while i < imax and delta_new > epsilon**2 * delta_0:
        q = A * d
        alpha = float(delta_new / (d.T * q))
        x = x + alpha * d
        if i%50 == 0:
            r = b - A * x
        else:
            r = r - alpha * q
        s = M_inv * r
        delta_old = delta_new
        delta_new = r.T * s
        beta = float(delta_new / delta_old)
        d = s + beta * d
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def plotAb2D(A, b, fig=None, fs=20):
    if fig==None:
        fig = plt.figure(figsize=(8,8), num='Figure 1') 
    ax = fig.gca()
    xl = -4
    xr = 6
    
    yl = (b[0,0] - xl*A[0,0])/A[0,1]
    yr = (b[0,0] - xr*A[0,0])/A[0,1]
    plt.plot([xl, xr], [yl, yr], color='g', linestyle='-', linewidth=2)
    
    yl = (b[1,0] - xl*A[1,0])/A[1,1]
    yr = (b[1,0] - xr*A[1,0])/A[1,1]
    plt.plot([xl, xr], [yl, yr], color='b', linestyle='-', linewidth=2)
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.text(0, 4.2, r'$x_2$', fontsize = 20)
    plt.text(-4.5, 0, r'$x_1$', fontsize = 20)
    plt.text(0.8, 0.8, r'$%.1fx_1 + %.1fx_2 = %.1f$'%(A[0,0], A[0,1],b[0,0]), fontsize = fs, color = 'g')
    plt.text(3, -2, r'$%.fx_1 + %.1fx_2 = %.1f$'%(A[1,0], A[1,1],b[1,0]), fontsize = fs, color = 'b')

    plt.axis([-4,6,-6,4])

def plotAbc3D(A, b, c, fig=None, alpha=1):
    if fig==None:
        fig = plt.figure(figsize=(8,8), num='Figure 2')
    ax = fig.gca(projection='3d')
    size = 20
    x1 = np.linspace(-4, 6, size)
    x2 = np.linspace(-6, 4, size)
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, A, b, c)
    ax.plot_surface(x1, x2, zs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=alpha)
    ax.set_xlabel(r'$x_1$', fontsize = 20)
    ax.set_ylabel(r'$x_2$', fontsize = 20)
    ax.set_zlabel(r'$f(x)$', fontsize = 20)

def plotcontours(A, b, c, fig=None, pltrange = (-4, 6, -6, 4, 20)):
    if fig==None:
        fig = plt.figure(figsize=(8,8), num='Figure 3')
    size = pltrange[4]
    x1 = np.linspace(pltrange[0], pltrange[1], size)
    x2 = np.linspace(pltrange[2], pltrange[3], size)
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, A, b, c)
    cp = plt.contour(x1, x2, zs, 20)
    plt.clabel(cp, inline=1, fontsize=10)
    plt.text(pltrange[0], pltrange[3]+0.2, r'$x_2$', fontsize = 20)
    plt.text(pltrange[1]+0.2, pltrange[2], r'$x_1$', fontsize = 20)

def vectorfield(A, b, c, fig=None):
    if fig==None:
        fig = plt.figure(figsize=(8,8), num='Figure 4')
    size = 20
    x1 = np.linspace(-4, 6, size)
    x2 = np.linspace(-6, 4, size)
    x1, x2 = np.meshgrid(x1, x2)
    plt.quiver(x1, x2, A[0,0]*x1 + A[0,1]*x2 - b[0,0] + c, A[1,0]*x1 + A[1,1]*x2 - b[1,0] + c)
    plotcontours(A, b, c, fig)

def fig_A():
    A_00 = 3.; A_01 = 2.; A_10 = 2.; A_11 = 6.
    b_0 = 2.; b_1 = -8.
    c_0 = 0.  

    fig = plt.figure(figsize=(12,5), num='Figure A')
    ax1 = fig.add_subplot(1, 2, 1)

    # lines
    xl = -4
    xr = 6
    yl = (b_0 - xl*A_00)/A_01
    yr = (b_0 - xr*A_00)/A_01
    line1, = ax1.plot([xl, xr], [yl, yr], color='g', linestyle='-', linewidth=2)
    yl = (b_1 - xl*A_10)/A_11
    yr = (b_1 - xr*A_10)/A_11
    line2, = ax1.plot([xl, xr], [yl, yr], color='b', linestyle='-', linewidth=2)
    eq1 = ax1.text(-2, 4.6, r'$%.1fx_1 + %.1fx_2 = %.1f$'%(A_00, A_01, b_0), fontsize = 14, color = 'g')
    eq2 = ax1.text(-2, 4.1, r'$%.1fx_1 + %.1fx_2 = %.1f$'%(A_10, A_11, b_1), fontsize = 14, color = 'b')

    plt.xlabel(r'$x_1$', fontsize = 20)
    plt.ylabel(r'$x_2$', fontsize = 20)
    plt.axis([-4,6,-6,4])

    # create 3D points
    size = 20
    x1 = np.linspace(-4, 6, size)
    x2 = np.linspace(-6, 4, size)
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), c_0)

    # plot contours
    ax1.contour(x1, x2, zs, 20)

    # gradients
    ax1.quiver(x1, x2, A_00*x1 + A_01*x2 - b_0 + c_0, A_10*x1 + A_11*x2 - b_1 + c_0)

    # plot 3D surface
    ax2 = fig.add_subplot(1, 2, 2 , projection='3d')
    size = 20
    ax2.plot_surface(x1, x2, zs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=0.8)
    ax2.set_xlabel(r'$x_1$', fontsize = 20)
    ax2.set_ylabel(r'$x_2$', fontsize = 20)
    ax2.set_zlabel(r'$f(x)$', fontsize = 20)
    return (ax1, ax2, line1, line2, eq1, eq2)
    
def sliders_figA(hdls):
    import matplotlib
    (ax1, ax2, line1, line2, eq1, eq2) = hdls
    size = 20
    x1 = np.linspace(-4, 6, size)
    x2 = np.linspace(-6, 4, size)
    x1, x2 = np.meshgrid(x1, x2)


    def update_plots(A_00, A_01, A_10, A_11, b_0, b_1, c_0):
        # update lines
        xl = -4
        xr = 6
        yl = (b_0 - xl*A_00)/(A_01 + 0.001)
        yr = (b_0 - xr*A_00)/(A_01 + 0.001)
        line1.set_ydata([yl, yr])
        yl = (b_1 - xl*A_10)/(A_11 + 0.001)
        yr = (b_1 - xr*A_10)/(A_11 + 0.001)
        line2.set_ydata([yl, yr])
        eq1.set_text(r'$%.1fx_1 + %.1fx_2 = %.1f$'%(A_00, A_01,b_0))
        eq2.set_text(r'$%.1fx_1 + %.1fx_2 = %.1f$'%(A_10, A_11,b_1))
        
        # update 3D points
        zs = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = np.matrix([[x1[i,j]], [x2[i,j]]])
                zs[i,j] = f(x, np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), c_0)
                
        # update contours
        [h.remove() for h in ax1.get_children() if isinstance(h, matplotlib.collections.LineCollection)]
        ax1.contour(x1, x2, zs, 20)
        
        # update quiver
        [h.remove() for h in ax1.get_children() if isinstance(h, matplotlib.quiver.Quiver)]
        ax1.quiver(x1, x2, A_00*x1 + A_01*x2 - b_0 + c_0, A_10*x1 + A_11*x2 - b_1 + c_0)
        
        # update 3D surface
        ax2.get_children()[1].remove()
        ax2.plot_surface(x1, x2, zs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=0.8)

    # define slider layout
    v_layout = ipw.Layout(display='flex',
                        flex_flow='column',
                        align_items='stretch',
                        justify_content = 'center',
                        width='100%')

    h_layout = ipw.Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        justify_content = 'center',
                        width='100%')
    
    A_00 = ipw.FloatSlider(3.0, min=-10., max=10., width='auto')
    A_01 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    A_10 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    A_11 = ipw.FloatSlider(6.0, min=-10., max=10., width='auto')
    b_0 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    b_1 = ipw.FloatSlider(-8.0, min=-10., max=10., width='100%')
    c_0 = ipw.FloatSlider(0, min=-10., max=10., width='auto')


    form = ipw.Box([ipw.Box([ipw.Box([A_00, A_10], layout=v_layout), ipw.Box([A_01, A_11], layout=v_layout)], layout=h_layout),
                  ipw.Box([ipw.Box([b_0, c_0], layout=v_layout), b_1], layout=h_layout)
                 ], layout=v_layout)
    display(form)
    ipw.interactive(update_plots, A_00=A_00, A_01=A_01, A_10=A_10, A_11=A_11, b_0=b_0, b_1=b_1, c_0=c_0)

def fig5():
    fig = plt.figure(figsize=(9.5,7.5), num='Figure 5')
    fig.add_subplot(2, 2, 1, projection='3d')
    A = np.matrix([[3., 2.], [2., 6.]]); b = np.matrix([[2.], [-8.]])
    plotAbc3D(A, b, 0., fig, alpha=0.7)
    plt.title('(a)')
    fig.add_subplot(2, 2, 2, projection='3d')
    A = np.matrix([[-6., -1.], [-1., -6.]]); b = np.matrix([[0.], [0.]])
    plotAbc3D(A, b, 0., fig, alpha=0.7)
    plt.title('(b)')
    fig.add_subplot(2, 2, 3, projection='3d')
    A = np.matrix([[1., 2.], [2., 4.]]); b = np.matrix([[-1.5], [-3.]])
    plotAbc3D(A, b, 0., fig, alpha=0.7)
    plt.title('(c)')
    fig.add_subplot(2, 2, 4, projection='3d')
    A = np.matrix([[8., -1.], [-1., -6.]]); b = np.matrix([[0.], [0.]])
    plotAbc3D(A, b, 0., fig, alpha=0.7)
    plt.title('(d)')

def fig6(A, b, c):
    fig = plt.figure(figsize=(8,8), num='Figure 6')
    x = np.matrix([[-2.0],[-2.0]])
    r = b - A * x
    delta = float(r.T * r)
    ax = fig.add_subplot(2, 2, 1)
    plotcontours(A, b, c, fig)
    plt.plot([x[0,0], (x+r)[0,0]], [x[1,0], (x+r)[1,0]], 'g')
    plt.plot(x[0,0], x[1,0], 'k', marker='o', markersize=6)
    plt.text(-2.5, -1.5, r'$x[0]$', fontsize = 14)
    plt.plot(2, -2, color='grey', marker='o', markersize=6)
    plt.text(2, -1.5, r'$x$', fontsize = 14)
    ax.arrow(2,-2,-3.5,0, color='grey', lw=2, width=.02)
    plt.axis([-4,6,-6,4])
    ax.set_title('(a)')

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    plotAbc3D(A, b, c, fig, alpha=0.5)
    xx = np.arange(-4,6)
    yy = (xx+2)*r[1,0]/r[0,0] - 2
    zz = 0.5*(A[0,0]*xx*xx + A[0,1]*xx*yy + A[1,0]*xx*yy + A[1,1]*yy*yy) - b[0,0]*xx - b[1,0]*yy
    ax.plot(xx,yy,zz, alpha=0.5)
    ax.set_title('(b)')

    ax = fig.add_subplot(2, 2, 3)
    alpha = np.arange(20)/30.
    tmp = x + r*alpha
    ax.plot(alpha, [f(tmp[:,i],A,b,c) for i in range(20)])
    ax.set_xlabel(r'$\alpha$')
    plt.text(0.1, 120, r'$f(x[i] + \alpha r[i])$', fontsize = 20)
    ax.set_title('(c)')

    ax = fig.add_subplot(2, 2, 4)
    plotcontours(A, b, c, fig)
    alpha = float(delta / (r.T * (A * r)))
    xnew = x + alpha*r
    rnew = b - A*xnew
    plt.plot([x[0,0], xnew[0,0]], [x[1,0], xnew[1,0]], 'k-', marker='o', markersize=6)
    plt.text(0.5, -1, r'$x[1]$', fontsize = 14)
    ax.arrow(xnew[0,0], xnew[1,0], -rnew[0,0]/10., -rnew[1,0]/10., color = 'k', lw=1, width=.01)
    plt.axis([-4,6,-6,4])
    ax.set_title('(d)')

def fig7(A, b, c):
    fig = plt.figure(figsize=(8,8), num='Figure 7')
    ax = fig.add_subplot(1, 1, 1)
    x = np.matrix([[-2.0],[-2.0]])
    r = b - A * x
    delta = float(r.T * r)
    alpha = float(delta / (r.T * (A * r)))
    plotcontours(A, b, c, fig)
    plt.plot([x[0,0], (x+r)[0,0]], [x[1,0], (x+r)[1,0]], 'g')
    plt.plot(x[0,0], x[1,0], 'k', marker='o', markersize=6)
    plt.plot(2, -2, color='grey', marker='o', markersize=6)
    plt.text(2, -1.8, r'$x$', fontsize = 18)
    for i in range(2,7):
        xnew = x + 0.25*i*alpha*r
        rnew = b - A*xnew
        rnew /= rnew.T*rnew
        rp = float(r.T*rnew)
        ax.arrow(xnew[0,0], xnew[1,0], -rnew[0,0]*3, -rnew[1,0]*3, color = 'k', lw=1, width=.005)
        if abs(rp) > 0.01:
            ax.arrow(xnew[0,0], xnew[1,0], -r[0,0]*3*rp/delta, -r[1,0]*3*rp/delta, color = 'k', linestyle = ':', lw=2, width=.005)
    plt.axis([-2,3,-3,2])

def fig8(A, b, c):
    x = np.matrix([[-2.0],[-2.0]])
    steps = SD(A, b, x)
    fig = plt.figure(figsize=(8,8), num='Figure 8')
    plotcontours(A, b, c, fig)
    plt.plot(steps[0,:], steps[1,:], '-o')

def fig_B():
    A_00 = 3.; A_01 = 2.; A_10 = 2.; A_11 = 6.
    b_0 = 2.; b_1 = -8.
    c_0 = 0.  
    x_0 = -2.; x_1 =-2.

    fig = plt.figure(figsize=(8,8), num='Figure B')
    ax1 = fig.add_subplot(1, 1, 1)

    # create 3D points
    size = 30
    x1 = np.linspace(-10, 10, size)
    x2 = np.linspace(-10, 10, size)
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), c_0)

    # plot contours
    ax1.contour(x1, x2, zs, 20)

    steps = SD(np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), np.matrix([[x_0],[x_1]]))
    ax1.plot(steps[0,:], steps[1,:], '-o', color='m')
    return ax1
    
def sliders_figB(ax1):
    import matplotlib
    size = 30
    x1 = np.linspace(-10, 10, size)
    x2 = np.linspace(-10, 10, size)
    x1, x2 = np.meshgrid(x1, x2)

    def update_plots(A_00, A_01, A_10, A_11, b_0, b_1, x_0, x_1, alpha, gd):
        # update 3D points
        zs = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = np.matrix([[x1[i,j]], [x2[i,j]]])
                zs[i,j] = f(x, np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), 0)
                
        # update contours
        [h.remove() for h in ax1.get_children() if isinstance(h, matplotlib.collections.LineCollection)]
        ax1.contour(x1, x2, zs, 20)
        
        # update steps
        [h.remove() for h in ax1.get_children() if isinstance(h, matplotlib.lines.Line2D)]
        if gd == 'Gradient Descent':
            steps = GD(np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), np.matrix([[x_0],[x_1]]), 
                alpha = alpha)
        else:
            steps = SD(np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), np.matrix([[x_0],[x_1]]))
        ax1.plot(steps[0,:], steps[1,:], '-o', color='m')
        
        plt.axis([-10,10,-10,10])

        
    # define slider layout
    v_layout = ipw.Layout(display='flex',
                        flex_flow='column',
                        align_items='stretch',
                        justify_content = 'center',
                        width='100%')

    h_layout = ipw.Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        justify_content = 'center',
                        width='100%')
    
    A_00 = ipw.FloatSlider(3.0, min=-10., max=10., width='auto')
    A_01 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    A_10 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    A_11 = ipw.FloatSlider(6.0, min=-10., max=10., width='auto')
    b_0 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    b_1 = ipw.FloatSlider(-8.0, min=-10., max=10., width='auto')
    x_0 = ipw.FloatSlider(-2.0, min=-10., max=10., width='auto')
    x_1 = ipw.FloatSlider(-2.0, min=-10., max=10., width='auto')
    alpha = ipw.FloatSlider(0.1, min=0., max=1., width='auto', step = 0.01)
    gd = ipw.ToggleButtons(options=['Steepest Decent', 'Gradient Descent'], description='Method: ')

    form = ipw.Box([ipw.Box([ipw.Box([A_00, A_10], layout=v_layout), 
                             ipw.Box([A_01, A_11], layout=v_layout)], layout=h_layout),
                    ipw.Box([ipw.Box([b_0, x_0, alpha], layout=v_layout), 
                             ipw.Box([b_1, x_1, ipw.Box([gd], layout=h_layout)], layout=v_layout)], layout=h_layout)], 
                    layout = v_layout)
    display(form)
    ipw.interactive(update_plots, A_00=A_00, A_01=A_01, A_10=A_10, A_11=A_11, 
                    b_0=b_0, b_1=b_1, x_0=x_0, x_1=x_1, alpha=alpha, gd=gd)

def fig9():
    fig = plt.figure(figsize=(9,2), num='Figure 9')
    B = np.matrix([[-.25, -.25], [-.25, -.25]])
    v = np.matrix([[1],[1]])
    ax = fig.add_subplot(1,4,1)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    ax.text(0.65, 0.25, r'$v$', fontsize=15)
    plt.axis([-1, 1, -1, 1])
    v = B*v
    ax = fig.add_subplot(1,4,2)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.45, -0.7, r'$Bv$', fontsize=15)
    v = B*v
    ax = fig.add_subplot(1,4,3)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(0.45, 0.45, r'$B^2v$', fontsize=15)
    v = B*v
    ax = fig.add_subplot(1,4,4)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.55, -0.45, r'$B^3v$', fontsize=15)
    
def fig10():
    fig = plt.figure(figsize=(9,2), num='Figure 10')
    B = np.matrix([[1, -1], [-1, 1]])
    v = np.matrix([[-.11],[.11]])
    ax = fig.add_subplot(1,4,1)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    ax.text(-0.45, 0.25, r'$v$', fontsize=15)
    plt.axis([-1, 1, -1, 1])
    v = B*v
    ax = fig.add_subplot(1,4,2)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.85, 0.25, r'$Bv$', fontsize=15)
    v = B*v
    ax = fig.add_subplot(1,4,3)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.95, 0.25, r'$B^2v$', fontsize=15)
    v = B*v
    ax = fig.add_subplot(1,4,4)
    ax.arrow(0, 0, v[0,0], v[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.95, 0.25, r'$B^3v$', fontsize=15)

def fig11():
    fig = plt.figure(figsize=(9,2), num='Figure 11')
    B = np.matrix([[-0.65, 1.35], [1.35, -.65]])
    v1 = np.matrix([[-.9],[-.9]])
    v2 = np.matrix([[0.1],[-0.1]])
    x = v1 + v2
    ax = fig.add_subplot(1,4,1)
    ax.arrow(0, 0, x[0,0], x[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, v1[0,0], v1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.arrow(0, 0, v2[0,0], v2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    ax.text(-0.55, -1, r'$x$', fontsize=15)
    ax.text(-0.65, -0.3, r'$v_1$', fontsize=15)
    ax.text(0.1, -0.35, r'$v_2$', fontsize=15)
    plt.axis([-1, 1, -1, 1])
    x = B*x
    v1 = B*v1
    v2 = B*v2
    ax = fig.add_subplot(1,4,2)
    ax.arrow(0, 0, x[0,0], x[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, v1[0,0], v1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.arrow(0, 0, v2[0,0], v2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.95, -0.25, r'$Bx$', fontsize=15)
    x = B*x
    v1 = B*v1
    v2 = B*v2
    ax = fig.add_subplot(1,4,3)
    ax.arrow(0, 0, x[0,0], x[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, v1[0,0], v1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.arrow(0, 0, v2[0,0], v2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])
    ax.text(-0.55, -1.1, r'$B^2x$', fontsize=15)
    x = B*x
    v1 = B*v1
    v2 = B*v2
    ax = fig.add_subplot(1,4,4)
    ax.arrow(0, 0, x[0,0], x[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, v1[0,0], v1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.arrow(0, 0, v2[0,0], v2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    ax.text(-1.25, 0.05, r'$B^3x$', fontsize=15)

def fig12(A, b, c):
    fig = plt.figure(figsize=(8,8), num='Figure 12')
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A,b,c, fig)
    plt.text(1, -2, '2', fontsize = 18)
    plt.text(2.5, -1.5, '7', fontsize = 18)
    ax.arrow(2, -2, -2, 1, color = 'k', lw=1, width=.005)
    ax.arrow(2, -2, 1, 2, color = 'k', lw=1, width=.005)
    plt.axis([-4,6,-6,4])

def fig13():
    fig = plt.figure(figsize=(8,12), num='Figure 13')
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])    
    B = np.matrix([[0., -2./3.], [-1./3., 0.]])
    z = np.matrix([[2./3.],[-4./3.]])
    v1 = np.matrix([[-2.],[-(2.**0.5)]])
    v2 = np.matrix([[-2.],[(2.**0.5)]])
    x = np.matrix([[-2.], [-2.]])
    e = np.matrix([[-4],[0]])

    ax = fig.add_subplot(3,2,1)
    plotcontours(A,b,0, fig)
    plt.text(2.7, -2, '-0.47', fontsize = 15)
    plt.text(-0.4, -2, '0.47', fontsize = 15)
    ax.arrow(2, -2, v1[0,0], v1[1,0], head_width = .2, head_length = .3, length_includes_head = True, color = 'k')
    ax.arrow(2, -2, v2[0,0], v2[1,0], head_width = .2, head_length = .3, length_includes_head = True, color = 'k')
    ax.set_title('(a)')
    plt.axis([-4, 6, -6, 4])
    steps = np.asarray(x)
    esteps = np.asarray(e)
    v1steps = np.asarray(v1)
    v2steps = np.asarray(v2)
    for i in range(5):
        x = B*x + z
        e = B*e
        v1 = -v1 * 2.**0.5 / 3.
        v2 = v2 * 2.**0.5 / 3.
        steps = np.append(steps, np.asarray(x), axis=1)
        esteps = np.append(esteps, np.asarray(e), axis=1)
        v1steps = np.append(v1steps, np.asarray(v1), axis=1)
        v2steps = np.append(v2steps, np.asarray(v2), axis=1)

    ax = fig.add_subplot(3,2,2)
    plotcontours(A,b,0, fig)
    ax.plot(steps[0,:], steps[1,:], '-o')
    ax.text(2.2, -2.2, r'$x$', fontsize=20)
    ax.text(-2.2, -1.5, r'$x_{[0]}$', fontsize=20)
    ax.set_title('(b)')
    plt.axis([-4, 6, -6, 4])

    ax = fig.add_subplot(3,2,3)
    ax.arrow(2., -2., esteps[0,0], esteps[1,0], head_width = .2, head_length = .3, length_includes_head = True, color='k')
    ax.arrow(2., -2., v1steps[0,0], v1steps[1,0], head_width = .2, head_length = .3, length_includes_head = True, color='k', ls=':')
    ax.arrow(2., -2., v2steps[0,0], v2steps[1,0], head_width = .2, head_length = .3, length_includes_head = True, color='k', ls=':')
    ax.text(-3.3, -2, r'$e_{[0]}$', fontsize=20)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('(c)')
    plt.axis([-4, 6, -6, 4])

    ax = fig.add_subplot(3,2,4)
    ax.arrow(2., -2., esteps[0,1], esteps[1,1], head_width = .2, head_length = .3, length_includes_head = True, color='k')
    ax.arrow(2., -2., v1steps[0,1], v1steps[1,1], head_width = .2, head_length = .3, length_includes_head = True, color='k', ls=':')
    ax.arrow(2., -2., v2steps[0,1], v2steps[1,1], head_width = .2, head_length = .3, length_includes_head = True, color='k', ls=':')
    ax.text(2.2, -1, r'$e_{[1]}$', fontsize=20)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('(d)')
    plt.axis([-4, 6, -6, 4])

    ax = fig.add_subplot(3,2,5)
    ax.arrow(2., -2., esteps[0,2], esteps[1,2], head_width = .2, head_length = .3, length_includes_head = True, color='k')
    ax.arrow(2., -2., v1steps[0,2], v1steps[1,2], head_width = .2, head_length = .3, length_includes_head = True, color='k', ls=':')
    ax.arrow(2., -2., v2steps[0,2], v2steps[1,2], head_width = .2, head_length = .3, length_includes_head = True, color='k', ls=':')
    ax.text(-0, -2.2, r'$e_{[2]}$', fontsize=20)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('(e)')
    plt.axis([-4, 6, -6, 4])
    ax = fig.add_subplot(3,2,6)
    ax.plot(steps[0,:], steps[1,:], '-o')
    ax.text(2.5, -2.2, r'$x$', fontsize=20)
    ax.text(-3.5, -2.5, r'$x_{[0]}$', fontsize=20)
    ax.plot([2-5*v1steps[0,0], 2+5*v1steps[0,0]],[-2-5*v1steps[1,0], -2+5*v1steps[1,0]], color='#777777')
    ax.plot([2-5*v2steps[0,0], 2+5*v2steps[0,0]],[-2-5*v2steps[1,0], -2+5*v2steps[1,0]], color='#777777')
    for i in range(4):
        ax.arrow(2., -2., v1steps[0,i], v1steps[1,i], head_width = .3, head_length = .3, length_includes_head = True, color='k')
        ax.arrow(2., -2., v2steps[0,i], v2steps[1,i], head_width = .3, head_length = .3, length_includes_head = True, color='k')
        ax.plot([steps[0,i], 2+v1steps[0,i]],[steps[1,i], -2+v1steps[1,i]], color='#777777')
        ax.plot([steps[0,i], 2+v2steps[0,i]],[steps[1,i], -2+v2steps[1,i]], color='#777777')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    ax.set_title('(f)')
    plt.axis([-4, 6, -6, 4])

def fig14():
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])
    v = np.matrix([[-2.],[1]])
    fig = plt.figure(figsize=(8,8), num='Figure 14')
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A, b, 0, fig)
    ax.arrow(2+2.5*v[0,0], -2+2.5*v[1,0], -2.5*v[0,0], -2.5*v[1,0], head_width = .2, head_length = .3, length_includes_head = True, color = 'k')
    ax.plot(2+2.5*v[0,0], -2+2.5*v[1,0], 'o')
    plt.axis([-4,6,-6,4])


def fig15():
    A = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    b = np.matrix([[-2.0], [-1.0]])
    v = np.matrix([[2.],[1.]])
    fig = plt.figure(figsize=(8,8), num='Figure 15')
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A, b, 0, fig)
    ax.arrow(-2+2.5*v[0,0], -1+2.5*v[1,0], -2.5*v[0,0], -2.5*v[1,0], head_width = .2, head_length = .3, length_includes_head = True, color = 'k')
    ax.plot(-2+2.5*v[0,0], -1+2.5*v[1,0], 'o')
    plt.axis([-4,6,-6,4])

def fig16():
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])
    v1 = np.matrix([[2.],[-1]])
    v2 = np.matrix([[1.],[2.]])
    fig = plt.figure(figsize=(8,8), num='Figure 16')
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A, b, 0, fig)
    ax.arrow(2, -2, 1.4*v1[0,0], 1.4*v1[1,0], head_width = .2, head_length = .3, length_includes_head = True, color = 'k')
    ax.arrow(2, -2, .75*v2[0,0], .75*v2[1,0], head_width = .2, head_length = .3, length_includes_head = True, color = 'k')
    plt.axis([-4,6,-6,4])

def fig17():
    fig = plt.figure(figsize=(8,6), num='Figure 17')
    ax = fig.gca(projection='3d')
    size = 20
    kappa = np.linspace(1, 100, size)
    mu = np.linspace(0, 20, size)
    k, m = np.meshgrid(kappa, mu)
    w = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            w[i,j] = 1 - ((k[i,j]**2 + m[i,j]**2)**2/((k[i,j] + m[i,j]**2) * (k[i,j]**3 + m[i,j]**2)))
    ax.plot_surface(k, m, w, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1)
    ax.invert_xaxis()
    ax.set_xlabel(r'$\kappa$', fontsize = 20)
    ax.set_ylabel(r'$\mu$', fontsize = 20)
    ax.set_zlabel(r'$\omega$', fontsize = 20)

def fig18():
    fig = plt.figure(figsize=(8,8), num='Figure 18')
    ax = fig.add_subplot(2,2,1)
    A = np.matrix([[5.0, 0.0], [0.0, 0.5]])
    b = np.matrix([[0.0], [0.0]])
    c = 0
    x = np.matrix([[5],[1]])
    plotcontours(A,b,c,fig,(-5,5,-4,6,20))
    steps = SD(A, b, x)
    ax.plot(steps[0,:], steps[1,:], '-o', color='m')
    ax.set_title('(a)')
    ax.set_ylabel(r'$v_2$', fontsize = 20)

    ax = fig.add_subplot(2,2,2)
    A = np.matrix([[5.0, 0.0], [0.0, 0.5]])
    b = np.matrix([[0.0], [0.0]])
    c = 0
    x = np.matrix([[.5],[5]])
    plotcontours(A,b,c,fig,(-5,5,-4,6,20))
    steps = SD(A, b, x, imax=20)
    ax.plot(steps[0,:], steps[1,:], '-o', color='m')
    ax.set_title('(b)')
    
    ax = fig.add_subplot(2,2,3)
    A = np.matrix([[2.0, 0.0], [0.0, 1.0]])
    b = np.matrix([[0.0], [0.0]])
    c = 0
    x = np.matrix([[5],[1]])
    plotcontours(A,b,c,fig,(-5,5,-4,6,20))
    steps = SD(A, b, x)
    ax.plot(steps[0,:], steps[1,:], '-o', color='m')
    ax.set_title('(c)')
    ax.set_xlabel(r'$v_1$', fontsize = 20)
    ax.set_ylabel(r'$v_2$', fontsize = 20)

    ax = fig.add_subplot(2,2,4)
    A = np.matrix([[2.0, 0.0], [0.0, 1.0]])
    b = np.matrix([[0.0], [0.0]])
    c = 0
    x = np.matrix([[.5],[5]])
    plotcontours(A,b,c,fig,(-5,5,-4,6,20))
    steps = SD(A, b, x)
    ax.plot(steps[0,:], steps[1,:], '-o', color='m')
    ax.set_title('(d)')
    ax.set_xlabel(r'$v_1$', fontsize = 20)    
    
def fig19():
    fig = plt.figure(figsize=(8,8), num='Figure 19')
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])
    c = 0.0
    v1 = np.matrix([[1.0],[2.0]])
    v2 = np.matrix([[-2.0],[1.0]])
    x = np.matrix([[2 + v1[0,0] + 3.5*v2[0,0]], [-2 + v1[1,0] + 3.5*v2[1,0]]])/1.5
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A,b,c, fig)
    ax.arrow(2, -2, v1[0,0], v1[1,0], color = '#777777', head_width = .2, head_length = .3, length_includes_head = True)
    ax.arrow(2, -2, v2[0,0], v2[1,0], color = '#777777', head_width = .2, head_length = .3, length_includes_head = True)
    ax.plot([2 - v1[0,0] - 3.5*v2[0,0], 2 + v1[0,0] + 3.5*v2[0,0]], [-2 - v1[1,0] - 3.5*v2[1,0], -2 + v1[1,0] + 3.5*v2[1,0]], color='k')
    ax.plot([2 - v1[0,0] + 3.5*v2[0,0], 2 + v1[0,0] - 3.5*v2[0,0]], [-2 - v1[1,0] + 3.5*v2[1,0], -2 + v1[1,0] - 3.5*v2[1,0]], color='k')
    steps = SD(A, b, x)
    ax.plot(steps[0,:], steps[1,:], '-o', color='m')
    plt.axis([-4,6,-6,4])

def fig20():
    plt.figure(figsize=(8,8), num='Figure 20')
    size = 99
    kappa = np.linspace(1, 100, size)
    omega = (kappa-1)/(kappa+1)
    plt.plot(kappa, omega)
    plt.minorticks_on()
    plt.grid(which='both')
    plt.xlabel(r'$\kappa$', fontsize=20)
    plt.ylabel(r'$\omega$', fontsize=20)
    plt.axis([0,100,0,1])

def fig21():
    fig = plt.figure(figsize=(8,8), num='Figure 21')
    A = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    b = np.matrix([[2.0], [-2.0]])
    c = 0.0
    v1 = np.matrix([[5.0],[0.0]])
    v2 = np.matrix([[0.0],[-1.0]])
    x = np.matrix([[-3.0], [-3.0]])
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A,b,c, fig)
    ax.plot(x[0,0], x[1,0], 'o', color='k')
    ax.plot(2, -2, 'o', color='k')
    ax.arrow(x[0,0], x[1,0], v1[0,0], v1[1,0], color = 'k', head_width = .2, head_length = .3, length_includes_head = True)
    ax.arrow(2, -2, v2[0,0], v2[1,0], color = '#777777', head_width = .2, head_length = .3, length_includes_head = True)
    ax.text(-3.8, -3, r'$x_{[0]}$', fontsize=20)
    ax.text(2.1, -3, r'$x_{[1]}$', fontsize=20)
    ax.text(2.1, -2, r'$x$', fontsize=20)
    ax.text(-0.3, -3.4, r'$d_{[0]}$', fontsize=20)
    ax.text(1.4, -2.5, r'$e_{[1]}$', fontsize=20)

def fig22():
    fig = plt.figure(figsize=(12,5.5), num='Figure 22')
    
    A = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    b = np.matrix([[1], [-1]])
    c = 0
    v = np.array((-1, 1, -0.8, -1.2, -1.4, -0.5, 1.4, 0.7, -1, 0, 1, -1.4, 1.4, 0, 1, 1,
                  -1, -1, 0.8, -1.2, 0.7, -1., -0.7, 1.4, 0, 1, 2, 0.7, 0, 1, 1, -1)).reshape((2,16))
    x = np.array((1, 5, 1, -3, 5, 5, -3, -3, 3, -1, -5, -1, 3, -5, -5, 3)).reshape((2,8))
    ax = fig.add_subplot(1, 2, 2)
    plotcontours(A,b,c, fig)
    for i in range(8):
        ax.arrow(x[0,i], x[1,i], v[0,2*i], v[1,2*i], color = 'k', head_width = .2, head_length = .3, length_includes_head = True)
        ax.arrow(x[0,i], x[1,i], v[0,2*i+1], v[1,2*i+1], color = 'k', head_width = .2, head_length = .3, length_includes_head = True)   

    A = np.matrix([[1.0, 0.0], [0.0, 9.0]])
    b = np.matrix([[1.], [-9.0]])
    c = 0
    ax = fig.add_subplot(1, 2, 1)
    plotcontours(A,b,c, fig)
    for i in range(8):
        ax.arrow(x[0,i], x[1,i], v[0,2*i], v[1,2*i]/3., color = 'k', head_width = .2, head_length = .3, length_includes_head = True)
        ax.arrow(x[0,i], x[1,i], v[0,2*i+1], v[1,2*i+1]/3., color = 'k', head_width = .2, head_length = .3, length_includes_head = True)   

def fig23():
    fig = plt.figure(figsize=(12,5.5), num='Figure 23')    
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2], [-8]])
    c = 0.0
    x = np.matrix([[-2],[-2]])
    e = np.matrix([[x[0,0]-2],[x[1,0]+2]])
    d = np.matrix([[0],[1]])
    alpha = -float((d.T * A * e)/(d.T * A * d))
    ax = fig.add_subplot(1, 2, 1)
    plotcontours(A,b,c, fig)
    ax.plot(x[0,0], x[1,0], 'o', color='k')
    ax.plot(2, -2, 'o', color='k')
    ax.arrow(2, -2, x[0,0] + alpha*d[0,0]-2, x[1,0] + alpha*d[1,0]+2, color = '#777777', head_width = .2, head_length = .3, length_includes_head = True)
    ax.arrow(x[0,0], x[1,0], alpha*d[0,0], alpha*d[1,0], color = 'k', head_width = .2, head_length = .3, length_includes_head = True)
    ax.text(-2.1, -2.4, r'$x_{[0]}$', fontsize=20)
    ax.text(2.1, -2, r'$x$', fontsize=20)
    ax.text(-2.9, -1.5, r'$d_{[0]}$', fontsize=20)
    ax.text(0, -1.2, r'$e_{[1]}$', fontsize=20)
    ax.set_title('(a)')

    ax = fig.add_subplot(1, 2, 2)
    plotcontours(A,b,c, fig)
    ax.plot(x[0,0], x[1,0], 'o', color='k')
    ax.plot(2, -2, 'o', color='k')
    ax.arrow(2, -2, x[0,0] + alpha*d[0,0]-2, x[1,0] + alpha*d[1,0]+2, color = '#777777', head_width = .2, head_length = .3, length_includes_head = True)
    ax.arrow(x[0,0] + alpha*d[0,0], x[1,0] + alpha*d[1,0], -alpha*d[0,0], -alpha*d[1,0], color = '#777777', head_width = .2, head_length = .3, length_includes_head = True)
    ax.arrow(2, -2, x[0,0]-2, x[1,0]+2, color = 'k', head_width = .2, head_length = .3, length_includes_head = True)
    ax.text(0, -2.5, r'$e_{[0]}$', fontsize=20)
    ax.set_title('(b)')

def fig24():
    fig = plt.figure(figsize=(9,2.5), num='Figure 24')
    u0 = np.matrix([[-1],[.5]])
    u1 = np.matrix([[0],[-1]])
    ax = fig.add_subplot(1,3,1)
    ax.arrow(0, 0, u0[0,0], u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, u1[0,0], u1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(-0.5, 0.35, r'$\mu_0$', fontsize=15)
    ax.text(0.05, -0.5, r'$\mu_1$', fontsize=15)
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])

    ax = fig.add_subplot(1,3,2)
    ax.arrow(0, 0, u0[0,0], u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, u1[0,0], u1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, -u0[0,0], -u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.arrow(-u0[0,0], -u0[1,0], u0[0,0], -u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.text(-0.5, 0.35, r'$d_{[0]}$', fontsize=15)
    ax.text(0.5, -0.1, r'$\mu^+$', fontsize=15)
    ax.text(0.5, -1, r'$\mu^*$', fontsize=15)
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])

    ax = fig.add_subplot(1,3,3)
    ax.arrow(0, 0, u0[0,0], u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, u0[0,0], -u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k', ls=':')
    ax.text(-0.5, 0.35, r'$d_{[0]}$', fontsize=15)
    ax.text(-0.5, -0.5, r'$d_{[1]}$', fontsize=15)
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-1, 1, -1, 1])

def fig25():
    fig = plt.figure(figsize=(8,8), num='Figure 25')
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])
    c = 0.0
    d0 = np.matrix([[1],[0]])
    e0 = np.matrix([[5], [1]])
    alpha = float((d0.T * A * e0)/(d0.T * A * d0))
    x = np.matrix([[-3], [-3]])
    ax = fig.add_subplot(1, 1, 1)
    plotcontours(A,b,c, fig)
    ax.plot(x[0,0], x[1,0], 'ko')
    ax.plot([x[0,0], x[0,0]+alpha, 2], [x[1,0], x[1,0], -2], color='m')
    plt.axis([-4,6,-6,4])

def fig26():
    from matplotlib.patches import Polygon, Ellipse
    fig = plt.figure(figsize=(8,6), num='Figure 26')
    d0 = np.matrix([[2], [0]])
    d1 = np.matrix([[1], [1]])
    ax = fig.add_subplot(1, 1, 1)
    ax.add_patch(Polygon(np.array([[-5,-3], [-3,-1], [4,-1], [2,-3]]),fc='#999999', ec='k'))
    ax.add_patch(Ellipse(np.array([0,0]), 2, 5, 40, fc='None', ec='k'))
    ax.arrow(-4.25, -2.75, d0[0,0], d0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-4.25, -2.75, d1[0,0], d1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(-2.2, -2.75, r'$d_{[0]}$', fontsize=15)
    ax.text(-3.25, -1.75, r'$d_{[1]}$', fontsize=15)
    ax.plot(0,0,'ko')
    ax.text(0, .2, r'$0$', fontsize=15)
    ax.arrow(0, 0, 0, -2.5, head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.plot(0,-2.5,'ko')
    ax.text(0, -2.75, r'$e_{[0]}$', fontsize=15)
    ax.arrow(0, -2.5, 1, 0, head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.plot(1,-2.5,'ko')
    ax.text(1, -2.75, r'$e_{[1]}$', fontsize=15)
    ax.arrow(1, -2.5, .5, .5, head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.plot(1.5,-2,'ko')
    ax.text(1.5, -2.25, r'$e_{[2]}$', fontsize=15)
    ax.arrow(0, 0, 1, -2.5, head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(0, 0, 1.5, -2, head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-5,4,-4,3])

def fig28():
    from matplotlib.patches import Polygon
    fig = plt.figure(figsize=(8,6), num='Figure 28')
    d0 = np.matrix([[2], [0]])
    d1 = np.matrix([[2], [1]])
    d2 = np.matrix([[-1], [1.25]])
    u0 = np.matrix([[2], [0]])
    u1 = np.matrix([[1], [1]])
    u2 = np.matrix([[-3], [1.25]])
    e2 = np.matrix([[-0.75], [0.75]])
    r2 = np.matrix([[0], [1.75]])
    ax = fig.add_subplot(1, 1, 1)
    ax.add_patch(Polygon(np.array([[-5,-3], [-3,-1], [4,-1], [2,-3]]),fc='#999999', ec='k'))
    ax.add_patch(Polygon(np.array([[-4.5,-0.5], [-4,0], [-0.5,0], [-1.,-0.5]]),fc='None', ec='k'))
    ax.arrow(-4.25, -2.75, d0[0,0], d0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-4.25, -2.75, d1[0,0], d1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-0.25, -1.5, d2[0,0], d2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(-2.2, -2.75, r'$d_{[0]}$', fontsize=15)
    ax.text(-3.25, -2., r'$d_{[1]}$', fontsize=15)
    ax.text(-0.8, -0.75, r'$d_{[2]}$', fontsize=15)
    ax.arrow(-1.25, -2.75, u0[0,0], u0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-1.25, -2.75, u1[0,0], u1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-1, -1.5, u2[0,0], u2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(.9, -2.75, r'$u_{[0]}$', fontsize=15)
    ax.text(-1.25, -2.25, r'$u_{[1]}$', fontsize=15)
    ax.text(-3.5, -.75, r'$u_{[2]}$', fontsize=15)
    ax.arrow(1.5, -1.5, e2[0,0], e2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(0.7, -0.7, r'$e_{[2]}$', fontsize=15)
    ax.arrow(2, -1.5, r2[0,0], r2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(2.1, -.5, r'$r_{[2]}$', fontsize=15)
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-5,4,-4,1])

def fig29():
    from matplotlib.patches import Polygon
    fig = plt.figure(figsize=(8,6), num='Figure 29')
    d0 = np.matrix([[2], [0]])
    d1 = np.matrix([[2], [1]])
    d2 = np.matrix([[1.2], [1.8]])
    r0 = np.matrix([[2], [0]])
    r1 = np.matrix([[1], [1]])
    r2 = np.matrix([[0], [1.2]])
    e2 = np.matrix([[0.75], [1.5]])
    ax = fig.add_subplot(1, 1, 1)
    ax.add_patch(Polygon(np.array([[-5,-3], [-3,-1], [4,-1], [2,-3]]),fc='#999999', ec='k'))
    ax.add_patch(Polygon(np.array([[-3.5,-0.5], [-2.5,0.5], [-.5,0.5], [-1.5,-0.5]]),fc='None', ec='k'))
    ax.arrow(-4.25, -2.75, d0[0,0], d0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-4.25, -2.75, d1[0,0], d1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-2.25, -1.5, d2[0,0], d2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(-2.2, -2.75, r'$d_{[0]}$', fontsize=15)
    ax.text(-3.25, -2., r'$d_{[1]}$', fontsize=15)
    ax.text(-1.65, -0.75, r'$d_{[2]}$', fontsize=15)
    ax.arrow(-1.25, -2.75, r0[0,0], r0[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-1.25, -2.75, r1[0,0], r1[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.arrow(-3, -1.5, r2[0,0], r2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(.9, -2.75, r'$r_{[0]}$', fontsize=15)
    ax.text(-1.25, -2.25, r'$r_{[1]}$', fontsize=15)
    ax.text(-2.9, -.75, r'$r_{[2]}$', fontsize=15)
    ax.arrow(2, -1.5, e2[0,0], e2[1,0], head_width = .12, head_length = .2, length_includes_head = True, color='k')
    ax.text(2.7, -.5, r'$e_{[2]}$', fontsize=15)
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticklabels('')
    plt.axis([-5,4,-4,1])

def fig30(A, b, c):
    x = np.matrix([[-2.0],[-2.0]])
    steps = CG(A, b, x)
    fig = plt.figure(figsize=(8,8), num='Figure 30')
    plotcontours(A, b, c, fig)
    plt.plot(steps[0,:], steps[1,:], '-mo')
    plt.text(-2.5, -2.0, r'$x_{[0]}$', fontsize=16)
    plt.text(2.1, -2.0, r'$x$', fontsize=16)
    
def fig_C():
    A_00 = 3.; A_01 = 2.; A_10 = 2.; A_11 = 6.
    b_0 = 2.; b_1 = -8.
    c_0 = 0.  
    x_0 = -2.; x_1 =-2.

    fig = plt.figure(figsize=(8,8), num='Figure C')
    ax1 = fig.add_subplot(1, 1, 1)

    # create 3D points
    size = 30
    x1 = np.linspace(-10, 10, size)
    x2 = np.linspace(-10, 10, size)
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), c_0)

    # plot contours
    ax1.contour(x1, x2, zs, 20)

    steps = CG(np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), np.matrix([[x_0],[x_1]]))
    ax1.plot(steps[0,:], steps[1,:], '-o', color='m')
    return ax1

def sliders_figC(ax1):
    import matplotlib
    size = 30
    x1 = np.linspace(-10, 10, size)
    x2 = np.linspace(-10, 10, size)
    x1, x2 = np.meshgrid(x1, x2)

    def update_plots(A_00, A_01, A_10, A_11, b_0, b_1, x_0, x_1):
        # update 3D points
        zs = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = np.matrix([[x1[i,j]], [x2[i,j]]])
                zs[i,j] = f(x, np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), 0)
                
        # update contours
        [h.remove() for h in ax1.get_children() if isinstance(h, matplotlib.collections.LineCollection)]
        ax1.contour(x1, x2, zs, 20)
        
        # update steps
        [h.remove() for h in ax1.get_children() if isinstance(h, matplotlib.lines.Line2D)]
        steps = CG(np.matrix([[A_00, A_01],[A_10, A_11]]), np.matrix([[b_0],[b_1]]), np.matrix([[x_0],[x_1]]))
        ax1.plot(steps[0,:], steps[1,:], '-o', color='m')
        
        plt.axis([-10,10,-10,10])

        
    # define slider layout
    v_layout = ipw.Layout(display='flex',
                        flex_flow='column',
                        align_items='stretch',
                        justify_content = 'center',
                        width='100%')

    h_layout = ipw.Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        justify_content = 'center',
                        width='100%')
    
    A_00 = ipw.FloatSlider(3.0, min=-10., max=10., width='auto')
    A_01 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    A_10 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    A_11 = ipw.FloatSlider(6.0, min=-10., max=10., width='auto')
    b_0 = ipw.FloatSlider(2.0, min=-10., max=10., width='auto')
    b_1 = ipw.FloatSlider(-8.0, min=-10., max=10., width='auto')
    x_0 = ipw.FloatSlider(-2.0, min=-10., max=10., width='auto')
    x_1 = ipw.FloatSlider(-2.0, min=-10., max=10., width='auto')

    form = ipw.Box([ipw.Box([ipw.Box([A_00, A_10], layout=v_layout), 
                             ipw.Box([A_01, A_11], layout=v_layout)], layout=h_layout),
                    ipw.Box([ipw.Box([b_0, x_0], layout=v_layout), 
                             ipw.Box([b_1, x_1], layout=v_layout)], layout=h_layout)], 
                    layout = v_layout)
    display(form)
    ipw.interactive(update_plots, A_00=A_00, A_01=A_01, A_10=A_10, A_11=A_11, 
                    b_0=b_0, b_1=b_1, x_0=x_0, x_1=x_1)
    
def fig31():
    fig = plt.figure(figsize=(8,8), num='Figure 31')
    ax = fig.add_subplot(2,2,1)
    ax.set_title('(a)')
    ax.plot([0, 10], [1, 1])
    ax.plot(2,1,'k.')
    ax.plot(7,1,'k.')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([2,7])
    ax.text(10, -0.1, r'$\lambda$', fontsize=16)
    ax.text(-1, 1.1, r'$P_0(\lambda)$', fontsize=16)
    plt.axis([0,10,-1,1])

    ax = fig.add_subplot(2,2,2)
    ax.set_title('(b)')
    ax.plot([0, 9], [1, -1])
    ax.plot(2,5./9,'k.')
    ax.plot(7,-5./9,'k.')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([2,7])
    ax.text(10, -0.1, r'$\lambda$', fontsize=16)
    ax.text(-1, 1.1, r'$P_1(\lambda)$', fontsize=16)
    plt.axis([0,10,-1,1])

    ax = fig.add_subplot(2,2,3)
    ax.set_title('(c)')
    x = np.linspace(0,9,20)
    a = 1/(4.5**2 - 2.5**2)
    b = -a*2.5**2
    y = a*(x-4.5)**2+b
    ax.plot(x, y)
    ax.plot(2,0,'k.')
    ax.plot(7,0,'k.')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([2,7])
    ax.text(10, -0.1, r'$\lambda$', fontsize=16)
    ax.text(-1, 1.1, r'$P_2(\lambda)$', fontsize=16)
    plt.axis([0,10,-1,1])

    ax = fig.add_subplot(2,2,4)
    ax.set_title('(d)')
    ax.plot(x, y)
    ax.plot(1.9,0.05,'k.')
    ax.plot(2,0,'k.')
    ax.plot(2.15,-0.05,'k.')
    ax.plot(6.85,-0.05,'k.')
    ax.plot(7,0,'k.')
    ax.plot(7.1,0.05,'k.')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([2,7])
    ax.text(10, -0.1, r'$\lambda$', fontsize=16)
    ax.text(-1, 1.1, r'$P_2(\lambda)$', fontsize=16)
    plt.axis([0,10,-1,1])

def fig32():
    from matplotlib.patches import Rectangle
    from numpy.polynomial.chebyshev import chebval
    w = np.linspace(-1.3, 1.3, 105)
    fig = plt.figure(figsize=(8,8), num='Figure 32')
    ax = fig.add_subplot(2,2,1)
    ax.plot(w, chebval(w,np.append(np.zeros(2),1)))
    ax.add_patch(Rectangle((-1,-1), 2, 2, fc='None', ec='k'))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(np.linspace(-1.3,1.3,27), minor=True)
    ax.text(0, 2.1, r'$T_2(\omega)$')
    ax.text(1.3, -0.1, r'$\omega$', fontsize=16)
    plt.axis([-1.3,1.3,-2,2])

    ax = fig.add_subplot(2,2,2)
    ax.plot(w, chebval(w,np.append(np.zeros(5),1)))
    ax.add_patch(Rectangle((-1,-1), 2, 2, fc='None', ec='k'))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(np.linspace(-1.3,1.3,27), minor=True)
    ax.text(0, 2.1, r'$T_5(\omega)$')
    ax.text(1.3, -0.1, r'$\omega$', fontsize=16)
    plt.axis([-1.3,1.3,-2,2])

    w = np.linspace(-1.3, 1.3, 209)
    ax = fig.add_subplot(2,2,3)
    ax.plot(w, chebval(w,np.append(np.zeros(10),1)))
    ax.add_patch(Rectangle((-1,-1), 2, 2, fc='None', ec='k'))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(np.linspace(-1.3,1.3,27), minor=True)
    ax.text(0, 2.1, r'$T_{10}(\omega)$')
    ax.text(1.3, -0.1, r'$\omega$', fontsize=16)
    plt.axis([-1.3,1.3,-2,2])

    w = np.linspace(-1.3, 1.3, 1665)
    ax = fig.add_subplot(2,2,4)
    ax.plot(w, chebval(w,np.append(np.zeros(49),1)))
    ax.add_patch(Rectangle((-1,-1), 2, 2, fc='None', ec='k'))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(np.linspace(-1.3,1.3,27), minor=True)
    ax.text(0, 2.1, r'$T_{49}(\omega)$')
    ax.text(1.3, -0.1, r'$\omega$', fontsize=16)
    plt.axis([-1.3,1.3,-2,2])

def fig33():
    from matplotlib.patches import Rectangle
    from numpy.polynomial.chebyshev import chebval
    l = np.linspace(0, 9, 40)
    fig = plt.figure(figsize=(8,8), num='Figure 33')
    ax = fig.add_subplot(1,1,1)
    ax.plot(l, chebval(9./5-2.*l/5,np.append(np.zeros(2),1))/chebval(9./5,np.append(np.zeros(2),1)))
    ax.add_patch(Rectangle((2,-.183), 5, 2*.183, fc='None', ec='k'))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.linspace(-1,1,41), minor=True)
    ax.text(0, 1.1, r'$P_2(\lambda)$')
    ax.text(8.3, -0.1, r'$\lambda$', fontsize=16)
    plt.axis([0,8,-1,1])

def fig34():
    plt.figure(figsize=(8,8), num='Figure 34')
    size = 99
    kappa = np.linspace(1, 100, size)
    omega = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))
    plt.plot(kappa, omega)
    plt.minorticks_on()
    plt.grid(which='both')
    plt.xlabel(r'$\kappa$', fontsize=20)
    plt.ylabel(r'$\omega$', fontsize=20)
    plt.axis([0,100,0,1])

def fig35():
    plt.figure(figsize=(8,8), num='Figure 35')
    size = 99
    kappa = np.linspace(1, 1000, size)
    omega = np.sqrt(kappa)
    plt.plot(kappa, omega)
    plt.minorticks_on()
    plt.grid(which='major')
    plt.xlabel(r'$\kappa$', fontsize=20)
    plt.ylabel(r'$\i$', fontsize=20)
    plt.axis([0,1000,0,40])

def fig36():
    fig = plt.figure(figsize=(8,8), num='Figure 36')
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])
    c = 0.0
    Minv = np.matrix([[1/3.0, 0.0], [0.0, 1/6.0]])
    fig.add_subplot(1, 1, 1)
    plotcontours(Minv*A,Minv*b,c, fig, (-4,6,-9,1,20))
    plt.axis([-4,6,-9,1])

def figload(num, figsize):
    from PIL import Image
    plt.figure(figsize=figsize, num='Figure %2d'%num)
    image = Image.open('fig%2d.png'%num)
    plt.imshow(np.asarray(image))
    plt.axis('off')



