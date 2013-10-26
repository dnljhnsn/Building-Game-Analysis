import numpy as np
import numpy.linalg
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import polyhedra
from math import floor,cos,acos,sin
import scipy.misc
import os.path
import cPickle
import numpy.linalg

import matplotlib.colors as colors
import matplotlib.cm as cm


def plot_polyhedron_f(v,f_inds,inter=np.array([])):
    if len(inter) == 0:
        inter = np.ones((len(f_inds),1))
    ax = Axes3D(plt.figure())
    scale = np.abs(v).max()*1.2
    ax.set_xlim(-scale,scale)
    ax.set_ylim(-scale,scale)
    ax.set_zlim(-scale,scale)
    for i in range(len(f_inds)):
        if inter[i] != 0:
            side = []
            for j in range(len(f_inds[i])):
                #side1 = [[0.,0.,0.],[0.,0.,1.],[0.,1.,1.],[0.,1.,0.]]
                side.append([v[f_inds[i][j],0],v[f_inds[i][j],1],v[f_inds[i][j],2]])

            tri = Poly3DCollection([side])
            color = colors.rgb2hex(sp.rand(3))
            tri.set_facecolor(color)
            tri.set_edgecolor('k')

            ax.add_collection3d(tri)

    plt.show()

def plot_polyhedra(vs,fis):
    scale = 0.0
    ax = Axes3D(plt.figure())
    for k in range(len(vs)):
        v = vs[k]
        f_inds = fis[k]
        #if k == 4:
        color = colors.rgb2hex(sp.rand(3))
        #else:
        #    color = [1,1,1]
        #if len(inter) == 0:
        inter = np.ones((len(f_inds),1))
        scale = max(scale,np.abs(v).max()*1.2)
        for i in range(len(f_inds)):
            if inter[i] != 0:
                side = []
                for j in range(len(f_inds[i])):
                    #side1 = [[0.,0.,0.],[0.,0.,1.],[0.,1.,1.],[0.,1.,0.]]
                    #print k,i,j,f_inds
                    #print f_inds[i][j],f_inds[i][j],f_inds[i][j]
                    #print v
                    #print v[f_inds[i][j],0]
                    #print v[f_inds[i][j],1]
                    #print v[f_inds[i][j],2]
                    side.append([v[f_inds[i][j],0],v[f_inds[i][j],1],v[f_inds[i][j],2]])

                tri = Poly3DCollection([side])
                tri.set_facecolor(color)
                tri.set_edgecolor('k')

                ax.add_collection3d(tri)
    ax.set_xlim(-scale,scale)
    ax.set_ylim(-scale,scale)
    ax.set_zlim(-scale,scale)
        
    plt.show()

def plot_polyhedron_g(v,f_inds,inter=np.array([])):
    if len(inter) == 0:
        inter = np.ones((len(f_inds),1))
    ax = Axes3D(plt.figure())
    scale = np.abs(v).max()*1.2
    #print 'scale',scale
    ax.set_xlim(-scale,scale)
    ax.set_ylim(-scale,scale)
    ax.set_zlim(-scale,scale)
    for i in range(len(f_inds)):
        if inter[i] != 0:
            side = []
            for j in range(len(f_inds[i])):
                #side1 = [[0.,0.,0.],[0.,0.,1.],[0.,1.,1.],[0.,1.,0.]]
                side.append([v[f_inds[i][j],0],v[f_inds[i][j],1],v[f_inds[i][j],2]])

            tri = Poly3DCollection([side])
            color = colors.rgb2hex(sp.rand(3))
            tri.set_facecolor(color)
            tri.set_edgecolor('k')

            ax.add_collection3d(tri)

    plt.show()

def plot_polyhedron(v,f_inds,inter=np.array([]),vtype='f'):
    if vtype == 'g':
        plot_polyhedron_g(v,f_inds,inter)
    else:
        plot_polyhedron_f(v,f_inds,inter)
    return

def plot_DoF_dist(dfs,ints,shape_name):
    M = max(dfs)
    N = len(ints[0,:])
    w = 0.8

    inds = np.arange(N+1)
    df_dist = np.zeros((M+1,N+1))

    for k,d in enumerate(dfs):
        t = sum([1 for x in ints[k,:] if x != 0])
        df_dist[d,t] += 1.0
    
    for j in range(N+1):
        df_dist[:,j] /= sum(df_dist[:,j])
 
    fig = plt.figure(202)
    plt.clf()
    ax = fig.add_subplot(111)
    
    rects_total = np.zeros((N+1))
    for i in range(M+1):
        colour = colors.rgb2hex(sp.rand(3))
        rects = ax.bar(inds,df_dist[i,:],width=w,color=colour,bottom=rects_total)
        rects_total += df_dist[i,:]

    ax.set_xlabel('Number of faces of intermediate')
    ax.set_ylabel('Distribution of degrees of freedom')
    ax.set_title('Degrees of freedom distribution by size of intermediate for the '+shape_name)
    ax.set_xticks(inds+w*.5)
    ax.set_xticklabels(inds)
    ax.set_xlim(-0.5*w,N+1.5*w)
    ax.set_ylim(0.0,1.0)
    
    plt.show()


def plot_DoF_dist2(dfs,ints,shape_name):
    M = max(dfs)
    N = len(ints[0,:])
    w = 0.8

    inds = np.arange(N+1)
    df_dist = np.zeros((M+1,N+1))

    for k,d in enumerate(dfs):
        t = sum([1 for x in ints[k,:] if x != 0])
        df_dist[d,t] += 1.0
    
    for j in range(N+1):
        df_dist[:,j] /= sum(df_dist[:,j])
 
    fig = plt.figure(202)
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(M+1):
        colour = colors.rgb2hex(sp.rand(3))
        rects = ax.plot(inds,df_dist[i,:],color=colour)
       
    ax.set_xlabel('Number of faces of intermediate')
    ax.set_ylabel('Proportion of intermediates')
    ax.set_title('Degrees of freedom distribution by size of intermediate for the '+shape_name)
    ax.set_xticks(inds)
    ax.set_xticklabels(inds)
    ax.legend()
    #ax.set_xlim(-0.5*w,N+1.5*w)
    #ax.set_ylim(0.0,1.0)
    
    plt.show()

def plot_DoF_dist3(dfs,ints,shape_name):
    M = max(dfs)
    N = len(ints[0,:])
    w = 0.8

    inds = np.arange(N+1)
    df_dist = np.zeros((M+1,N+1))

    for k,d in enumerate(dfs):
        t = sum([1 for x in ints[k,:] if x != 0])
        df_dist[d,t] += 1.0
    
    for j in range(N+1):
        df_dist[:,j] /= sum(df_dist[:,j])
 
    fig = plt.figure(203)
    plt.clf()
    ax = fig.add_subplot(111)

    grd = ax.imshow(df_dist, cmap=cm.gist_earth, interpolation='nearest')
    plt.colorbar(grd)
    ax.set_xlabel('Number of faces of intermediate')
    ax.set_ylabel('Proportion of intermediates with each numnber of degrees of freedom')
    ax.set_title('Degrees of freedom distribution by size of intermediate for the '+shape_name)
    ax.set_xticks(inds)
    ax.set_xticklabels(inds)
    #ax.set_xlim(-0.5*w,N+1.5*w)
    #ax.set_ylim(0.0,1.0)
    
    plt.show()



def plot_DoF_hist(dfs,ints,shape_name,faces=None):
    M = max(dfs)
    w = 0.8

    inds = np.arange(M+1)
    df_hist = np.zeros((M+1))
    
    if faces == None:
        for d in dfs:
            df_hist[d] += 1
    else:
        for k,d in enumerate(dfs):
            if sum([1 for x in ints[k,:] if x != 0]) == faces:
                df_hist[d] += 1

    fig = plt.figure(201)
    plt.clf()
    ax = fig.add_subplot(111)
    rects = ax.bar(inds,df_hist,width=w)
    ax.set_xlabel('Degrees of freedom')
    ax.set_ylabel('Number of intermediates')
    if faces == None:
        ax.set_title('Degrees of freedom histogram for '+shape_name+' intermediates')
    else:
        ax.set_title('Degrees of freedom histogram for '+str(faces)+' faced '+shape_name+' intermediates')
    ax.set_xticks(inds+w*.5)
    ax.set_xticklabels(inds)
    ax.set_xlim(-0.5*w,M+1.5*w)
    ax.set_ylim(0.0,1.15*max(df_hist))
    
    plt.show()

def plot_Nk_scaling(Nk):
    N = float(sum(Nk))
    T = float(len(Nk)-1)
    fig = plt.figure(301)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(T+1)/T,Nk/scipy.misc.comb(T*np.ones((T+1)),np.arange(T+1)))
    ax.set_ylabel('Nk/TCk')
    ax.set_ylabel('k/T')
    ax.set_title('Scaling of the number of intermediates Nk with fixed face count k')
    #ax.set_xticks(inds+w*.5)
    #ax.set_xticklabels(inds)
    #ax.set_xlim(-0.5*w,M+1.5*w)
    #ax.set_ylim(0.0,1.15*max(df_hist))
    
    plt.show()


def organize_by_num_faces(ints):
    T = len(ints[0])
    inds = [[] for x in range(T+1)]
    for k,x in enumerate(ints):
        ord = sum([1 for f in x if f != 0])
        inds[ord].append(k)
    Nk = np.array([float(len(w)) for w in inds])
    return Nk, inds

def bg_config_space(shape_name):
    version = "4_1_"
    f = open('bg_'+version+shape_name+'_arc.txt', 'r')
    vert_line = f.readline().split()
    V = int(vert_line[-1])
    ints = []
    paths = []
    shell_int = []
    shell_paths = []
    for k in range(V):
        line = [int(x) for x in f.readline().split()]
        if k != line[0]:
            print "OH DEAR."
        ints.append(line[1:-3])
        paths.append(line[-3])
        shell_int.append(line[-2])
        shell_paths.append(line[-1])
    ints = np.array(ints)
    paths = np.array(paths)
    shell_int = np.array(shell_int)
    shell_paths = np.array(shell_paths)
    edge_line = f.readline().split()
    E = int(edge_line[-1])
    edges = []
    degens = []
    open_edges = []
    shell_edge = []
    for j in range(E):
        line = [int(x) for x in f.readline().split()]
        edges.append(line[0:2])
        degens.append(line[2])
        open_edges.append(line[3])
        shell_edge.append(line[4])
    edges = np.array(edges)
    degens = np.array(degens)
    open_edges = np.array(open_edges)
    shell_edge = np.array(shell_edge)
    return ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge

def hasedge(a,b,f_inds):
    #a,b are faces
    count = 0
    for v in f_inds[a]: 
        if v in f_inds[b]:
            count += 1
    if count == 2:
        return True
    return False

def first_face(inter):
    for k,x in enumerate(inter):
        if x != 0:
            return k
    return -1

def v2w(v,f_inds):
    g_inds = []
    g_i = 0
    for q in range(len(f_inds)):
        g_inds.append(range(g_i,g_i+len(f_inds[q])))
        g_i += len(f_inds[q])
    w = np.zeros((g_i,3))
    for r in range(len(f_inds)):
        for s in range(len(f_inds[r])):
            w[g_inds[r][s],:] = v[f_inds[r][s],:]
    return w,g_inds   

def DoF(inter,v,f_inds,c):
    g_inds = []
    #g_inv_inds = []
    g_i = 0
    for q in range(len(inter)):
        g_inds.append(range(g_i,g_i+len(f_inds[q])))
        g_i += len(f_inds[q])
    #print g_inds, g_i

    if sum(inter) == 0:
        return 0, np.array([])
                       
    # Base Constraints
    N = 3*(g_i)
    #N = 3*v.shape[0]
    base = first_face(inter)
    if base == -1:
        return 0,np.array([])
    J = np.zeros((3*len(f_inds[base]),N))##########g_inds?\/too
    for i in range(len(f_inds[base])):
        J[3*i+0,3*g_inds[base][i]+0] = 1.0
        J[3*i+1,3*g_inds[base][i]+1] = 1.0
        J[3*i+2,3*g_inds[base][i]+2] = 1.0
    # Length Constraints
    for j in range(len(f_inds)):
        if inter[j] != 0:
            # Same Vertex Constraints
            for w in range(j):
                if inter[w] != 0 and hasedge(j,w,f_inds): ##should give same results as before. until this line used
                #if inter[w] != 0:
                    for r in range(len(f_inds[j])):
                        if f_inds[j][r] in f_inds[w]:
                            #w_ind = np.where(f_inds[j][r] == f_inds[w])
                            w_ind = f_inds[w].index(f_inds[j][r])
                            new_row = np.zeros((3,N))
                            new_row[0,3*g_inds[j][r]+0] = 1.0
                            new_row[0,3*g_inds[w][w_ind]+0] = -1.0                          
                            new_row[1,3*g_inds[j][r]+1] = 1.0
                            new_row[1,3*g_inds[w][w_ind]+1] = -1.0                          
                            new_row[2,3*g_inds[j][r]+2] = 1.0
                            new_row[2,3*g_inds[w][w_ind]+2] = -1.0                          
                            J = np.vstack((J,new_row))
                            #print 'SVC'
            # Edge Length Constraints
            for k in range(len(f_inds[j])):
                new_row = np.zeros((1,N))
                new_row[0,3*g_inds[j][k-0]+0] = 2.0*(v[f_inds[j][k-0],0] - v[f_inds[j][k-1],0])                          
                new_row[0,3*g_inds[j][k-1]+0] = 2.0*(v[f_inds[j][k-1],0] - v[f_inds[j][k-0],0])                          
                new_row[0,3*g_inds[j][k-0]+1] = 2.0*(v[f_inds[j][k-0],1] - v[f_inds[j][k-1],1])                          
                new_row[0,3*g_inds[j][k-1]+1] = 2.0*(v[f_inds[j][k-1],1] - v[f_inds[j][k-0],1])                          
                new_row[0,3*g_inds[j][k-0]+2] = 2.0*(v[f_inds[j][k-0],2] - v[f_inds[j][k-1],2])                          
                new_row[0,3*g_inds[j][k-1]+2] = 2.0*(v[f_inds[j][k-1],2] - v[f_inds[j][k-0],2])                          
                J = np.vstack((J,new_row))
            # Face Angle Constraints
            """for k in range(-2,len(f_inds[j,:])-2):
                new_row = np.zeros((1,N))
                new_row[0,3*f_inds[j,k+0]+0] = v[f_inds[j,k+2],0] - v[f_inds[j,k+1],0]                          
                new_row[0,3*f_inds[j,k+1]+0] = 2.0*v[f_inds[j,k+1],0] - v[f_inds[j,k+0],0] - v[f_inds[j,k+2],0] 
                new_row[0,3*f_inds[j,k+2]+0] = v[f_inds[j,k+0],0] - v[f_inds[j,k+1],0]                              
                new_row[0,3*f_inds[j,k+0]+1] = v[f_inds[j,k+2],1] - v[f_inds[j,k+1],1]                          
                new_row[0,3*f_inds[j,k+1]+1] = 2.0*v[f_inds[j,k+1],1] - v[f_inds[j,k+0],1] - v[f_inds[j,k+2],1] 
                new_row[0,3*f_inds[j,k+2]+1] = v[f_inds[j,k+0],1] - v[f_inds[j,k+1],1]                          
                new_row[0,3*f_inds[j,k+0]+2] = v[f_inds[j,k+2],2] - v[f_inds[j,k+1],2]                          
                new_row[0,3*f_inds[j,k+1]+2] = 2.0*v[f_inds[j,k+1],2] - v[f_inds[j,k+0],2] - v[f_inds[j,k+2],2] 
                new_row[0,3*f_inds[j,k+2]+2] = v[f_inds[j,k+0],2] - v[f_inds[j,k+1],2]                          
                J = np.vstack((J,new_row))"""
            new_row = np.zeros((1,N))
            new_row[0,3*g_inds[j][0]+0] = v[f_inds[j][2],0] - v[f_inds[j][1],0]                          
            new_row[0,3*g_inds[j][1]+0] = 2.0*v[f_inds[j][1],0] - v[f_inds[j][0],0] - v[f_inds[j][2],0] 
            new_row[0,3*g_inds[j][2]+0] = v[f_inds[j][0],0] - v[f_inds[j][1],0]                              
            new_row[0,3*g_inds[j][0]+1] = v[f_inds[j][2],1] - v[f_inds[j][1],1]                          
            new_row[0,3*g_inds[j][1]+1] = 2.0*v[f_inds[j][1],1] - v[f_inds[j][0],1] - v[f_inds[j][2],1] 
            new_row[0,3*g_inds[j][2]+1] = v[f_inds[j][0],1] - v[f_inds[j][1],1]                          
            new_row[0,3*g_inds[j][0]+2] = v[f_inds[j][2],2] - v[f_inds[j][1],2]                          
            new_row[0,3*g_inds[j][1]+2] = 2.0*v[f_inds[j][1],2] - v[f_inds[j][0],2] - v[f_inds[j][2],2] 
            new_row[0,3*g_inds[j][2]+2] = v[f_inds[j][0],2] - v[f_inds[j][1],2]                          
            J = np.vstack((J,new_row))
            for k in range(3,len(f_inds[j])):
                J = np.vstack((J,face_constraints(v,f_inds[j][0],f_inds[j][1],f_inds[j][2],f_inds[j][k],g_inds[j][0],g_inds[j][1],g_inds[j][2],g_inds[j][k],N)))
                
    free_vars = 0
    for k in range(v.shape[0]):
        af = 0
        for i in range(len(f_inds)):
            if k in f_inds[i] and inter[i] != 0:
                af += 1
                break
        if af == 0:
            free_vars += 3
    #print J

    rank1 = numpy.linalg.matrix_rank(J)
    #u, s, v = np.linalg.svd(J)
    #print u
    #rank2 = np.sum(s > 1e-4)
    #rank2 = -1
    #if rank1 != rank2:
    #     print "oh dear. the ranks is fucked."
    #print 'J size',J.shape,'free variables',free_vars,'rank',rank1,rank2
    #print J
    rank = rank1
    #return N - rank - free_vars, J
    fv = sum(np.array([len(g_inds[i]) for i in range(len(inter)) if inter[i] == 0]))
    #print "returns",N,rank,fv
    return N - rank - 3*fv, J

def R_partial(n,dn,theta):
    Rp = np.zeros((3,3))
    
    Rp[0,0] = 2.0*n[0]*dn[0]*(1.0-cos(theta))
    Rp[1,0] = (1.0-cos(theta))*(n[1]*dn[0] + dn[1]*n[0]) + dn[2]*sin(theta)
    Rp[2,0] = (1.0-cos(theta))*(n[2]*dn[0] + dn[2]*n[0]) - dn[1]*sin(theta)
    
    Rp[0,1] = (1.0-cos(theta))*(n[0]*dn[1] + dn[0]*n[1]) - dn[2]*sin(theta)
    Rp[1,1] = 2.0*n[1]*dn[1]*(1.0-cos(theta))
    Rp[2,1] = (1.0-cos(theta))*(n[2]*dn[1] + dn[2]*n[1]) + dn[0]*sin(theta)
    
    Rp[0,2] = (1.0-cos(theta))*(n[0]*dn[2] + dn[0]*n[2]) + dn[1]*sin(theta)
    Rp[1,2] = (1.0-cos(theta))*(n[1]*dn[2] + dn[1]*n[2]) - dn[0]*sin(theta)
    Rp[2,2] = 2.0*n[2]*dn[2]*(1.0-cos(theta))
    
    return Rp
    


def face_constraints(v,i0,i1,i2,ik,g0,g1,g2,gk,N):
    #N = 3*v.shape[0]
    # Template vertices
    t0 = v[i0,:].T
    t1 = v[i1,:].T
    t2 = v[i2,:].T
    tk = v[ik,:].T
    # Face vertices
    v0 = v[i0,:].T
    v1 = v[i1,:].T
    v2 = v[i2,:].T
    vk = v[ik,:].T
    # Constants
    gamma = 1.0/numpy.linalg.norm(np.cross(v0-v1,v2-v1))
    tau = numpy.linalg.norm(tk-t1)/numpy.linalg.norm(t0-t1)
    theta = acos(np.dot(tk-t1,t0-t1)/(numpy.linalg.norm(tk-t1)*numpy.linalg.norm(t0-t1)))
    #theta = -acos(np.dot(tk-t1,t0-t1)/(numpy.linalg.norm(tk-t1)*numpy.linalg.norm(t0-t1)))
    # Normal
    n = np.cross(v0-v1,v2-v1)*gamma

    #print "cross calc", v0-v1,v2-v1,gamma
    # Rotation matrix
    """
    R = np.zeros((3,3))

    R[0,0] = cos(theta) + n[0]**2*(1-cos(theta))
    R[1,0] = n[1]*n[0]*(1-cos(theta)) + n[2]*sin(theta)
    R[2,0] = n[2]*n[0]*(1-cos(theta)) - n[1]*sin(theta)

    R[0,1] = n[0]*n[1]*(1-cos(theta)) - n[2]*sin(theta)
    R[1,1] = cos(theta) + n[1]**2*(1-cos(theta))
    R[2,1] = n[2]*n[1]*(1-cos(theta)) + n[0]*sin(theta)

    R[0,2] = n[0]*n[2]*(1-cos(theta)) + n[1]*sin(theta)
    R[1,2] = n[1]*n[2]*(1-cos(theta)) - n[0]*sin(theta)
    R[2,2] = cos(theta) + n[2]**2*(1-cos(theta))
    """
    R = rotation_matrix(n,theta)
    #rotate to see if got the right rotation
    #print "v0,v1,v2,vk",v0,v1,v2,vk
    #print "gamma, theta, tau",gamma,theta,tau
    #print "normal",n
    #vk_hat = v1 + np.dot(R.T,v0-v1)*tau
    vk_hat = v1 + np.dot(R,v0-v1)*tau
    #print "vks",vk,vk_hat

    # Normal Jacobian
    K = np.zeros((3,9))

    K[0,0] = gamma*0
    K[1,0] = gamma*(-v2[2]+v1[2])
    K[2,0] = gamma*(+v2[1]-v1[1])

    K[0,1] = gamma*(+v2[2]-v1[2])
    K[1,1] = gamma*0
    K[2,1] = gamma*(-v2[0]+v1[0])

    K[0,2] = gamma*(-v2[1]+v1[1])
    K[1,2] = gamma*(+v2[0]-v1[0])
    K[2,2] = gamma*0

    K[0,3] = gamma*0
    K[1,3] = gamma*(+v2[2]-v0[2])
    K[2,3] = gamma*(-v2[1]+v0[1])

    K[0,4] = gamma*(-v2[2]+v0[2])
    K[1,4] = gamma*0
    K[2,4] = gamma*(+v2[0]-v0[0])

    K[0,5] = gamma*(+v2[1]-v0[1])
    K[1,5] = gamma*(-v2[0]+v0[0])
    K[2,5] = gamma*0

    K[0,6] = gamma*0
    K[1,6] = gamma*(+v0[2]-v1[2])
    K[2,6] = gamma*(-v0[1]+v1[1])

    K[0,7] = gamma*(-v0[2]+v1[2])
    K[1,7] = gamma*0
    K[2,7] = gamma*(+v0[0]-v1[0])

    K[0,8] = gamma*(+v0[1]-v1[1])
    K[1,8] = gamma*(-v0[0]+v1[0])
    K[2,8] = gamma*0

    new_row = np.zeros((3,N))

    new_row[:,3*g0+0] = tau*(np.dot(R_partial(n,K[:,0],theta),v0-v1) + R[:,0])
    new_row[:,3*g0+1] = tau*(np.dot(R_partial(n,K[:,1],theta),v0-v1) + R[:,1])
    new_row[:,3*g0+2] = tau*(np.dot(R_partial(n,K[:,2],theta),v0-v1) + R[:,2])

    new_row[:,3*g1+0] = tau*(np.dot(R_partial(n,K[:,3],theta),v0-v1) - R[:,0]) + np.array([1.0,0,0]).T 
    new_row[:,3*g1+1] = tau*(np.dot(R_partial(n,K[:,4],theta),v0-v1) - R[:,1]) + np.array([0,1.0,0]).T 
    new_row[:,3*g1+2] = tau*(np.dot(R_partial(n,K[:,5],theta),v0-v1) - R[:,2]) + np.array([0,0,1.0]).T 

    new_row[:,3*g2+0] = tau*(np.dot(R_partial(n,K[:,6],theta),v0-v1))
    new_row[:,3*g2+1] = tau*(np.dot(R_partial(n,K[:,7],theta),v0-v1))
    new_row[:,3*g2+2] = tau*(np.dot(R_partial(n,K[:,8],theta),v0-v1))

    new_row[:,3*gk+0] = np.array([-1.0,0,0]).T
    new_row[:,3*gk+1] = np.array([0,-1.0,0]).T
    new_row[:,3*gk+2] = np.array([0,0,-1.0]).T

    return new_row

def rotation_matrix(n,theta):
    R = np.zeros((3,3))

    R[0,0] = cos(theta) + n[0]**2*(1-cos(theta))
    R[1,0] = n[1]*n[0]*(1-cos(theta)) + n[2]*sin(theta)
    R[2,0] = n[2]*n[0]*(1-cos(theta)) - n[1]*sin(theta)

    R[0,1] = n[0]*n[1]*(1-cos(theta)) - n[2]*sin(theta)
    R[1,1] = cos(theta) + n[1]**2*(1-cos(theta))
    R[2,1] = n[2]*n[1]*(1-cos(theta)) + n[0]*sin(theta)

    R[0,2] = n[0]*n[2]*(1-cos(theta)) + n[1]*sin(theta)
    R[1,2] = n[1]*n[2]*(1-cos(theta)) - n[0]*sin(theta)
    R[2,2] = cos(theta) + n[2]**2*(1-cos(theta))
    
    return R

def rotate_vertices(v,n,theta):
    return 0


def get_e_ind(f1,f2,e_inds):
    e = [min(f1,f2),max(f1,f2)]
    if not e in e_inds:
        print "oh dear. i could not locate the edge",e
        return -1
    return e_inds.index(e)

def get_dual_entries(f1,f2,dual):
    # gets the entries from dual of the 2 vertices that are shared by f1 and f2 
    d1 = []
    d2 = []
    for d in dual:
        if f1 in d and f2 in d:
            if d1 == []:
                d1 = d
            else:
                d2 = d
    if d2 == []:
        print "oh dear. cant'd find vetexes desired."
        return -1
    return d1,d2


def plot_folding_int(v,f_inds,adj_list,e_inds,dual,inter):
    # Take info on which edges are cut (inter) and plot partially folded intermediate
    w,g_inds = v2w(v,f_inds)
    w_new = 0.0*w
    for q in range(len(g_inds[0])):
        #print w[g_inds[0][q],:],v[f_inds[0][q],:]
        w_new[g_inds[0][q],:] = w[g_inds[0][q],:]
    add_list = [0]
    drawn_list = [0]
    e1 = np.array([1,0,0])
    R_list = [rotation_matrix(e1,0.0)]
    #print len(add_list)
    print inter
    #print e_inds
    while len(add_list) > 0:
        #print add_list
        f = add_list.pop()
        #if f in drawn_list:
        #    continue
        #used_list.append(f)
        ##for each adjacent face
        for a in adj_list[f]:
            if a in drawn_list:
                continue
            ###if there is no cut between the two faces
            e_i = get_e_ind(a,f,e_inds)
            if inter[e_i] == 0:
                #add new face to stack
                if not a in add_list:
                    add_list.append(a)
                #print 'f,a',f,a,drawn_list,R_list
                    
                R = R_list[drawn_list.index(f)]
                
                #####plot 2nd face from f's roation matrix
                for k in range(len(g_inds[a])):
                    w_new[g_inds[a][k],:] = w_new[g_inds[f][1],:] + np.dot(R,w[g_inds[a][k],:]-w[g_inds[f][1],:])

                ####if either of the two shared vertices are vertex connections
                d1,d2 = get_dual_entries(f,a,dual)
                # If no corresponding vertex connections: flaten angle out
                #plot_polyhedron(w_new,g_inds,vtype='g')
                if vertex_connection(inter,d1,e_inds) == 0 and vertex_connection(inter,d2,e_inds) == 0:
                    #####plot 2nd face at dihedral angle of pi (planar)
                    ind_pairs = get_common_vertices(f,a,w_new,g_inds)#####
                    if not len(ind_pairs) == 2:
                        print 'oh dear. faces dont seem adjacent.',f,a,ind_pairs
                    if min(ind_pairs[0][0],ind_pairs[1][0]) == 0:
                        if max(ind_pairs[0][0],ind_pairs[1][0]) == 1:
                            vkm1 = -1
                            vk = 0
                            vk1 = 1
                        else:
                            vkm1 = -2
                            vk = -1
                            vk1 = 0
                    else:
                        vkm1 = min(ind_pairs[0][0],ind_pairs[1][0]) - 1
                        vk = min(ind_pairs[0][0],ind_pairs[1][0])
                        vk1 = max(ind_pairs[0][0],ind_pairs[1][0])
                        
                        if min(ind_pairs[0][0],ind_pairs[1][0])+1 != max(ind_pairs[0][0],ind_pairs[1][0]):
                            print 'oh dear. indexes much?'
                    u = w_new[g_inds[f][vk1],:]-w_new[g_inds[f][vk],:]
                    if numpy.linalg.norm(u) == 0.0:
                        print 'Y U IS 0!'
                        print vkm1,vk,vk1
                        
                    u = u/numpy.linalg.norm(u)
                    thet2 = acos(-1.0) - get_dihedral_angle(f,a,w_new,g_inds)#####
                    R2 = rotation_matrix(u,thet2)
             
                    for k in range(len(g_inds[a])):
                        w_new[g_inds[a][k],:] = w_new[g_inds[f][vk],:] + np.dot(R2,w_new[g_inds[a][k],:]-w_new[g_inds[f][vk],:])#####
                    
                    # Add face a and its rotation matrix to lists
                    drawn_list.append(a)
                    R_list.append(np.dot(R2,R))
                else:
                    # Add face a and its rotation matrix to lists
                    drawn_list.append(a)
                    R_list.append(R)

                #print 'common inds',f,a,get_common_vertices(f,a,w_new,g_inds),get_common_vertices(f,a,w,g_inds) 
                #print ''
                #plot_polyhedron(w_new,g_inds,vtype='g')
                    
    #plot_polyhedron(w_new,g_inds,vtype='g')
    #print 'drawn',drawn_list
    #print w_new, g_inds[3]
    return w_new


#def plot_folding_int(v,f_inds,adj_list,e_inds,dual,inter):
#    # Take info on which edges are cut (inter) and plot partially folded intermediate
#    w,g_inds = v2w(v,f_inds)
#    w_new = 0.0*w
#    for q in range(len(g_inds[0])):
#        #print w[g_inds[0][q],:],v[f_inds[0][q],:]
#        w_new[g_inds[0][q],:] = w[g_inds[0][q],:]
#    add_list = [0]
#    used_list = []
#    drawn_list = [0]
#    #print len(add_list)
#    print inter
#    print e_inds
#    while len(add_list) > 0:
#        #print add_list
#        f = add_list.pop()
#        if f in used_list:
#            continue
#        used_list.append(f)
#        ##for each adjacent face
#        
#        #print adj_list, f, adj_list[f]
#
#        for a in adj_list[f]:
#            if a in used_list:
#                continue
#            if a in drawn_list:
#                continue
#            ###if there is no cut between the two faces
#            e_i = get_e_ind(a,f,e_inds)
#            if inter[e_i] == 0:
#                #add new face to stack
#                if not a in add_list:
#                    add_list.append(a)
#                drawn_list.append(a)
#                print 'f,a',f,a,used_list,add_list,drawn_list
#                    
#                ####if either of the two shared vertices are vertex connections
#                d1,d2 = get_dual_entries(f,a,dual)
#
#                n1 = np.cross(w[g_inds[f][0],:]-w[g_inds[f][1],:],w[g_inds[f][2],:]-w[g_inds[f][1],:])
#                n1 = n1/numpy.linalg.norm(n1)
#                n2 = np.cross(w_new[g_inds[f][0],:]-w_new[g_inds[f][1],:],w_new[g_inds[f][2],:]-w_new[g_inds[f][1],:])
#                n2 = n2/numpy.linalg.norm(n2)
#                theta = acos(np.dot(n1,n2))
#                n = np.cross(n1,n2)
#                if theta != 0.0:
#                    n = n/numpy.linalg.norm(n)
#                R = rotation_matrix(n,theta)
#                print 'n,theta',n,theta
#                #print R
#                """if vertex_connection(inter,d1,e_inds) != 0 or vertex_connection(inter,d1,e_inds) != 0:
#                    #####plot 2nd face at dihedral angle from embedding
#                   #ang = get_dihedral_angle(f,a,v,f_inds)
#                    for k in range(len(g_inds[a])):
#                        w_new[g_inds[a][k],:] = w_new[g_inds[f][1],:] + np.dot(R,w[g_inds[a][k],:]-w[g_inds[f][1],:])
#                else:
#                    #####plot 2nd face at dihedral angle of pi (planar)
#                    for k in range(len(g_inds[a])):
#                        w_new[g_inds[a][k],:] = w_new[g_inds[f][1],:] + np.dot(R,w[g_inds[a][k],:]-w[g_inds[f][1],:])
#                """
#                #####plot 2nd face at dihedral angle from embedding
#                for k in range(len(g_inds[a])):
#                    w_new[g_inds[a][k],:] = w_new[g_inds[f][1],:] + np.dot(R,w[g_inds[a][k],:]-w[g_inds[f][1],:])
#                
#                count = len(g_inds[a])*3
#                # Check if verticies aligned right
#                while get_common_vertices(f,a,w_new,g_inds) != get_common_vertices(f,a,w,g_inds) and count > 0:
#                    #print 'f,a,gci w, w_new',f,a,get_common_vertices(f,a,w,g_inds), get_common_vertices(f,a,w_new,g_inds)
#                    #print 'before',w_new
#                    w_new = rotate_face(a,w_new,g_inds)
#                    #print 'after',w_new
#                    count -= 1
#                    if count == len(g_inds[a])*3/2:
#                        w_new = flip_face(a,w_new,g_inds)
#                    print 'f,a,gci w, w_new',f,a,get_common_vertices(f,a,w,g_inds), get_common_vertices(f,a,w_new,g_inds), count
#                    
#                # If no corresponding vertex connections: flaten angle out
#                #plot_polyhedron(w_new,g_inds,vtype='g')
#                if vertex_connection(inter,d1,e_inds) == 0 and vertex_connection(inter,d2,e_inds) == 0:
#                    #####plot 2nd face at dihedral angle of pi (planar)
#                    ind_pairs = get_common_vertices(f,a,w_new,g_inds)#####
#                    if not len(ind_pairs) == 2:
#                        print 'oh dear. faces dont seem adjacent.',f,a,ind_pairs
#                    ## some shit bout -0
#                    #if f == 5 and a == 3:
#                    #    print ind_pairs
#                    if min(ind_pairs[0][0],ind_pairs[1][0]) == 0:
#                        if max(ind_pairs[0][0],ind_pairs[1][0]) == 1:
#                            vkm1 = -1
#                            vk = 0
#                            vk1 = 1
#                        else:
#                            vkm1 = -2
#                            vk = -1
#                            vk1 = 0
#                    else:
#                        vkm1 = min(ind_pairs[0][0],ind_pairs[1][0]) - 1
#                        vk = min(ind_pairs[0][0],ind_pairs[1][0])
#                        vk1 = max(ind_pairs[0][0],ind_pairs[1][0])
#                        
#                        if min(ind_pairs[0][0],ind_pairs[1][0])+1 != max(ind_pairs[0][0],ind_pairs[1][0]):
#                            print 'oh dear. indexes much?'
#                    #u = np.cross(w_new[g_inds[f][vk1],:]-w_new[g_inds[f][vk],:],w_new[g_inds[f][vkm1],:]-w_new[g_inds[f][vk],:])#####
#                    u = w_new[g_inds[f][vk1],:]-w_new[g_inds[f][vk],:]
#                    if numpy.linalg.norm(u) == 0.0:
#                        print 'Y U IS 0!'
#                        print vkm1,vk,vk1
#                        
#                    u = u/numpy.linalg.norm(u)
#                    thet2 = acos(-1.0) - get_dihedral_angle(f,a,w_new,g_inds)#####
#                    R2 = rotation_matrix(u,thet2)
#                    #print "thet2,u", thet2,u
#                    #print "vk-1,vk,vk+1",vkm1,vk,vk1
#                    for k in range(len(g_inds[a])):
#                        w_new[g_inds[a][k],:] = w_new[g_inds[f][vk],:] + np.dot(R2,w_new[g_inds[a][k],:]-w_new[g_inds[f][vk],:])#####
#
#
#
#                #plot_polyhedron(w_new,g_inds,vtype='g')
#
#                """
#
#                    if get_dihedral_angle(f,a,w_new,g_inds) != acos(-1.0):#####
#                        R3 = rotation_matrix(u,-2.0*thet2)
#                        for k in range(len(g_inds[a])):
#                            w_new[g_inds[a][k],:] = w_new[g_inds[f][vk],:] + np.dot(R3,w_new[g_inds[a][k],:]-w_new[g_inds[f][vk],:])#####               
#                    """
#    """print 'w_new:'
#    for q in range(len(g_inds[0])):
#        print w[g_inds[0][q],:],v[f_inds[0][q],:],w_new[g_inds[0][q],:]
#    """
#                    
#    plot_polyhedron(w_new,g_inds,vtype='g')
#    print 'drawn',drawn_list
#    print w_new, g_inds[3]
#    return w_new


def rotate_face(f,w,g_inds):
    temp_v = np.copy(w[g_inds[f][0],:])
    for k in range(len(g_inds[f])-1):
        temp1 = w[g_inds[f][k+1],:]
        w[g_inds[f][k],:] = temp1
    w[g_inds[f][-1],:] = temp_v
    return w


def flip_face(f,w,g_inds):
    L = len(g_inds[f])
    for k in range(L/2):
        temp_1 = np.copy(w[g_inds[f][k],:])
        temp_2 = np.copy(w[g_inds[f][L-k-1],:])
        w[g_inds[f][k],:] = temp_2
        w[g_inds[f][L-k-1],:] = temp_1
    return w


def get_common_vertices(f1,f2,w,g_inds):
    ind_pairs = []
    for j in range(len(g_inds[f1])):
        for k in range(len(g_inds[f2])):
            #if np.array_equal(w[g_inds[f1][j],:],w[g_inds[f2][k],:]):
            #    ind_pairs.append([j,k])
            if numpy.linalg.norm(w[g_inds[f1][j],:]-w[g_inds[f2][k],:]) < 10.0**-6.0:
                ind_pairs.append([j,k])
    return ind_pairs
                                        

def get_dihedral_angle(f1,f2,v,f_inds):
    n1 = np.cross(v[f_inds[f1][2]]-v[f_inds[f1][1]],v[f_inds[f1][0]]-v[f_inds[f1][1]])
    n2 = np.cross(v[f_inds[f2][2]]-v[f_inds[f2][1]],v[f_inds[f2][0]]-v[f_inds[f2][1]])
    n1 = n1/numpy.linalg.norm(n1)
    n2 = n2/numpy.linalg.norm(n2)
    theta = acos(np.dot(n1,n2))
    
    return acos(-1.0) - theta

def DoF_folding(inter,v,f_inds,c):
    # Take infor on cut edges (inter) and compute the degrees of freedom of the folding intermediate
    return 0

def folding_state_space():
    ### this method, except starting at the completed poly (no cuts) and work the other way
    # Compute folding state space from the corresponding poly's BG state space
    ##remove nodes that are not connected
    ##identify nets (ints with no vertex connections)
    ##compute DoF for all remaining (connected) ints
    ##(*)from each net look at connected ints with less cuts
    ###if there is no difference in DoF search deeper in graph (even less cuts) until each branch finds an int with 
    ###these ints are nodes in the folding SS
    ###repeat from (*) except with newly found folding ints
    return 0

def invert_int(i):
    #print i
    for k,f in enumerate(i):
        if f == 0:
            i[k] = 1
        else:
            i[k] = 0
    #print i
    return i
        


def find_nets(poly, poly_name,version='4_1_'):
    # Get BGy output for folding_#poly#
    if os.path.isfile("folding_"+poly_name+"_ints.pkl"):
        [ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge] = cPickle.load(open("folding_"+poly_name+"_ints.pkl",'rb'))
    else:
        ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge = bg_config_space("folding_"+poly_name)
        g = open("folding_"+poly_name+"_ints.pkl",'wb')
        cPickle.dump([ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge],g,-1)
        g.close()
    # Get BG input file for #poly#
    V,E,F,S,species,f_types,adj_list,dual = get_poly(poly_name)
    # Get e_inds from folding_#poly#_inds
    f = open('folding_'+poly_name+'_inds.txt', 'r')
    e_inds = []
    for k in range(E):
        line = [int(x) for x in f.readline().split()]
        e_inds.append(line)
    f.close()
    #print e_inds
    net_list = []
    pos_int_list = []
    nets = 0
    possible_ints = 0
    for k,i in enumerate(ints):
        i = invert_int(i)
        if connected_cut(i,e_inds,adj_list):
            possible_ints += 1
            pos_int_list.append(k)
            if vertex_connections(i,e_inds,dual) == 0: 
                #print k
                net_list.append(k)
                nets += 1
    print len(net_list), len(pos_int_list)
    print net_list[-20:]
    v,f_inds,c = poly() 
    #plot_folding_int(v,f_inds,adj_list,e_inds,dual,ints[3,:])#62
    #for n in pos_int_list:
    for n in net_list:
        w_new = plot_folding_int(v,f_inds,adj_list,e_inds,dual,ints[n,:])#62
    #w_new = plot_folding_int(v,f_inds,adj_list,e_inds,dual,ints[0,:])#62(cube)
    #print w_new
    #print nets, possible_ints
    return net_list,pos_int_list

def vertex_connection(i,d,e_inds):
    # i: edge_cut intermediate
    # d: list of adjacent faces comprising the vertex in question
    cuts = 0
    for k in range(len(d)):
        ind = e_inds.index([min(d[k],d[k-1]),max(d[k],d[k-1])])
        if i[ind] != 0:
            cuts += 1
    if cuts == 0:
       return 1
    return 0


def vertex_connections(i,e_inds,dual):
    vcs = 0
    # For each vertex
    for d in dual:
        vcs += vertex_connection(i,d,e_inds)
        """
        cuts = 0
        for k in range(len(d)):
            ind = e_inds.index([min(d[k],d[k-1]),max(d[k],d[k-1])])
            if i[ind] != 0:
                cuts += 1
        if cuts == 0:
           vcs += 1
        """
    return vcs

def connected_cut(i,e_list,adj_list):
    s = [[n] for n in range(len(adj_list))]
    for k,e in enumerate(i):
        if e == 0:
            # merge groups with either adjacent face
            for j,x in enumerate(s):
                if e_list[k][0] in x:
                    g0,s0 = j,x
                if e_list[k][1] in x:
                    g1,s1 = j,x
            if g0 != g1:
                temp = s[g0] + s[g1]
                s.remove(s0)
                s.remove(s1)
                s.append(temp)
    if len(s) == 1:
        return True
    return False

def get_poly(poly_name):
    # Return data from BG input file
    f = open(poly_name+'_4_1.txt', 'r')
    # shape name 
    line = [x for x in f.readline().split()]
    if line[0] != poly_name:
        print "oh dear.",line[0],"is not actually the same as",poly_name
    # V E F
    [F,E,V] = [int(x) for x in f.readline().split()]
    # face types
    line_S = [int(x) for x in f.readline().split()]
    S = line_S[0]
    species = line_S[1:]
    # Adjacency list
    f_types = []
    adj_list = []    
    for k in range(F):
        line = [int(x) for x in f.readline().split()]
        f_types.append(line[1:1+line[0]])
        adj_list.append(line[1+line[0]:])
    dual = []
    for j in range(V):
        line = [int(x) for x in f.readline().split()]
        dual.append(line[1:])
    #print "V,E,F",V,E,F
    #print "S,species",S,species
    #print "f_types",f_types
    #print "adj_list",adj_list
    #print "dual",dual
    return V,E,F,S,species,f_types,adj_list,dual

def DoF_analysis(poly,poly_name):

    v,f_inds,c = poly() 


    if poly_name in nonbg:
        inter = np.ones((len(f_inds)))
        df = DoF(inter,v,f_inds,c)[0]
        print df
        plot_polyhedron(v,f_inds)


    else:
        if os.path.isfile(poly_name+"_ints.pkl"):
            [ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge] = cPickle.load(open(poly_name+"_ints.pkl",'rb'))
        else:
            ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge = bg_config_space(poly_name)
            g = open(poly_name+"_ints.pkl",'wb')
            cPickle.dump([ints, paths, shell_int, shell_paths, edges, degens, open_edges, shell_edge],g,-1)
            g.close()

        print "                                                    complete"


        if os.path.isfile(poly_name+"_dfs.pkl"):
            print "Unpickling degrees of freedom data...               "
            dfs = cPickle.load(open(poly_name+"_dfs.pkl",'rb'))
            print "                                                    complete"
        else:
            print "Computing Intermediate Degrees of Freedom...        "

            dfs = []
            M = ints.shape[0]
            fM = float(M)
            for k in range(M):
                df = DoF(ints[k,:],v,f_inds,c)[0]
                dfs.append(df)
                print '%','{0}\r'.format(round(100*k/fM,1)),
            # Pickle dfs
            g = open(poly_name+"_dfs.pkl",'wb')
            cPickle.dump(np.array(dfs),g,-1)
            g.close()

            print "                                                    complete"




#import polyhedron structure
#vertex coordinates
#face center coordinates

#import BG configuration space

print "Loading Polyhedron Data...                              "


## PLATONIC SOLIDS

#[poly,poly_name] = [polyhedra.tetrahedron,"tetrahedron"]
#[poly,poly_name] = [polyhedra.cube,"cube"]
#[poly,poly_name] = [polyhedra.octahedron,"octahedron"]
[poly,poly_name] = [polyhedra.dodecahedron,"dodecahedron"]
#[poly,poly_name] = [polyhedra.icosahedron,"icosahedron"]

## ARCHIMEDEAN SOLIDS

#[poly,poly_name] = [polyhedra.truncated_tetrahedron,"truncated_tetrahedron"]
#[poly,poly_name] = [polyhedra.cuboctahedron,"cuboctahedron"]
#[poly,poly_name] = [polyhedra.truncated_cube,"truncated_cube"]
#[poly,poly_name] = [polyhedra.truncated_octahedron,"truncated_octahedron"]
#[poly,poly_name] = [polyhedra.rhombicuboctahedron,"rhombicuboctahedron"]
"""[poly,poly_name] = [polyhedra.truncated_cuboctahedron,"truncated_cuboctahedron"]"""
"""[poly,poly_name] = [polyhedra.icosidodecahedron,"icosidodecahedron"]"""
"""[poly,poly_name] = [polyhedra.truncated_dodecahedron,"truncated_dodecahedron"]"""
"""[poly,poly_name] = [polyhedra.truncated_icosahedron,"truncated_icosahedron"]"""

## CATALAN SOLIDS

#[poly,poly_name] = [polyhedra.triakis_tetrahedron,"triakis_tetrahedron"]
"""[poly,poly_name] = [polyhedra.rhombic_dodecahedron,"rhombic_dodecahedron"]"""
"""[poly,poly_name] = [polyhedra.triakis_octahedron,"triakis_octahedron"]"""
#[poly,poly_name] = [polyhedra.tetrakis_hexahedron,"tetrakis_hexahedron"]
"""[poly,poly_name] = [polyhedra.deltoidal_icositetrahedron,"deltoidal_icositetrahedron"]"""
"""[poly,poly_name] = [polyhedra.pentagonal_icositetrahedron,"pentagonal_icositetrahedron"]"""
"""[poly,poly_name] = [polyhedra.rhombic_triacontahedron,"rhombic_triacontahedron"]"""
"""
## OTHER SHAPES
nonbg = ["grid22", "grid23", "grid13"]

#[poly,poly_name] = [polyhedra.grid22,"grid22"]
#[poly,poly_name] = [polyhedra.grid23,"grid23"]
#[poly,poly_name] = [polyhedra.grid23b0,"grid23"] # bend along y = 1
#[poly,poly_name] = [polyhedra.grid23b1,"grid23"] # bends along x = 1,2
#[poly,poly_name] = [polyhedra.grid23,"grid13"]

find_nets(poly,poly_name)
"""
