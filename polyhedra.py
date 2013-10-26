import numpy as np

#give coordinates of polyhedra centered at origin with unit edges and face centers, face inds

def get_face_centers(f,v):
    #N_f = f.shape[0]
    #N_s = f[0].shape[0]
    N_f = len(f)
    #N_s = len(f[0])
    centers = np.zeros((N_f,3))
    for k in range(N_f):
        fc = np.zeros((1,3))
        N_s = len(f[k])
        for j in range(N_s):
            #print j,k,N_f,N_s
            fc += v[f[k][j],:]
        fc /= N_s
        centers[k,:] = fc
    return centers
    



def tetrahedron():
    verts = np.zeros((4,3))
    verts[0,:] = .5*np.array([1.0,	0.0,	-1/2**.5])
    verts[1,:] = .5*np.array([-1.0,	0.0,	-1/2**.5])
    verts[2,:] = .5*np.array([0.0,	1.0,	1/2**.5])
    verts[3,:] = .5*np.array([0.0,	-1.0,	1/2**.5])
    
    face_inds = [[0,	1,	2],
                 [0,	1,	3],
                 [0,	2,	3],
                 [1,	2,	3]]

    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents

def cube():
    verts = np.zeros((8,3))
    verts[0,:] = .5*np.array([1.0,	1.0,	1.0])
    verts[1,:] = .5*np.array([1.0,	1.0,	-1.0])
    verts[2,:] = .5*np.array([1.0,	-1.0,	1.0])
    verts[3,:] = .5*np.array([1.0,	-1.0,	-1.0])
    verts[4,:] = .5*np.array([-1.0,	1.0,	1.0])
    verts[5,:] = .5*np.array([-1.0,	1.0,	-1.0])
    verts[6,:] = .5*np.array([-1.0,	-1.0,	1.0])
    verts[7,:] = .5*np.array([-1.0,	-1.0,	-1.0])

    face_inds = [[1,	5,	7,	3],
                 [0,	2,	6,	4],
                 [0,	1,	3,	2],
                 [2,	3,	7,	6],
                 [0,	4,	5,	1],
                 [4,	6,	7,	5]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def octahedron():
    verts = np.zeros((6,3))
    verts[0,:] = 2.0**-.5*np.array([1.0,	0.0,	0.0])
    verts[1,:] = 2.0**-.5*np.array([-1.0,	0.0,	0.0])
    verts[2,:] = 2.0**-.5*np.array([0.0,	1.0,	0.0])
    verts[3,:] = 2.0**-.5*np.array([0.0,	-1.0,	0.0])
    verts[4,:] = 2.0**-.5*np.array([0.0,	0.0,	1.0])
    verts[5,:] = 2.0**-.5*np.array([0.0,	0.0,	-1.0])
    
    face_inds = [[0,	3,	4],
                 [1,	4,	3],
                 [0,	4,	2],
                 [0,	5,	3],
                 [1,	3,	5],
                 [1,	2,	4],
                 [0,	2,	5],
                 [1,	5,	2]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents

def dodecahedron():
    phi = .5*(5**.5 + 1.0)
    verts = np.zeros((20,3))
    verts[0,:] = .5*phi*np.array([1.0,	1.0,	1.0])
    verts[1,:] = .5*phi*np.array([1.0,	1.0,	-1.0])
    verts[2,:] = .5*phi*np.array([1.0,	-1.0,	1.0])
    verts[3,:] = .5*phi*np.array([1.0,	-1.0,	-1.0])
    verts[4,:] = .5*phi*np.array([-1.0,	1.0,	1.0])
    verts[5,:] = .5*phi*np.array([-1.0,	1.0,	-1.0])
    verts[6,:] = .5*phi*np.array([-1.0,	-1.0,	1.0])
    verts[7,:] = .5*phi*np.array([-1.0,	-1.0,	-1.0])

    verts[8,:] = .5*phi*np.array([0.0,	1.0/phi,	phi])
    verts[9,:] = .5*phi*np.array([0.0,	1.0/phi,	-phi])
    verts[10,:] = .5*phi*np.array([0.0,	-1.0/phi,	phi])
    verts[11,:] = .5*phi*np.array([0.0,	-1.0/phi,	-phi])
    
    verts[12,:] = .5*phi*np.array([1.0/phi,	phi,	0.0])
    verts[13,:] = .5*phi*np.array([1.0/phi,	-phi,	0.0])
    verts[14,:] = .5*phi*np.array([-1.0/phi,	phi,	0.0])
    verts[15,:] = .5*phi*np.array([-1.0/phi,	-phi,	0.0])
    
    verts[16,:] = .5*phi*np.array([phi,	0.0,	1.0/phi])
    verts[17,:] = .5*phi*np.array([-phi,	0.0,	1.0/phi])
    verts[18,:] = .5*phi*np.array([phi,	0.0,	-1.0/phi])
    verts[19,:] = .5*phi*np.array([-phi,	0.0,	-1.0/phi])
    
    face_inds = [[0,	16,	18,	1,	12],
                 [0,	8,	10,	2,	16],
                 [2,	13,	3,	18,	16],
                 [0,	12,	14,	4,	8],
                 [4,	17,	6,	10,	8],
                 [2,	10,	6,	15,	13],
                 [3,	13,	15,	7,	11],
                 [1,	18,	3,	11,	9],
                 [1,	9,	5,	14,	12],
                 [4,	14,	5,	19,	17],
                 [6,	17,	19,	7,	15],
                 [5,	9,	11,	7,	19]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def icosahedron():
    phi = .5*(5**.5 + 1.0)
    verts = np.zeros((12,3))
    verts[0,:] = .5*np.array([0.0,	1.0,	phi])
    verts[1,:] = .5*np.array([0.0,	1.0,	-phi])
    verts[2,:] = .5*np.array([0.0,	-1.0,	phi])
    verts[3,:] = .5*np.array([0.0,	-1.0,	-phi])

    verts[4,:] = .5*np.array([1.0,	phi,	0.0])
    verts[5,:] = .5*np.array([1.0,	-phi,	0.0])
    verts[6,:] = .5*np.array([-1.0,	phi,	0.0])
    verts[7,:] = .5*np.array([-1.0,	-phi,	0.0])

    verts[8,:] = .5*np.array([phi,	0.0,	1.0])
    verts[9,:] = .5*np.array([phi,	0.0,	-1.0])
    verts[10,:] = .5*np.array([-phi,	0.0,	1.0])
    verts[11,:] = .5*np.array([-phi,	0.0,	-1.0])
    
    face_inds = [[0,	2,	10],
                 [0,	8,	2],
                 [2,	7,	10],
                 [0,	10,	6],
                 [0,	6,	4],
                 [0,	4,	8],
                 [4,	9,	8],
                 [5,	8,	9],
                 [2,	8,	5],
                 [2,	5,	7],
                 [3,	7,	5],
                 [3,	11,	7],
                 [7,	11,	10],
                 [6,	10,	11],
                 [1,	6,	11],
                 [1,	4,	6],
                 [1,	9,	4],
                 [1,	3,	9],
                 [3,	5,	9],
                 [1,	11,	3]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents

def truncated_tetrahedron():
    verts = np.zeros((12,3))
    verts[0,:] = 8.0**-0.5*np.array([3.0,	1.0,	1.0])
    verts[1,:] = 8.0**-0.5*np.array([1.0,	3.0,	1.0])
    verts[2,:] = 8.0**-0.5*np.array([1.0,	1.0,	3.0])

    verts[3,:] = 8.0**-0.5*np.array([-3.0,	-1.0,	1.0])
    verts[4,:] = 8.0**-0.5*np.array([-1.0,	-3.0,	1.0])
    verts[5,:] = 8.0**-0.5*np.array([-1.0,	-1.0,	3.0])

    verts[6,:] = 8.0**-0.5*np.array([-3.0,	1.0,	-1.0])
    verts[7,:] = 8.0**-0.5*np.array([-1.0,	3.0,	-1.0])
    verts[8,:] = 8.0**-0.5*np.array([-1.0,	1.0,	-3.0])

    verts[9,:] = 8.0**-0.5*np.array([3.0,	-1.0,	-1.0])
    verts[10,:] = 8.0**-0.5*np.array([1.0,	-3.0,	-1.0])
    verts[11,:] = 8.0**-0.5*np.array([1.0,	-1.0,	-3.0])

    
    face_inds = [[0,	2,	1],
                 [0,	1,	7,	8,	11,	9],
                 [0,	9,	10,	4,	5,	2],
                 [1,	2,	5,	3,	6,	7],
                 [6,	8,	7],
                 [9,	11,	10],
                 [3,	5,	4],
                 [3,	4,	10,	11,	8,	6]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def cuboctahedron():
    verts = np.zeros((12,3))

    verts[0,:] = 2.0**-0.5*np.array([1.0,	1.0,	0.0])
    verts[1,:] = 2.0**-0.5*np.array([1.0,	-1.0,	0.0])
    verts[2,:] = 2.0**-0.5*np.array([-1.0,	1.0,	0.0])
    verts[3,:] = 2.0**-0.5*np.array([-1.0,	-1.0,	0.0])

    verts[4,:] = 2.0**-0.5*np.array([1.0,	0.0,	1.0])
    verts[5,:] = 2.0**-0.5*np.array([1.0,	0.0,	-1.0])
    verts[6,:] = 2.0**-0.5*np.array([-1.0,	0.0,	1.0])
    verts[7,:] = 2.0**-0.5*np.array([-1.0,	0.0,	-1.0])

    verts[8,:] = 2.0**-0.5*np.array([0.0,	1.0,	1.0])
    verts[9,:] = 2.0**-0.5*np.array([0.0,	1.0,	-1.0])
    verts[10,:] = 2.0**-0.5*np.array([0.0,	-1.0,	1.0])
    verts[11,:] = 2.0**-0.5*np.array([0.0,	-1.0,	-1.0])

    
    face_inds = [[0,	4,	8],
                 [4,	10,	6,	8],
                 [0,	8,	2,	9],
                 [0,	5,	1,	4],
                 [1,	10,	4],
                 [3,	6,	10],
                 [2,	8,	6],
                 [2,	7,	9],
                 [0,	9,	5],
                 [1,	5,	11],
                 [1,	11,	3,	10],
                 [2,	6,	3,	7],
                 [5,	9,	7,	11],
                 [3,	11,	7]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def grid22():
    verts = np.zeros((12,3))

    verts[0,:] = np.array([0.0,		0.0,  	0.0])
    verts[1,:] = np.array([0.0,		1.0,	0.0])
    verts[2,:] = np.array([0.0,     	2.0,	0.0])
    verts[3,:] = np.array([1.0,		0.0,	0.0])
    verts[4,:] = np.array([1.0,		1.0,	0.0])
    verts[5,:] = np.array([1.0,		2.0,	0.0])
    verts[6,:] = np.array([2.0,		0.0,	0.0])
    verts[7,:] = np.array([2.0,		1.0,	0.0])
    verts[8,:] = np.array([2.0,		2.0,	0.0])
 
    
    face_inds = [[0,	1,	4,	3],
                 [1,	2,	5,	4],
                 [3,	4,	7,	6],
                 [4,	5,	8,	7]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents

def grid23():
    verts = np.zeros((12,3))

    verts[0,:] = np.array([0.0,		0.0,  	0.0])
    verts[1,:] = np.array([0.0,		1.0,	0.0])
    verts[2,:] = np.array([0.0,     	2.0,	0.0])
    verts[3,:] = np.array([1.0,		0.0,	0.0])
    verts[4,:] = np.array([1.0,		1.0,	0.0])
    verts[5,:] = np.array([1.0,		2.0,	0.0])
    verts[6,:] = np.array([2.0,		0.0,	0.0])
    verts[7,:] = np.array([2.0,		1.0,	0.0])
    verts[8,:] = np.array([2.0,		2.0,	0.0])
    verts[9,:] = np.array([3.0,		0.0,	0.0])
    verts[10,:] = np.array([3.0,	1.0,	0.0])
    verts[11,:] = np.array([3.0,	2.0,	0.0])

    
    face_inds = [[0,	1,	4,	3],
                 [1,	2,	5,	4],
                 [3,	4,	7,	6],
                 [4,	5,	8,	7],
                 [6,	7,	10,	9],
                 [7,	8,	11,	10]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents

def grid23b0():
    verts = np.zeros((12,3))

    verts[0,:] = np.array([0.0,		0.0,  			0.0])
    verts[1,:] = np.array([0.0,		1.0,			0.0])
    verts[2,:] = np.array([0.0,     	1.0+0.5*2.0**0.5,	0.5*2.0**0.5])
    verts[3,:] = np.array([1.0,		0.0,			0.0])
    verts[4,:] = np.array([1.0,		1.0,			0.0])
    verts[5,:] = np.array([1.0,		1.0+0.5*2.0**0.5,	0.5*2.0**0.5])
    verts[6,:] = np.array([2.0,		0.0,			0.0])
    verts[7,:] = np.array([2.0,		1.0,			0.0])
    verts[8,:] = np.array([2.0,		1.0+0.5*2.0**0.5,	0.5*2.0**0.5])
    verts[9,:] = np.array([3.0,		0.0,			0.0])
    verts[10,:] = np.array([3.0,	1.0,			0.0])
    verts[11,:] = np.array([3.0,	1.0+0.5*2.0**0.5,	0.5*2.0**0.5])

    
    face_inds = [[0,	1,	4,	3],
                 [1,	2,	5,	4],
                 [3,	4,	7,	6],
                 [4,	5,	8,	7],
                 [6,	7,	10,	9],
                 [7,	8,	11,	10]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents

def grid23b1():
    verts = np.zeros((12,3))

    verts[0,:] = np.array([0.0,		0.0,  	0.0])
    verts[1,:] = np.array([0.0,		1.0,	0.0])
    verts[2,:] = np.array([0.0,     	2.0,	0.0])
    verts[3,:] = np.array([1.0,		0.0,	0.0])
    verts[4,:] = np.array([1.0,		1.0,	0.0])
    verts[5,:] = np.array([1.0,		2.0,	0.0])
    verts[6,:] = np.array([1.0+0.5*2.0**0.5,		0.0,	0.5*2.0**0.5])
    verts[7,:] = np.array([1.0+0.5*2.0**0.5,		1.0,	0.5*2.0**0.5])
    verts[8,:] = np.array([1.0+0.5*2.0**0.5,		2.0,	0.5*2.0**0.5])
    verts[9,:] = np.array( [1.0+0.5*2.0**0.5,		0.0,	1.0+0.5*2.0**0.5])
    verts[10,:] = np.array([1.0+0.5*2.0**0.5,		1.0,	1.0+0.5*2.0**0.5])
    verts[11,:] = np.array([1.0+0.5*2.0**0.5,		2.0,	1.0+0.5*2.0**0.5]) 

    
    face_inds = [[0,	1,	4,	3],
                 [1,	2,	5,	4],
                 [3,	4,	7,	6],
                 [4,	5,	8,	7],
                 [6,	7,	10,	9],
                 [7,	8,	11,	10]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def grid13():
    verts = np.zeros((12,3))

    verts[0,:] = np.array([0.0,		0.0,  	0.0])
    verts[1,:] = np.array([0.0,		1.0,	0.0])
    verts[2,:] = np.array([1.0,		0.0,	0.0])
    verts[3,:] = np.array([1.0,		1.0,	0.0])
    verts[4,:] = np.array([2.0,		0.0,	0.0])
    verts[5,:] = np.array([2.0,		1.0,	0.0])
    verts[6,:] = np.array([3.0,		0.0,	0.0])
    verts[7,:] = np.array([3.0,		1.0,	0.0])
 
    
    face_inds = [[0,	1,	3,	2],
                 [2,	3,	5,	4],
                 [4,	5,	7,	6]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents



def truncated_cube():
    verts = np.zeros((24,3))
    xi = 2.0**.5 - 1.0

    verts[0,:] = 0.5/xi*np.array([xi,		1.0,	1.0])
    verts[1,:] = 0.5/xi*np.array([xi,		1.0,	-1.0])
    verts[2,:] = 0.5/xi*np.array([xi,		-1.0,	1.0])
    verts[3,:] = 0.5/xi*np.array([xi,		-1.0,	-1.0])
    verts[4,:] = 0.5/xi*np.array([-xi,		1.0,	1.0])
    verts[5,:] = 0.5/xi*np.array([-xi,		1.0,	-1.0])
    verts[6,:] = 0.5/xi*np.array([-xi,		-1.0,	1.0])
    verts[7,:] = 0.5/xi*np.array([-xi,		-1.0,	-1.0])

    verts[8,:] = 0.5/xi*np.array([1.0,		xi,	1.0])
    verts[9,:] = 0.5/xi*np.array([1.0,		xi,	-1.0])
    verts[10,:] = 0.5/xi*np.array([1.0,		-xi,	1.0])
    verts[11,:] = 0.5/xi*np.array([1.0,		-xi,	-1.0])
    verts[12,:] = 0.5/xi*np.array([-1.0,	xi,	1.0])
    verts[13,:] = 0.5/xi*np.array([-1.0,	xi,	-1.0])
    verts[14,:] = 0.5/xi*np.array([-1.0,	-xi,	1.0])
    verts[15,:] = 0.5/xi*np.array([-1.0,	-xi,	-1.0])

    verts[16,:] = 0.5/xi*np.array([1.0,		1.0,	xi])
    verts[17,:] = 0.5/xi*np.array([1.0,		1.0,	-xi])
    verts[18,:] = 0.5/xi*np.array([1.0,		-1.0,	xi])
    verts[19,:] = 0.5/xi*np.array([1.0,		-1.0,	-xi])
    verts[20,:] = 0.5/xi*np.array([-1.0,	1.0,	xi])
    verts[21,:] = 0.5/xi*np.array([-1.0,	1.0,	-xi])
    verts[22,:] = 0.5/xi*np.array([-1.0,	-1.0,	xi])
    verts[23,:] = 0.5/xi*np.array([-1.0,	-1.0,	-xi])
    
    
    face_inds = [[6,	22,	14],
                 [0,	8,	10,	2,	6,	14,	12,	4],
                 [2,	18,	19,	3,	7,	23,	22,	6],
                 [12,	14,	22,	23,	15,	13,	21,	20],
                 [4,	12,	20],
                 [0,	16,    	8],
                 [2,	10,	18],
                 [3,	19,	11],
                 [7,	15,	23],
                 [5,	21,	13],
                 [0,	4,	20,	21,	5,	1,	17,	16],
                 [8,	16,	17,	9,	11,	19,	18,	10],
                 [1,	5,	13,	15,	7,	3,	11,	9],
                 [1,	9,	17]]
    
    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def rhombicuboctahedron():
    verts = np.zeros((24,3))

    verts[0,:] = 0.5*np.array([1.0,	1.0,	(1.0+2.0**0.5)])
    verts[1,:] = 0.5*np.array([1.0,	1.0,	-(1.0+2.0**0.5)])
    verts[2,:] = 0.5*np.array([1.0,	-1.0,	(1.0+2.0**0.5)])
    verts[3,:] = 0.5*np.array([1.0,	-1.0,	-(1.0+2.0**0.5)])
    verts[4,:] = 0.5*np.array([-1.0,	1.0,	(1.0+2.0**0.5)])
    verts[5,:] = 0.5*np.array([-1.0,	1.0,	-(1.0+2.0**0.5)])
    verts[6,:] = 0.5*np.array([-1.0,	-1.0,	(1.0+2.0**0.5)])
    verts[7,:] = 0.5*np.array([-1.0,	-1.0,	-(1.0+2.0**0.5)])

    verts[8,:] =  0.5*np.array([1.0,	(1.0+2.0**0.5),		1.0])
    verts[9,:] =  0.5*np.array([1.0,	(1.0+2.0**0.5),		-1.0])
    verts[10,:] = 0.5*np.array([1.0,	-(1.0+2.0**0.5),	1.0])
    verts[11,:] = 0.5*np.array([1.0,	-(1.0+2.0**0.5),	-1.0])
    verts[12,:] = 0.5*np.array([-1.0,	(1.0+2.0**0.5),		1.0])
    verts[13,:] = 0.5*np.array([-1.0,	(1.0+2.0**0.5),		-1.0])
    verts[14,:] = 0.5*np.array([-1.0,	-(1.0+2.0**0.5),	1.0])
    verts[15,:] = 0.5*np.array([-1.0,	-(1.0+2.0**0.5),	-1.0])

    verts[16,:] = 0.5*np.array([(1.0+2.0**0.5),		1.0,	1.0])
    verts[17,:] = 0.5*np.array([(1.0+2.0**0.5),		1.0,	-1.0])
    verts[18,:] = 0.5*np.array([(1.0+2.0**0.5),		-1.0,	1.0])
    verts[19,:] = 0.5*np.array([(1.0+2.0**0.5),		-1.0,	-1.0])
    verts[20,:] = 0.5*np.array([-(1.0+2.0**0.5),	1.0,	1.0])
    verts[21,:] = 0.5*np.array([-(1.0+2.0**0.5),	1.0,	-1.0])
    verts[22,:] = 0.5*np.array([-(1.0+2.0**0.5),	-1.0,	1.0])
    verts[23,:] = 0.5*np.array([-(1.0+2.0**0.5),	-1.0,	-1.0]) 
    
    
    face_inds = [[0, 2, 6, 4],
                 [0, 4, 12, 8],
                 [0, 8, 16],
                 [0, 16, 18, 2],
                 [2, 18, 10],
                 [2, 10, 14, 6],
                 [6, 14, 22],
                 [4, 6, 22, 20],
                 [4, 20, 12],
                 [8, 12, 13, 9],
                 [8, 9, 17, 16],
                 [16, 17, 19, 18],
                 [10, 18, 19, 11],
                 [10, 11, 15, 14],
                 [14, 15, 23, 22],
                 [20, 22, 23, 21],
                 [12, 20, 21, 13],
                 [1, 17, 9],
                 [3, 11, 19],
                 [7, 23, 15],
                 [5, 13, 21],
                 [1, 9, 13, 5],
                 [1, 3, 19, 17],
                 [3, 7, 15, 11],
                 [5, 21, 23, 7],
                 [1, 5, 7, 3]]

    cents = get_face_centers(face_inds,verts)

    return verts, face_inds, cents


def truncated_octahedron():
    verts = np.zeros((24,3))

    verts[0,:] = 2.0**-0.5*np.array([0.0,	1.0,	2.0])
    verts[1,:] = 2.0**-0.5*np.array([0.0,	1.0,	-2.0])
    verts[2,:] = 2.0**-0.5*np.array([0.0,	-1.0,	2.0])
    verts[3,:] = 2.0**-0.5*np.array([0.0,	-1.0,	-2.0])
    verts[4,:] = 2.0**-0.5*np.array([0.0,	2.0,	1.0])
    verts[5,:] = 2.0**-0.5*np.array([0.0,	2.0,	-1.0])
    verts[6,:] = 2.0**-0.5*np.array([0.0,	-2.0,	1.0])
    verts[7,:] = 2.0**-0.5*np.array([0.0,	-2.0,	-1.0])

    verts[8,:] =  2.0**-0.5*np.array([1.0,	0.0,	2.0])
    verts[9,:] =  2.0**-0.5*np.array([1.0,	0.0,	-2.0])
    verts[10,:] = 2.0**-0.5*np.array([-1.0,	0.0,	2.0])
    verts[11,:] = 2.0**-0.5*np.array([-1.0,	0.0,	-2.0])
    verts[12,:] = 2.0**-0.5*np.array([1.0,	2.0,	0.0])
    verts[13,:] = 2.0**-0.5*np.array([1.0,	-2.0,	0.0])
    verts[14,:] = 2.0**-0.5*np.array([-1.0,	2.0,	0.0])
    verts[15,:] = 2.0**-0.5*np.array([-1.0,	-2.0,	0.0])

    verts[16,:] = 2.0**-0.5*np.array([2.0,	0.0,	1.0])
    verts[17,:] = 2.0**-0.5*np.array([2.0,	0.0,	-1.0])
    verts[18,:] = 2.0**-0.5*np.array([-2.0,	0.0,	1.0])
    verts[19,:] = 2.0**-0.5*np.array([-2.0,	0.0,	-1.0])
    verts[20,:] = 2.0**-0.5*np.array([2.0,	1.0,	0.0])
    verts[21,:] = 2.0**-0.5*np.array([2.0,	-1.0,	0.0])
    verts[22,:] = 2.0**-0.5*np.array([-2.0,	1.0,	0.0])
    verts[23,:] = 2.0**-0.5*np.array([-2.0,	-1.0,	0.0])
    
    
    face_inds = [[0, 10, 18, 22, 14, 4],
                 [0, 8, 2, 10],
                 [18, 23, 19, 22],
                 [4, 14, 5, 12],
                 [2, 6, 15, 23, 18, 10],
                 [1, 5, 14, 22, 19, 11],
                 [0, 4, 12, 20, 16, 8],
                 [2, 8, 16, 21, 13, 6],
                 [3, 11, 19, 23, 15, 7],
                 [1, 9, 17, 20, 12, 5],
                 [6, 13, 7, 15],
                 [1, 11, 3, 9],
                 [16, 20, 17, 21],
                 [3, 7, 13, 21, 17, 9]]
    
    cents = get_face_centers(face_inds,verts)
    #cents = []
    return verts, face_inds, cents


def triakis_tetrahedron():
    verts = np.zeros((8,3))

    verts[0,:] = 8.0**-0.5*np.array([(5.0/3.0),	(5.0/3.0),	(5.0/3.0)])
                 
    verts[1,:] = 8.0**-0.5*np.array([1.0,	1.0,	-1.0])
    verts[2,:] = 8.0**-0.5*np.array([1.0,	-1.0,	1.0])
    verts[3,:] = 8.0**-0.5*np.array([-1.0,	1.0,	1.0])
                 
    verts[4,:] = 8.0**-0.5*np.array([-(5.0/3.0),	(5.0/3.0),	-(5.0/3.0)])
    verts[5,:] = 8.0**-0.5*np.array([(5.0/3.0),		-(5.0/3.0),	-(5.0/3.0)])
    verts[6,:] = 8.0**-0.5*np.array([-(5.0/3.0),	-(5.0/3.0),	(5.0/3.0)])
                 
    verts[7,:] = 8.0**-0.5*np.array([-1.0,	-1.0,	-1.0])
    
    
    face_inds = [[3, 6, 4],
                 [0, 3, 4],
                 [0, 6, 3],
                 [4, 6, 7],
                 [0, 4, 1],
                 [0, 2, 6],
                 [5, 7, 6],
                 [4, 7, 5],
                 [1, 4, 5],
                 [0, 1, 5],
                 [0, 5, 2],
                 [2, 5, 6]]
    
    cents = get_face_centers(face_inds,verts)
    #cents = []
    return verts, face_inds, cents


def tetrakis_hexahedron():
    verts = np.zeros((14,3))

    verts[0,:] = 3.0**-0.5*np.array([-1.0,	1.0,	1.0])              
    verts[1,:] = 4.0**-0.5*np.array([0.0,	0.0,	2.0])
    verts[2,:] = 4.0**-0.5*np.array([-2.0,	0.0,	0.0])
    verts[3,:] = 4.0**-0.5*np.array([0.0,	2.0,	0.0])    
    verts[4,:] = 3.0**-0.5*np.array([-1.0,	-1.0,	1.0])
    verts[5,:] = 3.0**-0.5*np.array([-1.0,	1.0,	-1.0])
    verts[6,:] = 3.0**-0.5*np.array([1.0,	1.0,	1.0])
    verts[7,:] = 3.0**-0.5*np.array([1.0,	-1.0,	1.0])
    verts[8,:] = 3.0**-0.5*np.array([-1.0,	-1.0,	-1.0])              
    verts[9,:] = 3.0**-0.5*np.array([1.0,	1.0,	-1.0])
    verts[10,:] = 4.0**-0.5*np.array([0.0,	-2.0,	0.0])
    verts[11,:] = 4.0**-0.5*np.array([0.0,	0.0,	-2.0])    
    verts[12,:] = 4.0**-0.5*np.array([2.0,	0.0,	0.0])
    verts[13,:] = 3.0**-0.5*np.array([1.0,	-1.0,	-1.0])
    
    
    face_inds = [[2, 4, 8],
                 [2, 8, 5],
                 [0, 2, 5],
                 [0, 4, 2],
                 [0, 1, 4],
                 [4, 10, 8],
                 [5, 8, 11],
                 [0, 5, 3],
                 [0, 6, 1],
                 [1, 7, 4],
                 [4, 7, 10],
                 [8, 10, 13],
                 [8, 13, 11],
                 [5, 11, 9],
                 [3, 5, 9],
                 [0, 3, 6],
                 [1, 6, 7],
                 [7, 13, 10],
                 [9, 11, 13],
                 [3, 9, 6],
                 [6, 12, 7],
                 [7, 12, 13],
                 [9, 13, 12],
                 [6, 9, 12]]
    
    cents = get_face_centers(face_inds,verts)
    #cents = []
    return verts, face_inds, cents



def deltoidal_icositetrahedron():
    verts = np.zeros((26,3))

    verts[0,:] = 3.0**-0.5*np.array([-1.0,	1.0,	1.0])              
    verts[1,:] = 4.0**-0.5*np.array([0.0,	0.0,	2.0])
    verts[2,:] = 4.0**-0.5*np.array([-2.0,	0.0,	0.0])
    verts[3,:] = 4.0**-0.5*np.array([0.0,	2.0,	0.0])    
    verts[4,:] = 3.0**-0.5*np.array([-1.0,	-1.0,	1.0])
    verts[5,:] = 3.0**-0.5*np.array([-1.0,	1.0,	-1.0])
    verts[6,:] = 3.0**-0.5*np.array([1.0,	1.0,	1.0])
    verts[7,:] = 3.0**-0.5*np.array([1.0,	-1.0,	1.0])
    verts[8,:] = 3.0**-0.5*np.array([-1.0,	-1.0,	-1.0])              
    verts[9,:] = 3.0**-0.5*np.array([1.0,	1.0,	-1.0])
    verts[10,:] = 4.0**-0.5*np.array([0.0,	-2.0,	0.0])
    verts[11,:] = 4.0**-0.5*np.array([0.0,	0.0,	-2.0])    
    verts[12,:] = 4.0**-0.5*np.array([2.0,	0.0,	0.0])
    verts[13,:] = 3.0**-0.5*np.array([1.0,	-1.0,	-1.0])
    
    
    face_inds = [[2, 4, 8],
                 [2, 8, 5],
                 [0, 2, 5],
                 [0, 4, 2],
                 [0, 1, 4],
                 [4, 10, 8],
                 [5, 8, 11],
                 [0, 5, 3],
                 [0, 6, 1],
                 [1, 7, 4],
                 [4, 7, 10],
                 [8, 10, 13],
                 [8, 13, 11],
                 [5, 11, 9],
                 [3, 5, 9],
                 [0, 3, 6],
                 [1, 6, 7],
                 [7, 13, 10],
                 [9, 11, 13],
                 [3, 9, 6],
                 [6, 12, 7],
                 [7, 12, 13],
                 [9, 13, 12],
                 [6, 9, 12]]
    
    cents = get_face_centers(face_inds,verts)
    #cents = []
    return verts, face_inds, cents

