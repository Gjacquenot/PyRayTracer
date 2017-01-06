# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:59:12 2015

Preliminary code for ray tracing with python
prereq: vtk.
Thanks to https://pyscience.wordpress.com/2014/10/05/from-ray-casting-to-ray-tracing-with-python-and-vtk/

In order to proceed, you will have to exit the renderwindow, it is waiting for your input.

@author: jaap
"""
from __future__ import division
import vtk
import numpy as np

RayCastLength = 2000
ColorRay = [1.0, 1.0, 0.0]
ColorRayMiss = [1.0, 1.0, 1.0]

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def addPoint(ren, appendFilter, p, color=[0.0, 0.0, 0.0], radius=0.2, ):
    point = vtk.vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)
    # map point
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())
    # set actor for point
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    #draw point in renderer
    ren.AddActor(actor)
    #appendFilter.AddInputData(line.GetOutput())
    #appendFilter.Update()

def addLine(ren, appendFilter, p1, p2, color=[0.0, 0.0, 1.0], opacity=1.0):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty()

    ren.AddActor(actor)
    
    appendFilter.AddInputData(line.GetOutput())
    appendFilter.Update()


def isHit(obbTree, pSource, pTarget):
    """Returns True if the line intersects with the mesh in 'obbTree'"""
    code = obbTree.IntersectWithLine(pSource, pTarget, None, None)
    if code == 0:
        return False
    return True

def GetIntersect(obbTree, pSource, pTarget):
    # Create an empty 'vtkPoints' object to store the intersection point coordinates
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    # Create an empty 'vtkIdList' object to store the ids of the cells that intersect
    # with the cast rays
    cellIds = vtk.vtkIdList()
    
    # Perform intersection
    code = obbTree.IntersectWithLine(pSource, pTarget, points, cellIds)
    assert (code != 0)
    # Get point-data 
    pointData = points.GetData()
    # Get number of intersection points found
    noPoints = pointData.GetNumberOfTuples()
    # Get number of intersected cell ids
    noIds = cellIds.GetNumberOfIds()
    
    assert (noPoints == noIds)
    assert (noPoints > 0)
    # Loop through the found points and cells and store
    # them in lists
    pointsInter = []
    cellIdsInter = []
    for idx in range(noPoints):
        pointsInter.append(pointData.GetTuple3(idx))
        cellIdsInter.append(cellIds.GetId(idx))
    
    return pointsInter, cellIdsInter

def calcVecReflect(vecInc, vecNor):
    '''    
    http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
    Vector reflect(const Vector& normal, const Vactor& incident)
    {
        const double cosI = -dot(normal, incident);
        return incident + 2 * cosI * normal;
    }  
    '''
    vecInc = l2n(vecInc)
    vecNor = l2n(vecNor)
    cosI = -np.dot(vecNor, vecInc)
    vecRef = vecInc + 2*cosI*vecNor
    return n2l(vecRef)


def calcVecRefract(vecInc, vecNor, n1=1.0, n2=1.33):
    '''
    http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
    Vector refract(Const Vector& normal, const Vector& incident, double n1, double n2)
    {
    	const double n = n1/n2;
    	const double cosI = -dot(normal, incident)
    	const double sinT2 = n*n*(1.0-cosI*cosI);
    	if (sinT2 > 1.0) return Vactor::invalid; //TIR
    	const double cosT = sqrt(1.0 - sinT2);
    	return n * incident + (n * cosI - cosT) * normal;
    }
    n1 = first medium, n2 is second medium
    '''
    n=n1/n2
    vecInc = l2n(vecInc)
    vecNor = l2n(vecNor)    
    cosI = -np.dot(vecNor, vecInc)
    sinT2 = n**2*(1-cosI**2)
    #assert (sinT2 < 1.0)
    if sinT2 < 1.0:
        cosT = np.sqrt(1.0-sinT2)
        vecRef = n * vecInc + (n * cosI - cosT) * vecNor
        
    else:
        # If sinT2 > 1.0, lets send the ray back for now
        vecRef = vecInc
    return n2l(vecRef)

# Transformations
def is_same_transform(matrix0, matrix1):
    """Return True if two matrices perform same transformation.

    >>> is_same_transform(np.identity(4), np.identity(4))
    True
    >>> is_same_transform(np.identity(4), random_rotation_matrix())
    False

    """
    matrix0 = np.array(matrix0, dtype=np.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = np.array(matrix1, dtype=np.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return np.allclose(matrix0, matrix1)

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def Cartesian2Spherical(xyz):
    ptsnew = np.empty(np.shape(xyz))
    ptsnew[:,0] = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    ptsnew[:,1] = np.arccos(xyz[:,2]/np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)) # for elevation angle defined from Z-axis down
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
    
def Spherical2Cartesian(rtp):
    ptsnew = np.empty(np.shape(rtp))
    ptsnew[:,0] = rtp[:,0]*np.sin(rtp[:,1])*np.cos(rtp[:,2])
    ptsnew[:,1] = rtp[:,0]*np.sin(rtp[:,1])*np.sin(rtp[:,2])
    ptsnew[:,2] = rtp[:,0]*np.cos(rtp[:,1])    
    return ptsnew