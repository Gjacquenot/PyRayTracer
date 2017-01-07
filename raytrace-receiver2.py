# -*- coding: utf-8 -*-
"""
@author: Jaap Verheggen, Guillaume Jacquenot
"""

from __future__ import division
import vtk
import numpy as np
import matplotlib.pyplot as plt
import vtk.util.numpy_support as ns

import raytracer as ray


def glyphs(cells, color=[1.0, 1.0, 1.0], size=1):
    # Visualize normals as done previously but using refracted or reflected cells
    arrow = vtk.vtkArrowSource()
    glyphCell = vtk.vtkGlyph3D()
    glyphCell.SetInputData(cells)
    glyphCell.SetSourceConnection(arrow.GetOutputPort())
    glyphCell.SetVectorModeToUseNormal()
    glyphCell.SetScaleFactor(size)

    glyphMapperCell = vtk.vtkPolyDataMapper()
    glyphMapperCell.SetInputConnection(glyphCell.GetOutputPort())

    glyphActorCell = vtk.vtkActor()
    glyphActorCell.SetMapper(glyphMapperCell)
    glyphActorCell.GetProperty().SetColor(color)
    return glyphActorCell


def getnormals(surf, vertices=False, flip=False):
    # Calculate normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    normals.SetInputData(surf.GetOutput())
    if vertices:
        # Enable normal calculation at cell vertices
        normals.ComputePointNormalsOn()
        # Disable normal calculation at cell centers
        normals.ComputeCellNormalsOff()
    else:
        # Disable normal calculation at cell vertices
        normals.ComputePointNormalsOff()
        # Enable normal calculation at cell centers
        normals.ComputeCellNormalsOn()

    # Disable splitting of sharp edges
    normals.ConsistencyOn()
    normals.SplittingOff()
    # Disable global flipping of normal orientation
    if flip:
        normals.FlipNormalsOn()
        print('flip is true')
    else:
        normals.FlipNormalsOff()
    # Enable automatic determination of correct normal orientation
    # normals.AutoOrientNormalsOn()
    # Perform calculation
    normals.Update()
    # Create dummy array for glyphs
    if vertices:
        verticepoints = surf.GetOutput().GetPoints()
        normal_polydata = normals.GetOutput()
        return normal_polydata, verticepoints
    else:
        normalcellcenters = vtk.vtkCellCenters()
        normalcellcenters.VertexCellsOn()
        normalcellcenters.SetInputConnection(normals.GetOutputPort())
        normalcellcenters.Update()
        #
        pointsCellCenters = normalcellcenters.GetOutput(0)
        normal_points = vtk.vtkPoints()
        normal_points.SetDataTypeToDouble()
        # Vectors where intersections are found
        normal_vectors = vtk.vtkDoubleArray()
        normal_vectors.SetNumberOfComponents(3)
        # Loop through all point centers and add a point-actor through 'addPoint'
        for idx in range(pointsCellCenters.GetNumberOfPoints()):
            normal_points.InsertNextPoint(pointsCellCenters.GetPoint(idx))
            normalsurf2 = normalcellcenters.GetOutput().GetCellData().GetNormals().GetTuple(idx)
            # Insert the normal vector of the intersection cell in the dummy container
            normal_vectors.InsertNextTuple(normalsurf2)
        # Need to transform polydatanormals to polydata so I can reuse functions
        normal_polydata = vtk.vtkPolyData()
        normal_polydata.SetPoints(normal_points)
        normal_polydata.GetPointData().SetNormals(normal_vectors)
        return normal_polydata, normalcellcenters.GetOutput()


def stop(ren, appendFilter, srf1, srf2, scene):
    # rays between two surfaces. A stop.
    obbsurf2 = vtk.vtkOBBTree()
    obbsurf2.SetDataSet(srf2['surface'].GetOutput())
    obbsurf2.BuildLocator()
    # where intersections are found
    intersection_points = vtk.vtkPoints()
    intersection_points.SetDataTypeToDouble()
    # Loop through all of surface1 cell-centers
    for idx in range(srf1['raypoints'].GetNumberOfPoints()):
        # Get coordinates of surface1 cell center
        pointSurf1 = srf1['raypoints'].GetPoint(idx)
        # Get incident vector at that cell
        normalsurf1 = srf1['rays'].GetPointData().GetNormals().GetTuple(idx)
        # Calculate the 'target' of the ray based on 'RayCastLength'
        pointRaySurf2 = list(np.array(pointSurf1) + ray.RayCastLength * np.array(normalsurf1))
        # Check if there are any intersections for the given ray
        if ray.isHit(obbsurf2, pointSurf1, pointRaySurf2):
            # Retrieve coordinates of intersection points and intersected cell ids
            pointsInter, cellIdsInter = ray.getIntersect(obbsurf2, pointSurf1, pointRaySurf2)
            # print(cellIdsInter)
            # Render lines/rays emanating from the Source. Rays that intersect are
            ray.addLine(ren, appendFilter, pointSurf1, pointsInter[0], [1.0, 1.0, 0.0], opacity=0.25)
            # Insert the coordinates of the intersection point in the dummy container
            intersection_points.InsertNextPoint(pointsInter[0])

    return intersection_points


def flat(srf):
    # Setup four points
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    points.InsertNextPoint(-0.5 * srf['width'], -0.5 * srf['height'], 0.0)
    points.InsertNextPoint(+0.5 * srf['width'], -0.5 * srf['height'], 0.0)
    points.InsertNextPoint(+0.5 * srf['width'], +0.5 * srf['height'], 0.0)
    points.InsertNextPoint(-0.5 * srf['width'], +0.5 * srf['height'], 0.0)
    # Create the polygon
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # rotate and translate
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    if 'rotateWXYZ' in srf:
        transform.RotateWXYZ(srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
    if 'center' in srf:
        transform.Translate(srf['center'])
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    pgt.SetTransform(transform)
    pgt.SetInputData(polygonPolyData)
    pgt.Update()
    extrude = pgt
    # Subdivision filters only work on triangles
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(extrude.GetOutputPort())
    triangles.Update()
    # Lets subdivide it for no reason at all
    return triangles


def cylinder(srf):
    source = vtk.vtkArcSource()
    z1 = srf['curvx'] - (srf['curvx'] / abs(srf['curvx'])) * np.sqrt(srf['curvx']**2 - (srf['width'] / 2)**2)  # s +/- sqrt(s^2-(w/2)^2)
    x1 = +0.5 * srf['width']
    y1 = -0.5 * srf['height']
    source.SetCenter(0, y1, srf['curvx'])
    source.SetPoint1(x1, y1, z1)
    source.SetPoint2(-x1, y1, z1)
    source.SetResolution(srf['resolution'])
    # Linear extrude arc
    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(source.GetOutputPort())
    extrude.SetExtrusionTypeToVectorExtrusion()
    extrude.SetVector(0, 1, 0)
    extrude.SetScaleFactor(srf['height'])
    extrude.Update()
    # Rotate and translate
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    if 'rotateWXYZ' in srf:
        transform.RotateWXYZ(srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
    if 'center' in srf:
        transform.Translate(srf['center'])
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    pgt.SetTransform(transform)
    pgt.SetInputData(extrude.GetOutput())
    pgt.Update()
    extrude = pgt
    # Subdivision filters only work on triangles
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(extrude.GetOutputPort())
    triangles.Update()
    # Lets subdivide it for no reason at all
    return triangles


def asphere(srf):
    # Delaunay mesh
    dr = (srf['diameter']) / 2 / (srf['resolution'] - 1)  # radius
    R = srf['coeffs']['R']
    k = srf['coeffs']['k']
    A2 = srf['coeffs']['A2']
    A4 = srf['coeffs']['A4']
    A6 = srf['coeffs']['A6']
    A8 = srf['coeffs']['A8']
    A10 = srf['coeffs']['A10']
    A12 = srf['coeffs']['A12']
    sizep = [sum([1 + x * 5 for x in range(srf['resolution'])]), 3]
    array_xyz = np.empty(sizep)
    cnt = 0
    for ii in range(srf['resolution']):
        for ss in range(1 + ii * 5):
            phi = ss * 2 / (1 + ii * 5)
            r = dr * ii
            xx = np.sin(np.pi * phi) * r
            yy = np.cos(np.pi * phi) * r
            zz = r**2 / (R * (1 + np.sqrt(1 - (1 + k) * r**2 / R**2))) + A2 * r**2 + A4 * r**4 + A6 * r**6 + A8 * r**8 + A10 * r**10 + A12 * r**12
            array_xyz[cnt] = np.array([xx, yy, zz])
            cnt += 1

    # Second pass optimization
    if 'raypoints' in srf:
        # Opposite transformations
        i_points = vtk.vtkPolyData()
        i_points.SetPoints(srf['raypoints'])
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        if 'center' in srf:
            transform.Translate([-p for p in srf['center']])
        if 'rotateWXYZ' in srf:
            transform.RotateWXYZ(-srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
        pgt = vtk.vtkTransformPolyDataFilter()
        pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
        pgt.SetTransform(transform)
        pgt.SetInputData(i_points)
        pgt.Update()
        # Get intersection point array
        parray_xyz = ns.vtk_to_numpy(i_points.GetPoints().GetData())
        # Add 2nd pass arrays ath these points
        refine = srf['resolution'] * 100
        res = 4  # was srf['resolution']
        d2r = (srf['diameter']) / 2 / (refine - 1)
        for xyz in parray_xyz:
            cnt = 0
            rxy = np.hypot(xyz[0], xyz[1])
            if rxy > d2r * (res + 1):
                phic = np.arctan2(xyz[0], xyz[1])
                r_range = int(np.ceil((rxy / (srf['diameter'] / 2)) * refine))
                # Counter to get size of array
                var = 0
                for ii in range(r_range - res, r_range + res):  # was 10
                    phi_range = int(np.ceil((phic / (2 * np.pi)) * (1 + ii * 5)))
                    for ss in range(phi_range - res, phi_range + res):
                        var += 1
                sizep = [var, 3]
                arr2nd_xyz = np.empty(sizep)
                for ii in range(r_range - res, r_range + res):  # was 10
                    phi_range = int(np.ceil((phic / (2 * np.pi)) * (1 + ii * 5)))
                    for ss in range(phi_range - res, phi_range + res):
                        phi = ss * 2 / (1 + ii * 5)
                        r = d2r * ii
                        xx = np.sin(np.pi * phi) * r
                        yy = np.cos(np.pi * phi) * r
                        zz = r**2 / (R * (1 + np.sqrt(1 - (1 + k) * r**2 / R**2))) + A2 * r**2 + A4 * r**4 + A6 * r**6 + A8 * r**8 + A10 * r**10 + A12 * r**12
                        arr2nd_xyz[cnt] = np.array([xx, yy, zz])
                        cnt += 1
            else:
                sizep = [sum([1 + x * 5 for x in range(srf['resolution'])]), 3]
                arr2nd_xyz = np.empty(sizep)
                cnt = 0
                for ii in range(srf['resolution']):
                    for ss in range(1 + ii * 5):
                        phi = ss * 2 / (1 + ii * 5)
                        r = d2r * ii
                        xx = np.sin(np.pi * phi) * r
                        yy = np.cos(np.pi * phi) * r
                        zz = r**2 / (R * (1 + np.sqrt(1 - (1 + k) * r**2 / R**2))) + A2 * r**2 + A4 * r**4 + A6 * r**6 + A8 * r**8 + A10 * r**10 + A12 * r**12
                        arr2nd_xyz[cnt] = np.array([xx, yy, zz])
                        cnt += 1

            array_xyz = np.vstack((array_xyz, arr2nd_xyz))

        # Delete non unique values
        b = np.ascontiguousarray(array_xyz).view(np.dtype((np.void, array_xyz.dtype.itemsize * array_xyz.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = array_xyz[idx]
        # I need to sort this in spherical coordinates, first phi, then theta then r
        rtp_a = ray.Cartesian2Spherical(unique_a[1:])  # Skip 0,0,0
        rtp_a = np.vstack((np.array([0.0, 0.0, 0.0]), rtp_a))
        # Now sort
        ind = np.lexsort((rtp_a[:, 2], rtp_a[:, 1], rtp_a[:, 0]))  # Sort by a, then by b
        sorted_rtp = rtp_a[ind]
        sorted_xyz = ray.Spherical2Cartesian(sorted_rtp)
    else:
        sorted_xyz = array_xyz
    # numpy array to vtk array
    pcoords = ns.numpy_to_vtk(num_array=sorted_xyz, deep=True, array_type=vtk.VTK_DOUBLE)
    # Shove coordinates in points container
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    points.SetData(pcoords)
    # Create a polydata object
    point_pd = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    point_pd.SetPoints(points)
    # make the delaunay mesh
    delaunay = vtk.vtkDelaunay2D()
    if vtk.VTK_MAJOR_VERSION < 6:
        delaunay.SetInput(point_pd)
    else:
        delaunay.SetInputData(point_pd)

    # delaunay.SetTolerance(0.00001)
    delaunay.Update()
    # Rotate and translate
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    if 'rotateWXYZ' in srf:
        transform.RotateWXYZ(srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
    if 'center' in srf:
        transform.Translate(srf['center'])
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    pgt.SetTransform(transform)
    if vtk.VTK_MAJOR_VERSION < 6:
        pgt.SetInput(delaunay.GetOutput())
    else:
        pgt.SetInputData(delaunay.GetOutput())
    pgt.Update()
    delaunay = pgt
    # Rotate polydata
    return delaunay


def flatcircle(srf):
    # Create rotational filter of a straight line
    dx = (srf['diameter']) / 2 / (srf['resolution'] - 1)  # radius
    # print(dx, dx * srf['resolution'])
    points = vtk.vtkPoints()
    line = vtk.vtkLine()
    lines = vtk.vtkCellArray()
    for ii in range(srf['resolution']):
        xx, yy, zz = dx * ii, 0, 0
        points.InsertNextPoint(xx, yy, zz)
        if ii != (srf['resolution'] - 1):
            line.GetPointIds().SetId(0, ii)
            line.GetPointIds().SetId(1, ii + 1)
            lines.InsertNextCell(line)
    # Create a PolyData
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetLines(lines)
    # Radial extrude polygon
    extrude = vtk.vtkRotationalExtrusionFilter()
    if vtk.VTK_MAJOR_VERSION < 6:
        extrude.SetInput(polygonPolyData)
    else:
        extrude.SetInputData(polygonPolyData)
    extrude.CappingOff()
    extrude.SetResolution(srf['angularresolution'])
    extrude.Update()
    # It would be best to rotate it by 360/res, so simple rays
    # don't hit eges and low res can be used
    rotate = vtk.vtkTransform()
    rotate.RotateWXYZ(180 / srf['angularresolution'], 0, 0, 1)
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetTransform(rotate)
    if vtk.VTK_MAJOR_VERSION < 6:
        pgt.SetInput(extrude.GetOutput())
    else:
        pgt.SetInputData(extrude.GetOutput())
    pgt.Update()
    extrude = pgt
    # stretch, rotate and translate
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    if 'scalex' in srf:
        transform.Scale(srf['scalex'], 1.0, 1.0)
    if 'rotateWXYZ' in srf:
        transform.RotateWXYZ(srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
    if 'center' in srf:
        transform.Translate(srf['center'])
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    pgt.SetTransform(transform)
    if vtk.VTK_MAJOR_VERSION < 6:
        pgt.SetInput(extrude.GetOutput())
    else:
        pgt.SetInputData(extrude.GetOutput())

    pgt.Update()
    extrude = pgt
    # Subdivision filters only work on triangles
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(extrude.GetOutputPort())
    triangles.Update()
    # Create a mapper and actor
    return triangles


def sphere(srf):
    # Create and configure sphere, using delaunay mesh
    dr = (srf['diameter']) / 2 / (srf['resolution'] - 1)  # radius
    R = srf['radius']
    sizep = [sum([1 + x * 5 for x in range(srf['resolution'])]), 3]
    array_xyz = np.empty(sizep)
    cnt = 0
    for ii in range(srf['resolution']):
        for ss in range(1 + ii * 5):
            phi = ss * 2 / (1 + ii * 5)
            r = dr * ii
            xx = np.sin(np.pi * phi) * r
            yy = np.cos(np.pi * phi) * r
            zz = R * (1 - np.sqrt(1 - (r / R)**2))
            array_xyz[cnt] = np.array([xx, yy, zz])
            cnt += 1

    # Second pass optimization
    if 'raypoints' in srf:
        # Opposite transformations
        i_points = vtk.vtkPolyData()
        i_points.SetPoints(srf['raypoints'])
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        if 'center' in srf:
            transform.Translate([-p for p in srf['center']])
        if 'rotateWXYZ' in srf:
            transform.RotateWXYZ(-srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
        pgt = vtk.vtkTransformPolyDataFilter()
        pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
        pgt.SetTransform(transform)
        pgt.SetInputData(i_points)
        if vtk.VTK_MAJOR_VERSION < 6:
            pgt.SetInput(i_points)
        else:
            pgt.SetInputData(i_points)

        pgt.Update()
        # Get intersection point array
        parray_xyz = ns.vtk_to_numpy(pgt.GetOutput().GetPoints().GetData())
        # Add 2nd pass arrays ath these points
        refine = srf['resolution'] * 100
        res = 4  # was srf['resolution']
        d2r = (srf['diameter']) / 2 / (refine - 1)
        for xyz in parray_xyz:
            cnt = 0
            rxy = np.hypot(xyz[0], xyz[1])
            if rxy > d2r * (res + 1):
                phic = np.arctan2(xyz[0], xyz[1])
                r_range = int(np.ceil((rxy / (srf['diameter'] / 2)) * refine))
                # Counter to get size of array
                var = 0
                for ii in range(r_range - res, r_range + res):  # was 10
                    phi_range = int(np.ceil((phic / (2 * np.pi)) * (1 + ii * 5)))
                    for ss in range(phi_range - res, phi_range + res):
                        var += 1

                sizep = [var, 3]
                arr2nd_xyz = np.empty(sizep)
                for ii in range(r_range - res, r_range + res):  # was 10
                    phi_range = int(np.ceil((phic / (2 * np.pi)) * (1 + ii * 5)))
                    for ss in range(phi_range - res, phi_range + res):
                        phi = ss * 2 / (1 + ii * 5)
                        r = d2r * ii
                        xx = np.sin(np.pi * phi) * r
                        yy = np.cos(np.pi * phi) * r
                        zz = R * (1 - np.sqrt(1 - (r / R)**2))
                        arr2nd_xyz[cnt] = np.array([xx, yy, zz])
                        cnt += 1

            else:
                sizep = [sum([1 + x * 5 for x in range(srf['resolution'])]), 3]
                arr2nd_xyz = np.empty(sizep)
                cnt = 0
                for ii in range(srf['resolution']):
                    for ss in range(1 + ii * 5):
                        phi = ss * 2 / (1 + ii * 5)
                        r = d2r * ii
                        xx = np.sin(np.pi * phi) * r
                        yy = np.cos(np.pi * phi) * r
                        zz = R * (1 - np.sqrt(1 - (r / R)**2))
                        arr2nd_xyz[cnt] = np.array([xx, yy, zz])
                        cnt += 1

            array_xyz = np.vstack((array_xyz, arr2nd_xyz))

        # Delete non unique values
        b = np.ascontiguousarray(array_xyz).view(np.dtype((np.void, array_xyz.dtype.itemsize * array_xyz.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = array_xyz[idx]
        # I need to sort this in spherical coordinates, first phi, then theta then r
        rtp_a = ray.Cartesian2Spherical(unique_a[1:])  # Skip 0,0,0
        rtp_a = np.vstack((np.array([0.0, 0.0, 0.0]), rtp_a))
        # Now sort
        ind = np.lexsort((rtp_a[:, 2], rtp_a[:, 1], rtp_a[:, 0]))  # Sort by a, then by b
        sorted_xyz = unique_a[ind]
    else:
        sorted_xyz = array_xyz
    # numpy array to vtk array
    pcoords = ns.numpy_to_vtk(num_array=sorted_xyz, deep=True, array_type=vtk.VTK_DOUBLE)
    # Shove coordinates in points container
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    points.SetData(pcoords)
    # Create a polydata object
    point_pd = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    point_pd.SetPoints(points)
    # Make the delaunay mesh
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(point_pd)
    # delaunay.SetTolerance(0.00001)
    delaunay.Update()
    # Rotate and translate
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    if 'rotateWXYZ' in srf:
        transform.RotateWXYZ(srf['rotateWXYZ'][0], srf['rotateWXYZ'][1:])
    if 'center' in srf:
        transform.Translate(srf['center'])
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    pgt.SetTransform(transform)
    pgt.SetInputData(delaunay.GetOutput())
    pgt.Update()
    delaunay = pgt
    # rotate polydata
    return delaunay


def objectsource(srf1, srf2, ratio=0.8):
    # make points on surface 1 should this be elliptical?
    # Use given points, don't create them, so all poitns on vertices of surf1 surface
    sourcepoints = [srf1['raypoints'].GetPoint(i) for i in range(srf1['raypoints'].GetNumberOfPoints())]
    # Make points on target
    targetlist = [[0.0, 0.0, 0.0]]
    if 'diameter' in srf2:
        targetlist.append([ratio * srf2['diameter'] / 2, 0.0, 0.0])
        targetlist.append([0.0, ratio * srf2['diameter'] / 2, 0.0])
        targetlist.append([-ratio * srf2['diameter'] / 2, 0.0, 0.0])
        targetlist.append([0.0, -ratio * srf2['diameter'] / 2, 0.0])
    elif 'width' in srf2 and 'height' in srf2:
        targetlist.append([ratio * srf2['width'] / 2, 0.0, 0.0])
        targetlist.append([0.0, ratio * srf2['height'] / 2, 0.0])
        targetlist.append([-ratio * srf2['width'] / 2, 0.0, 0.0])
        targetlist.append([0.0, -ratio * srf2['height'] / 2, 0.0])
    else:
        print('Could not make targetlist in objectsource')
        return
    # Transform points, I'm going to cheat and use the vtk functions
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for tl in targetlist:
        points.InsertNextPoint(tl)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    if 'scalex' in srf2:
        transform.Scale(srf2['scalex'], 1.0, 1.0)
    if 'rotateWXYZ' in srf2:
        transform.RotateWXYZ(srf2['rotateWXYZ'][0], srf2['rotateWXYZ'][1:])
    if 'centerH' in srf2:
        transform.Translate(srf2['centerH'])
    pgt = vtk.vtkTransformPolyDataFilter()
    pgt.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    pgt.SetTransform(transform)
    pgt.SetInputData(polydata)
    pgt.Update()
    # And now I'm going to extract the points again
    targetpoints = [pgt.GetOutput().GetPoint(i) for i in range(pgt.GetOutput().GetNumberOfPoints())]
    # Get normal vector from source to target, 5 vectors per point
    object_points = vtk.vtkPoints()
    object_points.SetDataTypeToDouble()
    object_normalvectors = vtk.vtkDoubleArray()
    object_normalvectors.SetNumberOfComponents(3)
    for sp in sourcepoints:
        for tp in targetpoints:
            vec = (tp[0] - sp[0], tp[1] - sp[1], tp[2] - sp[2])
            object_normalvectors.InsertNextTuple(list(vec / np.linalg.norm(vec)))
            object_points.InsertNextPoint(sp)

    object_polydata = vtk.vtkPolyData()
    object_polydata.SetPoints(object_points)
    object_polydata.GetPointData().SetNormals(object_normalvectors)
    return object_polydata, object_points


def pointsource(srf, simple=True):
    if simple:
        anglex = srf['anglex'] / 180 * np.pi
        angley = srf['angley'] / 180 * np.pi
        tuples = [(0, 0, 1)]
        phix = [1, 0, -1, 0]
        phiy = [0, 1, 0, -1]
        theta = [anglex, angley, anglex, angley]
        x = phix * np.sin(theta)
        y = phiy * np.sin(theta)
        z = np.cos(theta)
        tuples = tuples + ([(xx, yy, zz) for xx, yy, zz in zip(x, y, z)])
        # for tp in tuples:  #has to be one
        #     print(np.sqrt(tp[0]**2+tp[1]**2+tp[2]**2))
    else:
        res = [4, 6, 8]
        anglex = srf['anglex'] / 180 * np.pi
        angley = srf['angley'] / 180 * np.pi
        # Center line
        tuples = [(0, 0, 1)]
        for rr in res:
            # Define pointsource, cylindrical
            thetax = (res.index(rr) + 1) / len(res) * anglex
            thetay = (res.index(rr) + 1) / len(res) * angley
            phi = np.arange(rr) * (2 * np.pi / rr)
            theta = thetax * thetay / np.hypot(thetay * np.cos(phi), thetax * np.sin(phi))
            x = np.cos(phi) * np.sin(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(theta)
            tuples = tuples + ([(xx, yy, zz) for xx, yy, zz in zip(x, y, z)])
            # plt.plot(x,y)
    intersection_points = vtk.vtkPoints()
    intersection_points.SetDataTypeToDouble()
    normal_vectors = vtk.vtkDoubleArray()
    normal_vectors.SetNumberOfComponents(3)

    for sp in srf['sourcepoints']:
        for tp in tuples:
            normal_vectors.InsertNextTuple(tp)
            intersection_points.InsertNextPoint(sp)

    normal_polydata = vtk.vtkPolyData()
    normal_polydata.SetPoints(intersection_points)
    normal_polydata.GetPointData().SetNormals(normal_vectors)
    return normal_polydata, intersection_points


def surfaceActor(ren, appendFilter, srf):
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(srf['surface'].GetOutput())
    # mapper.SetInputConnection(delaunay.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor([0.0, 0.0, 1.0])       # set color to blue
    actor.GetProperty().EdgeVisibilityOn()              # show edges/wireframe
    actor.GetProperty().SetEdgeColor([1.0, 1.0, 1.0])   # render edges as white
    # Plot and save
    ren.AddActor(actor)
    ren.ResetCamera()
    appendFilter.AddInputData(srf['surface'].GetOutput())
    appendFilter.Update()


def shape(ren, appendFilter, srf, addActor=False):
    if srf['shape'] == 'sphere':
        srf['surface'] = sphere(srf)
    elif srf['shape'] == 'flat':
        srf['surface'] = flat(srf)
    elif srf['shape'] == 'flatcircle':
        srf['surface'] = flatcircle(srf)
    elif srf['shape'] == 'asphere':
        srf['surface'] = asphere(srf)
    elif srf['shape'] == 'cylinder':
        srf['surface'] = cylinder(srf)
    elif srf['shape'] == 'pointsource':
        pass
    else:
        print('Couldnt understand shape')
        return
    if addActor:
        surfaceActor(ren, appendFilter, srf)
    return srf


def trace(ren, appendFilter, srf1, srf2, addActor=True):
    obbsurf2 = vtk.vtkOBBTree()
    obbsurf2.SetDataSet(srf2['surface'].GetOutput())
    obbsurf2.BuildLocator()
    # I dont know from where the ray is coming, use 'curv' for this as a hack
    if srf2['shape'] == "sphere":
        if srf2['curv'] == 'positive':
            Flip = False
        elif srf2['curv'] == 'negative':
            Flip = True
        else:
            print('Misunderstood curv in trace')
    elif srf2['shape'] == 'asphere':
        if srf2['rn'] >= srf1['rn']:
            Flip = False
        elif srf2['rn'] < srf1['rn']:
            Flip = True
    elif srf2['shape'] == 'cylinder':
        if srf2['rn'] >= srf1['rn']:
            Flip = False
        elif srf2['rn'] < srf1['rn']:
            Flip = True
    elif srf2['shape'] == 'flat':
        if srf2['rn'] >= srf1['rn']:
            Flip = False
        elif srf2['rn'] < srf1['rn']:
            Flip = True
    else:
        Flip = False

    srf2['normals'], srf2['normalpoints'] = getnormals(srf2['surface'], flip=Flip)
    # #Sometimes, something goes wrong with the number of cells
    # count1 = srf2['normals'].GetCellData().GetNormals().GetNumberOfTuples()
    # count2 = obbsurf2.GetDataSet().GetNumberOfCells()
    # assert count1 == count2, 'The number of normals does not match the number of cells in the obbtree'
    # where intersections are found
    intersection_points = vtk.vtkPoints()
    intersection_points.SetDataTypeToDouble()
    # normal vectors at intersection
    normal_vectors = vtk.vtkDoubleArray()
    normal_vectors.SetNumberOfComponents(3)
    # normals of refracted vectors
    reflect_vectors = vtk.vtkDoubleArray()
    reflect_vectors.SetNumberOfComponents(3)
    # Loop through all of surface1 cell-centers
    for idx in range(srf1['raypoints'].GetNumberOfPoints()):
        # Get coordinates of surface1 cell center
        pointSurf1 = srf1['raypoints'].GetPoint(idx)
        # Get incident vector at that cell
        normalsurf1 = srf1['rays'].GetPointData().GetNormals().GetTuple(idx)
        # Calculate the 'target' of the ray based on 'RayCastLength'
        pointRaySurf2 = list(np.array(pointSurf1) + ray.RayCastLength * np.array(normalsurf1))
        # Check if there are any intersections for the given ray
        if ray.isHit(obbsurf2, pointSurf1, pointRaySurf2):
            # Retrieve coordinates of intersection points and intersected cell ids
            pointsInter, cellIdsInter = ray.getIntersect(obbsurf2, pointSurf1, pointRaySurf2)
            # print(cellIdsInter)
            # ray.addPoint(ren, False, pointsInter[0], [0.5, 0.5, 0.5])
            # Render lines/rays emanating from the Source. Rays that intersect are
            if addActor:
                ray.addLine(ren, appendFilter, pointSurf1, pointsInter[0], [1.0, 1.0, 0.0], opacity=0.5)
            # Insert the coordinates of the intersection point in the dummy container
            intersection_points.InsertNextPoint(pointsInter[0])
            # Get the normal vector at the surf2 cell that intersected with the ray
            normalsurf2 = srf2['normals'].GetPointData().GetNormals().GetTuple(cellIdsInter[0])
            # Insert the normal vector of the intersection cell in the dummy container
            normal_vectors.InsertNextTuple(normalsurf2)
            # Calculate the incident ray vector
            vecInc2 = np.array(pointRaySurf2) - np.array(pointSurf1)
            vecInc = list(vecInc2 / np.linalg.norm(vecInc2))
            # Calculate the reflected ray vector
            if srf2['type'] == 'lens':
                vecRef = ray.calcVecRefract(vecInc / np.linalg.norm(vecInc), normalsurf2, srf1['rn'], srf2['rn'])  # refract
            elif srf2['type'] == 'stop' or 'mirror' or 'source':
                vecRef = ray.calcVecReflect(vecInc / np.linalg.norm(vecInc), normalsurf2)  # reflect
            # Add to container
            reflect_vectors.InsertNextTuple(vecRef)

    # store intersection points
    # intersection_points.Update()
    # Create a dummy 'vtkPolyData' to store refracted vecs
    reflect_polydata = vtk.vtkPolyData()
    reflect_polydata.SetPoints(intersection_points)
    reflect_polydata.GetPointData().SetNormals(reflect_vectors)
    # Create a dummy 'vtkPolyData' to store normal vecs
    normal_polydata = vtk.vtkPolyData()
    normal_polydata.SetPoints(intersection_points)
    normal_polydata.GetPointData().SetNormals(normal_vectors)
    return intersection_points, reflect_polydata, normal_polydata


def run(surfaces, project, Directory, scene, refine=True, plot=True):
    # Write output to vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    filename = Directory + project + "%04d.vtp" % scene
    writer.SetFileName(filename)
    appendFilter = vtk.vtkAppendPolyData()
    # Create a render window
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 600)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    # Set camera position
    camera = ren.MakeCamera()
    camera.SetPosition(0, 100, 50)
    camera.SetFocalPoint(0.0, 0, 0.0)
    camera.SetViewAngle(0.0)
    camera.SetParallelProjection(1)
    ren.SetActiveCamera(camera)

    traceit = range(len(surfaces))
    for tri in traceit:
        # Surface one is source
        if tri == 0:
            assert surfaces[tri]['type'] == 'source', 'surface zero needs to be source'
            if surfaces[tri]['shape'] == 'point':  # Use point source
                surfaces[tri]['rays'], surfaces[tri]['raypoints'] = pointsource(surfaces[0], simple=True)
            else:  # Use object source
                surfaces[tri] = shape(ren, appendFilter, surfaces[tri], addActor=True)
                surfaces[tri]['rays'], surfaces[tri]['raypoints'] = getnormals(surfaces[tri]['surface'], vertices=True, flip=False)
                surfaces[tri]['rays'], surfaces[tri]['raypoints'] = objectsource(surfaces[tri], surfaces[tri + 1], ratio=0.3)
                glyphsa = glyphs(surfaces[tri]['rays'], color=[0.0, 1.0, 0.0])  # Green
                ren.AddActor(glyphsa)
                renWin.Render()
        elif tri == len(surfaces) - 1:
            surfaces[tri] = shape(ren, appendFilter, surfaces[tri], addActor=False)     # TODO should be True
            surfaces[tri]['raypoints'] = stop(ren, appendFilter, surfaces[tri - 1], surfaces[tri], scene)
            renWin.Render()
            print('Tracing {0} and {1}'.format(tri - 1, tri))
        else:
            if refine:  # If refine, shape again, trace again
                surfaces[tri] = shape(ren, appendFilter, surfaces[tri], addActor=False)
                surfaces[tri]['raypoints'], surfaces[tri]['rays'], surfaces[tri]['normals'] = trace(ren, appendFilter, surfaces[tri - 1], surfaces[tri], addActor=False)
            surfaces[tri] = shape(ren, appendFilter, surfaces[tri], addActor=True)
            surfaces[tri]['raypoints'], surfaces[tri]['rays'], surfaces[tri]['normals'] = trace(ren, appendFilter, surfaces[tri - 1], surfaces[tri])
            # Plot glyphs
            glyphsa = glyphs(surfaces[tri]['rays'], color=[0.0, 1.0, 0.0], size=1)  # Green
            ren.AddActor(glyphsa)
            glyphsc = glyphs(surfaces[tri]['normals'], color=[0.0, 0.0, 1.0], size=1)   # Blue
            ren.AddActor(glyphsc)
            renWin.Render()
            print('Tracing {0} and {1}'.format(tri - 1, tri))

    # Write output to vtp file
    # appendFilter.Update()
    # polydatacontainer = appendFilter
    # writer.SetInputData(polydatacontainer.GetOutput())
    # writer.Write()
    #
    # Check results in viewer, by exit screen, proceed
    if plot:
        iren.Start()


def main():
    LBF_254_050 = {'tc': 6.5,
                   'f': 50.0,
                   'fb': 46.4,
                   'R1': -172,
                   'R2': 30.1,
                   'rn': 1.5168,
                   'diameter': 25.4}
    LK1037L1 = {'f': -19.0,
                'fb': -25.30,  # is -20.3
                'R1': 'flat',
                'R2': 9.8,
                'tc': 2.0,
                'rn': 1.5168,
                'width': 19.0,
                'height': 21.0}  # extrude over height
    LJ1309L1 = {'f': 200.0,
                'fb': 189.5,  # is -20.3
                'R1': 'flat',
                'R2': 103.4,
                'tc': 15.9,
                'rn': 1.5168,
                'width': 100,
                'height': 90}  # extrude over height
    LJ1728L1 = {'f': 50.99,
                'fb': 36.7,  # is -20.3
                'R1': 'flat',
                'R2': 26.4,
                'tc': 21.6,
                'rn': 1.5168,
                'width': 50.5,
                'height': 52.7}  # extrude over height

    so1h = 100
    so1v = 52.1239
    si = 2000
    angle = 2.0  # graden

    surfaces = [{'type': 'source',
                 'shape': 'flat',
                 'center': [np.sin(angle / 180 * np.pi) * si, 0.0, si - np.cos(angle / 180 * np.pi) * si],
                 'width': 2 * np.tan(0.1 / 180 * np.pi) * si,    #
                 'height': 2 * np.tan(4.44 / 180 * np.pi) * si,  # could be anything really
                 'rotateWXYZ': [-angle, 0, 1, 0],                # Normal is [0, 0, -1]
                 'rn': 1.0},
                {'type': 'lens',
                 'shape': 'cylinder',
                 'centerH': [0.0, 0.0, si - so1h],
                 'center': [0.0, 0.0, si - (so1h - (LJ1309L1['f'] - LJ1309L1['fb']) + LJ1309L1['tc'])],
                 # 'rotateWXYZ':[90.0, 0, 0, 1], # Normal is [0, 0, -1] in graden
                 'width': LJ1309L1['width'],
                 'height': LJ1309L1['height'],
                 'resolution': 1000,
                 'rn': LJ1309L1['rn'],  # n-bk7
                 'curvx': LJ1309L1['R2']},
                {'type': 'lens',
                 'shape': 'flat',
                 'centerH': [0.0, 0.0, si - so1h],  # 2.76 = 4.36
                 'center': [0.0, 0.0, si - (so1h - (LJ1309L1['f'] - LJ1309L1['fb']))],  # 2.91
                 'width': LJ1309L1['width'],
                 'height': LJ1309L1['height'],
                 # 'rotateWXYZ':[90, 0, 0, 1], # Normal is [0, 0, -1]
                 'rn': 1.0},
                {'type': 'lens',
                 'shape': 'cylinder',
                 'centerH': [0.0, 0.0, si - so1v],
                 'center': [0.0, 0.0, si - (so1v - (LJ1728L1['f'] - LJ1728L1['fb']) + LJ1728L1['tc'])],
                 'rotateWXYZ':[90.0, 0, 0, 1],  # Normal is [0, 0, -1] in graden
                 'width': LJ1728L1['width'],
                 'height': LJ1728L1['height'],
                 'resolution': 1000,
                 'rn': LJ1728L1['rn'],  # n-bk7
                 'curvx': LJ1728L1['R2']},
                {'type': 'lens',
                 'shape': 'flat',
                 'centerH': [0.0, 0.0, si - so1v],  # 2.76 = 4.36
                 'center': [0.0, 0.0, si - (so1v - (LJ1728L1['f'] - LJ1728L1['fb']))],  # 2.91
                 'width': LJ1728L1['width'],
                 'height': LJ1728L1['height'],
                 'rotateWXYZ':[90, 0, 0, 1],  # Normal is [0, 0, -1]
                 'rn': 1.0},
                {'type': 'stop',
                 'shape': 'flat',
                 'center': [0.0, 0.0, si],
                 # 'rotateWXYZ': [45, 0, 1, 0],  # Normal is [0, 0, -1]
                 'width': 25.0,
                 'height': 25.0,
                 'rn': 1.0}]

    import os
    project = 'receiverB'
    Directory = os.getcwd()
    run(surfaces, project, Directory, 0, plot=True)

if __name__ == "__main__":
    main()
