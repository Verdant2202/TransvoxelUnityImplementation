using System.Collections.Generic;
using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System;
public static class MarchingCubes
{
    public static void Execute(float[,,] voxels, Mesh.MeshDataArray meshDataArray, int LOD, float voxelScale)
    {
        //Marching Cubes Start
        int step = 1 << LOD;
        CreateRegularMesh(voxels, voxelScale, step, meshDataArray[0]);

        //Transvoxel Start, if you dont like it, remove anything below and any functions it uses.
        if(LOD == 0)
        {
            return;
        }
        int xSize = voxels.GetLength(0); //length of voxel array, so cubes length are size - 1.
        int ySize = voxels.GetLength(1);
        int zSize = voxels.GetLength(2);
        float transitionCellSize = 0.0f; //2 - full step, 1 - half step. Value should be below 1, probably best is 0.5. Also it requires to use shader code to pull in vertices of main mesh.
        CreateTransitionFace(voxels, voxelScale, step, new int3(xSize-1, 0, 0), new AxisTriplet(Axis.X), TransitionFaceDirection.Positive, meshDataArray[1], transitionCellSize);
        CreateTransitionFace(voxels, voxelScale, step, new int3(0, 0, 0), new AxisTriplet(Axis.X), TransitionFaceDirection.Negative, meshDataArray[2], transitionCellSize);
        CreateTransitionFace(voxels, voxelScale, step, new int3(0, ySize-1, 0), new AxisTriplet(Axis.Y), TransitionFaceDirection.Positive, meshDataArray[3], transitionCellSize);
        CreateTransitionFace(voxels, voxelScale, step, new int3(0, 0, 0), new AxisTriplet(Axis.Y), TransitionFaceDirection.Negative, meshDataArray[4], transitionCellSize);
        CreateTransitionFace(voxels, voxelScale, step, new int3(0, 0, zSize-1), new AxisTriplet(Axis.Z), TransitionFaceDirection.Positive, meshDataArray[5], transitionCellSize);
        CreateTransitionFace(voxels, voxelScale, step, new int3(0, 0, 0), new AxisTriplet(Axis.Z), TransitionFaceDirection.Negative, meshDataArray[6], transitionCellSize);
    }

    public static void SetMeshData(Mesh.MeshData meshData, List<Vector3> vertices, List<int> triangles)
    {
        // 1. Setup vertex buffer: positions as Vector3 (float3)
        meshData.SetVertexBufferParams(vertices.Count,
            new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3));

        // 2. Setup index buffer
        meshData.SetIndexBufferParams(triangles.Count, IndexFormat.UInt16);

        // 3. Copy vertex data
        var vertexData = meshData.GetVertexData<Vector3>();
        for (int i = 0; i < vertices.Count; i++)
        {
            vertexData[i] = vertices[i];
        }

        // 4. Copy index data (triangles)
        // Note: since your triangles are ints, but mesh expects ushort if UInt16 is used,
        // you must ensure no index > 65535. Otherwise use UInt32 and uint.
        var indexData = meshData.GetIndexData<ushort>();
        for (int i = 0; i < triangles.Count; i++)
        {
            indexData[i] = (ushort)triangles[i];
        }

        // 5. Setup submesh info (one submesh here)
        meshData.subMeshCount = 1;
        meshData.SetSubMesh(0, new SubMeshDescriptor(0, triangles.Count));
    }

    //Transvoxel functions

    public static void CreateTransitionFace(float[,,] voxels, float voxelScale, int step, int3 startPosition, 
        AxisTriplet axisTriplet, TransitionFaceDirection faceDirection, Mesh.MeshData meshData, float transitionCellSize)
    {
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();

        int xSize = voxels.GetLength(0);
        int ySize = voxels.GetLength(1);
        int zSize = voxels.GetLength(2);

        int firstSize = axisTriplet.First == Axis.X ? xSize : axisTriplet.First == Axis.Y ? ySize : zSize;
        int secondSize = axisTriplet.Second == Axis.X ? xSize : axisTriplet.Second == Axis.Y ? ySize : zSize;
        //+1 to account for -1 values
        int[,,] vertexSharingIndexes = new int[firstSize + 1, secondSize + 1, 7];
        for (int i = 0; i < firstSize + 1; i++)
        {
            for(int j = 0; j < secondSize + 1; j++)
            {
                for(int k = 0; k < 7; k++)
                {
                    vertexSharingIndexes[i, j, k] = -1;
                }
            }
        }

        int3 current = startPosition;
        int3 maxBounds = new int3(xSize - 1, ySize - 1, zSize - 1);

        while (true)
        {
            int3 cubePosition = current;
            float[] cubeCorners = new float[13];
            Vector3[] transitionCornerOffset = new Vector3[13];

            SetTransitionCubeCornersValues(cubeCorners, voxels, cubePosition, step, axisTriplet);
            SetTransitionCornerOffset(transitionCornerOffset, transitionCellSize, axisTriplet, faceDirection);
            ProcessTransitionCube(cubePosition, cubeCorners, 0f, voxelScale, step,
                                vertices, triangles, vertexSharingIndexes,
                                transitionCornerOffset, axisTriplet);

            // Move to next cell in 2D grid
            current.Set(axisTriplet.First, current.Get(axisTriplet.First) + step);
            if (current.Get(axisTriplet.First) >= maxBounds.Get(axisTriplet.First))
            {
                current.Set(axisTriplet.First, startPosition.Get(axisTriplet.First));
                current.Set(axisTriplet.Second, current.Get(axisTriplet.Second) + step);

                if (current.Get(axisTriplet.Second) >= maxBounds.Get(axisTriplet.Second))
                    break;
            }
        }

        SetMeshData(meshData, vertices, triangles);
    }

    public static void ProcessTransitionCube(int3 cubePosition, float[] corners, float isoLevel, float scale, int step,
       List<Vector3> vertices, List<int> triangles, int[,,] vertexSharingIndexes, Vector3[] transitionCornerOffset, AxisTriplet axisTriplet)
    {
        int cubeIndex = CalculateTransitionCubeIndex(corners, isoLevel);

        int transitionCellClassCode = MarchingCubesTables.transitionCellClass[cubeIndex];
        bool reverseWinding = (transitionCellClassCode & 0x80) != 0;
        TransitionCellData cellData = MarchingCubesTables.transitionCellData[transitionCellClassCode & 0x7F];

        int[] vertexIndexes = new int[cellData.GetVertexCount()];

        for (int i = 0; i < cellData.GetVertexCount(); i++)
        {
            int vertexCode = MarchingCubesTables.TransitionVertexData[cubeIndex][i];
            int3 ownerCubePosition = GetOwnerTransitionCubePosition(cubePosition, vertexCode, step, axisTriplet);
            int vertexIndexToReuse = (vertexCode & (0x0F00)) >> 8;
            int vFirst = ownerCubePosition.Get(axisTriplet.First) / step + 1;
            int vSecond = ownerCubePosition.Get(axisTriplet.Second) / step + 1;
            int directionNibble = ((vertexCode >> 12) & 0xF); //4 = create but not own, 8 = create and own

            if (vertexSharingIndexes[vFirst, vSecond, vertexIndexToReuse] == -1 || directionNibble == 4) //directionNibble == 4 means its inside the cube, and we have to create a vertex there
            {
                //Creating a vertex
                int firstVertex = vertexCode & (0x000F);
                int secondVertex = (vertexCode & (0x00F0)) >> 4;
                vertices.Add(InterpolateTransitionEdge(cubePosition, scale, step, firstVertex, secondVertex, corners, transitionCornerOffset));
                vertexIndexes[i] = vertices.Count - 1;
                if(directionNibble == 8)
                {
                    vertexSharingIndexes[vFirst, vSecond, vertexIndexToReuse] = vertices.Count - 1;
                }
            }
            else
            {
                //Reusing a vertex
                int vertexIndex = vertexSharingIndexes[vFirst, vSecond, vertexIndexToReuse];
                vertexIndexes[i] = vertexIndex;
            }
        }

        for (int i = 0; i < cellData.vertices.Length; i += 3)
        {
            if (reverseWinding)
            {
                triangles.Add(vertexIndexes[cellData.vertices[i]]);
                triangles.Add(vertexIndexes[cellData.vertices[i + 1]]);
                triangles.Add(vertexIndexes[cellData.vertices[i + 2]]);
            }
            else
            {
                triangles.Add(vertexIndexes[cellData.vertices[i]]);
                triangles.Add(vertexIndexes[cellData.vertices[i + 2]]);
                triangles.Add(vertexIndexes[cellData.vertices[i + 1]]);
            }
        }
    }

    public static void SetTransitionCornerOffset(Vector3[] transitionCornerOffset, float transitionCellSize, AxisTriplet axisTriplet, TransitionFaceDirection faceDirection)
    {
        float sign = faceDirection == TransitionFaceDirection.Positive ? 1f : -1f;

        // 2D grid positions (first x, second y), fixed axis is constant
        int2[] offsets2D = new int2[]
        {
        new int2(0, 0),
        new int2(1, 0),
        new int2(2, 0),
        new int2(0, 1),
        new int2(1, 1),
        new int2(2, 1),
        new int2(0, 2),
        new int2(1, 2),
        new int2(2, 2),
        };

        // Inner 3x3 face
        for (int i = 0; i < 9; i++)
        {
            Vector3 v = Vector3.zero;
            v[(int)axisTriplet.First] = offsets2D[i].x;
            v[(int)axisTriplet.Second] = offsets2D[i].y;
            // Fixed axis remains at 0
            transitionCornerOffset[i] = v;
        }

        // Outer corners on the adjacent face
        int2[] outer2D = new int2[]
        {
        new int2(0, 0),
        new int2(2, 0),
        new int2(0, 2),
        new int2(2, 2),
        };

        for (int i = 0; i < 4; i++)
        {
            Vector3 v = Vector3.zero;
            // Then use this to get outer2D coords:
            v[(int)axisTriplet.First] = outer2D[i].x;
            v[(int)axisTriplet.Second] = outer2D[i].y;
            v[(int)axisTriplet.Fixed] = sign * transitionCellSize;
            transitionCornerOffset[i+9] = v;
        }
    }

    public static void SetTransitionCubeCornersValues(float[] cubeCorners, float[,,] voxels, int3 cubePosition, int step, AxisTriplet axisTriplet)
    {
        int halfStep = step / 2;

        // Offsets in 2D grid (firstAxis x, secondAxis y), fixedAxis stays constant
        int2[] offsets2D = new int2[]
        {
            new int2(0, 0),
            new int2(halfStep, 0),
            new int2(step, 0),
            new int2(0, halfStep),
            new int2(halfStep, halfStep),
            new int2(step, halfStep),
            new int2(0, step),
            new int2(halfStep, step),
            new int2(step, step),
        };

        for (int i = 0; i < 9; i++)
        {
            int3 pos = cubePosition;
            pos.Set(axisTriplet.First, pos.Get(axisTriplet.First) + offsets2D[i].x);
            pos.Set(axisTriplet.Second, pos.Get(axisTriplet.Second) + offsets2D[i].y);
            // Fixed axis stays unchanged
            cubeCorners[i] = voxels[pos.x, pos.y, pos.z];
        }

        // These are reused corners (transition face rules)
        cubeCorners[9] = cubeCorners[0];
        cubeCorners[10] = cubeCorners[2];
        cubeCorners[11] = cubeCorners[6];
        cubeCorners[12] = cubeCorners[8];
    }

    public static int CalculateTransitionCubeIndex(float[] corners, float isoLevel)
    {
        int idx = 0;
        if (corners[0] < isoLevel) idx |= 0x1;
        if (corners[1] < isoLevel) idx |= 0x2;
        if (corners[2] < isoLevel) idx |= 0x4;
        if (corners[3] < isoLevel) idx |= 0x80;
        if (corners[4] < isoLevel) idx |= 0x100;
        if (corners[5] < isoLevel) idx |= 0x08;
        if (corners[6] < isoLevel) idx |= 0x40;
        if (corners[7] < isoLevel) idx |= 0x20;
        if (corners[8] < isoLevel) idx |= 0x10;

        return idx;
    }

    private static Vector3 InterpolateTransitionEdge(int3 cubePosition, float scale, int step, int a, int b, float[] corners, Vector3[] transitionCornerOffset)
    {
        int halfStep = step / 2;
        Vector3 position = new Vector3(cubePosition.x, cubePosition.y, cubePosition.z) * scale;
        float t = (0 - corners[a]) / (corners[b] - corners[a]);
        return position + Vector3.Lerp(
            transitionCornerOffset[a] * scale * halfStep,
            transitionCornerOffset[b] * scale * halfStep,
            t
        );
    }

    private static int3 GetOwnerTransitionCubePosition(int3 cubePosition, int vertexCode, int step, AxisTriplet triplet)
    {
        int direction = (vertexCode >> 12) & 0xF;
        int3 result = cubePosition;

        if ((direction & 0x1) != 0)
            result.Set(triplet.First, result.Get(triplet.First) - step);
        if ((direction & 0x2) != 0)
            result.Set(triplet.Second, result.Get(triplet.Second) - step);

        return result;
    }

    //Transvoxel functions end

    //Regular cube functions

    private static void CreateRegularMesh(float[,,] voxels, float voxelScale, int step, Mesh.MeshData meshData)
    {

        int xSize = voxels.GetLength(0); //length of voxel array, so cubes length are size - 1.
        int ySize = voxels.GetLength(1);
        int zSize = voxels.GetLength(2);

        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();

        //Initialize the array with -1, -1 means a vertex isnt created there.
        int[,,,] vertexSharingIndexes = new int[xSize, ySize, zSize, 3];
        for (int i = 0; i < xSize; i++)
        {
            for (int j = 0; j < ySize; j++)
            {
                for (int k = 0; k < zSize; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        vertexSharingIndexes[i, j, k, l] = -1;
                    }
                }
            }
        }
        //March every cube
        for (int z = 0; z < zSize - 1; z += step)
        {
            for (int y = 0; y < ySize - 1; y += step)
            {
                for (int x = 0; x < xSize - 1; x += step)
                {
                    float[] cubeCorners = new float[8];
                    cubeCorners[0] = voxels[x, y, z];
                    cubeCorners[1] = voxels[x + step, y, z];
                    cubeCorners[2] = voxels[x, y + step, z];
                    cubeCorners[3] = voxels[x + step, y + step, z];
                    cubeCorners[4] = voxels[x, y, z + step];
                    cubeCorners[5] = voxels[x + step, y, z + step];
                    cubeCorners[6] = voxels[x, y + step, z + step];
                    cubeCorners[7] = voxels[x + step, y + step, z + step];

                    ProcessRegularCube(
                        new int3(x, y, z),
                        cubeCorners,
                        0f,
                    voxelScale,
                        step,
                        vertices,
                        triangles,
                        vertexSharingIndexes
                        );
                }
            }
        }
        SetMeshData(meshData, vertices, triangles);
    }
    private static void ProcessRegularCube(int3 cubePosition, float[] corners, float isolevel, float scale, int step,
                                List<Vector3> vertices, List<int> triangles, int[,,,] vertexSharingIndexes)
    {
        int cubeIndex = CalculateRegularCubeIndex(corners, isolevel);
        if (cubeIndex == 0 || cubeIndex == 255) return;

        RegularCellData cellData = MarchingCubesTables.regularCellData[MarchingCubesTables.RegularCellClass[cubeIndex]];
        int[] vertexIndexes = new int[cellData.GetVertexCount()];

        for (int i = 0; i < cellData.GetVertexCount(); i++)
        {
            int vertexCode = MarchingCubesTables.RegularVertexData[cubeIndex][i];
            int3 ownerCubePosition = GetOwnerRegularCubePosition(cubePosition, vertexCode, step);
            int vertexIndexToReuse = (vertexCode & (0x0F00)) >> 8;

            //Add one because we can have a -1 position.
            int vx = ownerCubePosition.x / step + 1;
            int vy = ownerCubePosition.y / step + 1;
            int vz = ownerCubePosition.z / step + 1;
            if (vertexSharingIndexes[vx, vy, vz, vertexIndexToReuse] == -1)
            {
                //Create the vertex and make owner cube own it.
                int firstVertex = vertexCode & (0x000F);
                int secondVertex = (vertexCode & (0x00F0)) >> 4;
                vertices.Add(InterpolateRegularEdge(cubePosition, scale, step, firstVertex, secondVertex, corners));
                vertexIndexes[i] = vertices.Count - 1;
                vertexSharingIndexes[vx, vy, vz, vertexIndexToReuse] = vertices.Count - 1;
            }
            else
            {
                //Reuse the vertex from the owner cube
                int vertexIndex = vertexSharingIndexes[vx, vy, vz, vertexIndexToReuse];
                vertexIndexes[i] = vertexIndex;
            }
        }

        for (int i = 0; i < cellData.vertices.Length; i += 3)
        {
            triangles.Add(vertexIndexes[cellData.vertices[i]]);
            triangles.Add(vertexIndexes[cellData.vertices[i + 1]]);
            triangles.Add(vertexIndexes[cellData.vertices[i + 2]]);
        }
    }

    private static int CalculateRegularCubeIndex(float[] corners, float isolevel)
    {
        int index = 0;
        for (int i = 0; i < 8; i++)
        {
            if (corners[i] < isolevel) index |= 1 << i;
        }
        return index;
    }

    private static Vector3 InterpolateRegularEdge(int3 cubePosition, float scale, int step, int a, int b, float[] corners)
    {
        Vector3 position = new Vector3(cubePosition.x, cubePosition.y, cubePosition.z) * scale;
        float t = (0 - corners[a]) / (corners[b] - corners[a]);
        return position + Vector3.Lerp(
            (Vector3)MarchingCubesTables.RegularCornerOffset[a] * scale * step,
            (Vector3)MarchingCubesTables.RegularCornerOffset[b] * scale * step,
            t
        );
    }

    public static int3 GetOwnerRegularCubePosition(int3 cubePosition, int vertexCode, int step)
    {
        int direction = (vertexCode >> 12) & 0xF; // Direction Nibble

        int3 result = cubePosition;

        if ((direction & 0x1) != 0) result.x -= step; // Bit 0 = X
        if ((direction & 0x2) != 0) result.y -= step; // Bit 1 = Y
        if ((direction & 0x4) != 0) result.z -= step; // Bit 2 = Z

        return result;
    }
}

public enum Axis { X, Y, Z }
public enum TransitionFaceDirection { Positive, Negative }

public readonly struct AxisTriplet
{
    public readonly Axis Fixed;
    public readonly Axis First;
    public readonly Axis Second;

    public AxisTriplet(Axis fixedAxis)
    {
        Fixed = fixedAxis;
        switch (fixedAxis)
        {
            case Axis.X:
                First = Axis.Y;
                Second = Axis.Z;
                break;
            case Axis.Y:
                First = Axis.X;
                Second = Axis.Z;
                break;
            case Axis.Z:
                First = Axis.X;
                Second = Axis.Y;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(fixedAxis), fixedAxis, null);
        }
    }
}

public static class Int3Extensions
{
    public static int Get(this int3 v, Axis axis)
    {
        return axis switch
        {
            Axis.X => v.x,
            Axis.Y => v.y,
            Axis.Z => v.z,
            _ => throw new ArgumentOutOfRangeException(nameof(axis), axis, null)
        };
    }

    public static void Set(this ref int3 v, Axis axis, int value)
    {
        switch (axis)
        {
            case Axis.X: v.x = value; break;
            case Axis.Y: v.y = value; break;
            case Axis.Z: v.z = value; break;
            default: throw new ArgumentOutOfRangeException(nameof(axis), axis, null);
        }
    }
}
