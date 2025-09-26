# %%
from dolfinx import fem, io, mesh, plot, nls, log
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
print(f"Using dolfinx version: {dolfinx.__version__}")

import numpy as np
import sys
import os
import re
import cv2
import trimesh
import tetgen 
import time
import gmsh
import meshio 
import math
import shutil
import json
import gc

import matplotlib.pyplot as plt
import SimpleITK as sitk
import pymeshfix as pfix
import pyvista as pv
import skimage.morphology as skm
import iso2mesh as i2m
import pygalmesh as pygm
import pymeshlab as ml
import triangle as tr

from iso2mesh import plotmesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import io as skio
from skimage.filters import threshold_otsu
from tqdm import tqdm
from skimage.transform import resize
from typing import Any, Dict, Optional, Tuple, List

sys.path.append('./src')  # to import alveoRVE from parent directory

from alveoRVE.plot.mpl import show_four_panel_volume
from alveoRVE.plot.pv import view_surface


# %%
def matlab_to_python_conv(no, fc): 
    no_out = no[:, :3].copy()
    fc_out = (np.atleast_2d(fc).astype(np.int64)[:, :3] - 1).astype(np.int32)
    return no_out, fc_out


def _to_pv_polydata(tm: trimesh.Trimesh) -> pv.PolyData:
    """Trimesh -> PyVista PolyData (faces in VTK cell array format)."""
    pts = np.asarray(tm.vertices, dtype=float)
    tri = np.asarray(tm.faces,    dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("Expected triangular faces Nx3")
    cells = np.empty((tri.shape[0], 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = tri
    cells = cells.ravel()
    pd = pv.PolyData(pts, cells)
    return pd

def _from_pv_polydata(pd: pv.PolyData) -> trimesh.Trimesh:
    """PyVista PolyData -> Trimesh (triangles)."""
    # ensure triangles
    if not pd.is_all_triangles:
        pd = pd.triangulate()
    faces_vtk = np.asarray(pd.faces, dtype=np.int64)
    if faces_vtk.size == 0:
        return trimesh.Trimesh(vertices=np.asarray(pd.points, float), faces=np.empty((0,3), int), process=False)
    faces = faces_vtk.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    verts = np.asarray(pd.points, dtype=float)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def vtk_weld_two(triA: trimesh.Trimesh, triB: trimesh.Trimesh, tol: float = 5e-9) -> trimesh.Trimesh:
    """Append two touching shells and weld coincident seam vertices with absolute tolerance."""
    pa = _to_pv_polydata(triA)
    pb = _to_pv_polydata(triB)

    print(f"[pv] A: V={pa.n_points:,} F={pa.n_cells:,} | B: V={pb.n_points:,} F={pb.n_cells:,}")
    p = pa + pb  # vtkAppendPolyData
    print(f"[pv] appended: V={p.n_points:,} F={p.n_cells:,}")

    # clean: absolute tolerance; merge coincident points; drop duplicate cells
    p = p.clean(tolerance=tol, absolute=True, point_merging=True)
    print(f"[pv] after clean: V={p.n_points:,} F={p.n_cells:,}")

    # ensure triangles and drop duplicates again just in case
    p = p.triangulate()
    print(f"[pv] after triangulate+dedup: V={p.n_points:,} F={p.n_cells:,}")

    tm = _from_pv_polydata(p)
    print(f"[pv] back to trimesh: V={len(tm.vertices):,} F={len(tm.faces):,} | watertight={tm.is_watertight}")
    return tm


def _axis_pair(ax:str):
    a = AXIS_ID[ax]
    t = [0,1,2]; t.remove(a)
    return a, t[0], t[1]

def preview_periodic_pairs(V:np.ndarray, axis:str, tol:float=1e-6):
    a,t1,t2 = _axis_pair(axis)
    Smin = np.where(np.isclose(V[:,a], 0.0, atol=tol))[0]
    Smax = np.where(np.isclose(V[:,a], 1.0, atol=tol))[0]
    kd = cKDTree(V[Smax][:,[t1,t2]])
    d,j = kd.query(V[Smin][:,[t1,t2]], distance_upper_bound=max(tol, 10*tol))
    ok = np.isfinite(d)
    n_ok = int(np.sum(ok)); n_tot = len(Smin)
    print(f"[pairs {axis}] matched {n_ok}/{n_tot} within tol={tol:g}")

    if n_ok < n_tot:
        print(f"[pairs {axis}] WARNING: some vertices on min/max planes could not be matched within tol={tol:g}")

    return Smin[ok], Smax[j[ok]], n_ok, n_tot

def area_of_flat_faces(V: np.ndarray, F: np.ndarray, axis: str, plane_value: float, tol: float = 1e-9) -> float:
    ax = AXIS_ID[axis]
    oth = [i for i in range(3) if i != ax]

    # Select faces that have all three vertices on the plane
    face_mask = np.all(np.isclose(V[F][:, :, ax], plane_value, atol=tol), axis=1)
    F_plane = F[face_mask]

    if len(F_plane) == 0:
        print(f"No faces found on the {axis}={plane_value} plane within tol={tol}")
        return 0.0

    # Calculate area of these faces
    p0, p1, p2 = V[F_plane][:, 0], V[F_plane][:, 1], V[F_plane][:, 2]
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    total_area = float(np.sum(A))

    print(f"Total area of faces on the {axis}={plane_value} plane: {total_area:.6g} (from {len(F_plane)} faces)")
    return total_area


# %%
def remove_duplicate_faces(no: np.ndarray, fc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate faces from the mesh.

    Parameters:
    no (np.ndarray): Array of node coordinates.
    fc (np.ndarray): Array of face indices.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the updated node coordinates and face indices.
    """
    # Sort each face's vertex indices to ensure consistent ordering
    sorted_fc = np.sort(fc, axis=1)
    
    # Find unique faces and their indices
    unique_faces, unique_indices = np.unique(sorted_fc, axis=0, return_index=True)
    
    # Select only the unique faces
    fc_unique = fc[unique_indices]

    print(f"original number of faces: {fc.shape[0]}, unique faces: {fc_unique.shape[0]}")
    
    return fc_unique

def _triangle_quality(V, F):
    # 2*sqrt(3)*A / sum(l^2)  -> 1 for equilateral; 0 for degenerate
    p0, p1, p2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    e0 = np.linalg.norm(p1 - p0, axis=1)
    e1 = np.linalg.norm(p2 - p1, axis=1)
    e2 = np.linalg.norm(p0 - p2, axis=1)
    A  = 0.5*np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    denom = e0**2 + e1**2 + e2**2
    with np.errstate(divide='ignore', invalid='ignore'):
        q = (2.0*math.sqrt(3.0))*A/denom
        q[~np.isfinite(q)] = 0.0
    return np.clip(q, 0.0, 1.0)

def mesh_report(mesh: trimesh.Trimesh, name="mesh", plot=True, tol_dup=1e-12) -> dict:
    """
    Print thorough stats and return a dict. Does NOT mutate the mesh.
    """
    m = mesh.copy()

    print(f"\n=== {name}: geometric/graph checks START ===")
    print(f"trimesh metrics:")
    print(f"mesh is_winding_consistent? {m.is_winding_consistent}")
    print(f"mesh is_watertight? {m.is_watertight}")
    print(f"verts={len(m.vertices):,}, faces={len(m.faces):,}")
    bbox = m.bounds
    L = bbox[1] - bbox[0]
    print(f"bbox min={bbox[0]}, max={bbox[1]}, extents={L}")

    # basic areas/volumes
    A = float(m.area)
    Vvol = float(m.volume) if m.is_watertight else None
    print(f"surface area = {A:.6g}")
    if Vvol is not None:
        print(f"enclosed volume (watertight) = {Vvol:.6g}")
    else:
        print("enclosed volume = N/A (mesh not watertight)")

    # edge-length stats (proxy for 'h')
    edges = m.edges_unique
    elen  = np.linalg.norm(m.vertices[edges[:,0]] - m.vertices[edges[:,1]], axis=1)
    print(f"h (edge length): min={elen.min():.4g}, mean={elen.mean():.4g}, max={elen.max():.4g}")

    # duplicates (faces & vertices)
    F_sorted = np.sort(m.faces, axis=1)
    _, idx = np.unique(F_sorted, axis=0, return_index=True)
    dup_faces = len(m.faces) - len(idx)
    Vkey = np.round(m.vertices / tol_dup).astype(np.int64)
    _, vidx = np.unique(Vkey, axis=0, return_index=True)
    dup_verts = len(m.vertices) - len(vidx)
    print(f"duplicate faces: {dup_faces:,}, duplicate vertices (<= {tol_dup:g}): {dup_verts:,}")

    V, F = m.vertices.copy(), m.faces.copy()

    print(f"== pymeshlab metrics:")

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V, F))

    geo_measures = ms.get_geometric_measures()
    topo_measures = ms.get_topological_measures()
    print("Geometric measures:")
    for k, v in geo_measures.items():
        if k in ['surface_area', 'avg_edge_length', 'volume']:
            v = float(v)
            print(f"  {k}: {v}")
    print("Topological measures:")
    for k, v in topo_measures.items():
        if k in ['non_two_manifold_edges', 'boundary_edges', 'non_two_manifold_vertices', 'genus', 'faces_number', 'vertices_number', 'edges_number', 'connected_components_number']:
            v = int(v)
            print(f"  {k}: {v}")
    print(f"== custom metrics:")
    # triangle quality
    q = _triangle_quality(m.vertices, m.faces)
    q_stats = dict(min=float(q.min()), p5=float(np.percentile(q,5)),
                   mean=float(q.mean()), p95=float(np.percentile(q,95)),
                   max=float(q.max()))
    print(f"triangle quality q in [0,1] (equilateral=1): "
          f"min={q_stats['min']:.3f}, p5={q_stats['p5']:.3f}, "
          f"mean={q_stats['mean']:.3f}, p95={q_stats['p95']:.3f}, max={q_stats['max']:.3f}")


    if plot:
        plt.figure(figsize=(5,3))
        plt.hist(q, bins=40, range=(0,1), alpha=0.8)
        plt.xlabel("triangle quality q"); plt.ylabel("count"); plt.title(f"Quality histogram: {name}")
        plt.tight_layout(); plt.show()

    return dict(
        verts=len(m.vertices), faces=len(m.faces),
        area=A, volume=Vvol, bbox=bbox, h_stats=(float(elen.min()), float(elen.mean()), float(elen.max())),
        watertight=bool(m.is_watertight),
        dup_faces=int(dup_faces), dup_verts=int(dup_verts),
        tri_quality=q_stats
    )

def quick_mesh_report(ms: ml.MeshSet | trimesh.Trimesh, i: int = 0):
    # print(f"\n == pymeshlab quick metrics:")
    # number of vertices and faces

    flag = False
    if isinstance(ms, trimesh.Trimesh):
        flag = True
        V, F = ms.vertices, ms.faces
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(V, F))
    else: 
        V = ms.current_mesh().vertex_matrix()
        F = ms.current_mesh().face_matrix()

    n_verts = ms.current_mesh().vertex_number()
    n_faces = ms.current_mesh().face_number()
    geo_measures = ms.get_geometric_measures()
    topo_measures = ms.get_topological_measures()
    h = geo_measures['avg_edge_length']

    connected_components = topo_measures['connected_components_number']

    # trimesh watertightness
    trimesh_mesh = trimesh.Trimesh(
        vertices=np.asarray(ms.current_mesh().vertex_matrix(), float),
        faces=np.asarray(ms.current_mesh().face_matrix(), int),
        process=False
    )
    is_watertight = trimesh_mesh.is_watertight
    is_winding_consistent = trimesh_mesh.is_winding_consistent

    # trimesh volume
    vol = trimesh_mesh.volume if is_watertight else None
    area = trimesh_mesh.area

    # pymeshlab nonmanifold edges/faces
    n_nonmanifold_edges = int(topo_measures['non_two_manifold_edges'])
    n_nonmanifold_vertices = int(topo_measures['non_two_manifold_vertices'])

    print(f"[quick {i} 1/3] {n_verts} verts, {n_faces} faces, watertight: {is_watertight}, genus: {topo_measures['genus']}, wind-consistent: {is_winding_consistent}, h = {np.round(h, 3)}, components: {connected_components}\n[quick {i} 2/3] vol = {vol}, area = {area}, non-manifold edges: {n_nonmanifold_edges}/ vertices: {n_nonmanifold_vertices}\n[quick {i} 3/3] bbox: {V.min(axis=0)} to {V.max(axis=0)}")

    if flag: 
        del ms

    return {
        "n_verts": n_verts,
        "n_faces": n_faces,
        "is_watertight": is_watertight,
        "is_winding_consistent": is_winding_consistent,
        "h": h,
        "connected_components": connected_components,
        "vol": vol,
        "area": area,
        "n_nonmanifold_edges": n_nonmanifold_edges,
        "n_nonmanifold_vertices": n_nonmanifold_vertices
    }

def heal(ms: ml.MeshSet, manifold_method = 0, verbose: bool = False):
    i = 0
    print("[REMOVING DUPLICATE VERTICES]")
    ms.meshing_remove_duplicate_vertices()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING DUPLICATE FACES]")
    ms.meshing_remove_duplicate_faces()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING NULL FACES]")
    ms.meshing_remove_null_faces()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING UNREFERENCED VERTICES]")
    ms.meshing_remove_unreferenced_vertices()
    if verbose: quick_mesh_report(ms, i)
    # NOTE: Intentionally do NOT remove T vertices as per user request
    # ms.meshing_remove_t_vertices()  # (kept disabled)
    # i+1; quick_mesh_report(ms, i)
    print(f"[REPAIRING NON-MANIFOLD EDGES with manifold_method = {manifold_method}]")
    ms.meshing_repair_non_manifold_edges(method=manifold_method)
    if verbose: quick_mesh_report(ms, i)
    print("[REPAIRING NON-MANIFOLD VERTICES]")
    ms.meshing_repair_non_manifold_vertices()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING NULL FACES AGAIN]")
    ms.meshing_remove_null_faces()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING UNREFERENCED VERTICES AGAIN]")
    ms.meshing_remove_unreferenced_vertices()
    if verbose: quick_mesh_report(ms, i)
    print(f"[REPAIRING NON-MANIFOLD EDGES AGAINx2 with manifold_method = {1-manifold_method}]")
    ms.meshing_repair_non_manifold_edges(method=1-manifold_method)
    if verbose: quick_mesh_report(ms, i)
    print("[REPAIRING NON-MANIFOLD VERTICES AGAINx2]")
    ms.meshing_repair_non_manifold_vertices()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING NULL FACES AGAINx2]")
    ms.meshing_remove_null_faces()
    if verbose: quick_mesh_report(ms, i)
    print("[REMOVING UNREFERENCED VERTICES AGAINx2]")
    ms.meshing_remove_unreferenced_vertices()
    if verbose: quick_mesh_report(ms, i)

    if trimesh.Trimesh(
        vertices=np.asarray(ms.current_mesh().vertex_matrix(), float),
        faces=np.asarray(ms.current_mesh().face_matrix(), int),
        process=False
    ).volume < 0:
        ms.meshing_invert_face_orientation()

    # geo_measures = ms.get_geometric_measures()
    # topo_measures = ms.get_topological_measures()
    # print("\nGeometric measures:")
    # for k, v in geo_measures.items():
    #     print(f"  {k}: {v}")
    # print("\nTopological measures:")
    # for k, v in topo_measures.items():
    #     print(f"  {k}: {v}")

def remove_close_verts(ms: ml.MeshSet, tol=1e-5):
    print(f"Removing close vertices with tol={tol}")
    initial_v = ms.current_mesh().vertex_number()
    ms.meshing_merge_close_vertices(threshold = ml.PercentageValue(tol*100))
    final_v = ms.current_mesh().vertex_number()
    print(f" - removed {initial_v - final_v} vertices")

def print_python_or_matlab_indexing(fc, name=""): 
    if np.min(fc) == 0:
        print(f" - {name} (python)")
    elif np.min(fc) == 1:
        print(f" - {name} (matlab)")
    else:
        print(f" - {name} (unknown indexing)")

def normalize_vertices_inplace(no):
    no[:, :3] = (no[:, :3] - no[:, :3].min(axis=0)) / (no[:, :3].max(axis=0) - no[:, :3].min(axis=0))

def normalize_vertices(no): 
    no_out = no.copy()
    no_out[:, :3] = (no_out[:, :3] - no_out[:, :3].min(axis=0)) / (no_out[:, :3].max(axis=0) - no_out[:, :3].min(axis=0))
    return no_out

def view_wireframe(V: np.ndarray, F: np.ndarray, title="surface"):
    faces = np.hstack([np.full((len(F),1),3), F]).ravel()
    mesh = pv.PolyData(V, faces)
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, color='black', style='wireframe')
    p.add_axes(); p.show(title=title)

def view_cropped(V, F, title, point_on_plane=None, normal=None, mode="wireframe"):
    # default is half the model
    if point_on_plane is None:
        point_on_plane = V.mean(axis=1)
    if normal is None:
        normal = np.array([0,0,1])
    faces = np.hstack([np.full((len(F),1),3), F]).ravel()
    mesh = pv.PolyData(V, faces)
    clipped = mesh.clip(normal=normal, origin=point_on_plane, invert=False)
    p = pv.Plotter()
    p.add_mesh(clipped, show_edges=True, style="wireframe" if mode=="wireframe" else "surface")
    p.add_axes(); p.show(title=title)

# Utility: robust JSON saver handling numpy types
def _to_jsonable(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_to_jsonable)




# %% [markdown]

# Global configs
ct_folder           = "/mnt/c/Users/bherr/Downloads/CT-1um/images_reconstructed_1um"
e8_folder           = "/mnt/c/Users/bherr/Downloads/E8-1um/images_reconstructed_1um"
supported_modes     = ["CT", "E8"]
folders             = [ct_folder, e8_folder]
npz_ct_base_path    = "/home/bnherrerac/CHASKi/alveoRVE/results/CT/rves/"
npz_e8_base_path    = "/home/bnherrerac/CHASKi/alveoRVE/results/E8/rves/"
npz_base_paths      = [npz_ct_base_path, npz_e8_base_path]

resize_factor       = 2.0
size_rve            = 100 # um
n_samples           = 30 # 150
radbound            = 2.0

# random seed 
np.random.seed(38)
opts = {"radbound": float(radbound)}    

resume_loop2_from_existing = False

# Optional fast I/O: if provided, use numpy.memmap stacks to slice ROIs instead of per-file imread.
# Set NUMPY_MEMMAP_DIR to a folder containing files named CT.npy / E8.npy and sidecar CT.json / E8.json
# with metadata: {"shape": [Y, X, Z], "dtype": "uint8"}. Slices are treated as 8-bit images identical to BMP inputs.
NUMPY_MEMMAP_DIR = "/home/bnherrerac/CHASKi/alveoRVE/data/memmap_stacks"  # leave as-is unless you create memmaps

AXIS_ID = {'x':0,'y':1,'z':2}

# %% [markdown]
for mode, folder, npz_base_path in zip(supported_modes, folders, npz_base_paths):
    print(f"\n\n=== Processing mode: {mode} ===")
    RVEs_bounds = []
    RVEs_img_paths = []
    RVEs_binary_paths = []
    RVEs_folders = []
    n_generated_samples = 0
    samples = np.zeros((n_samples, 3))

    # Try to locate a prepacked memmap for faster loading
    memmap_path = os.path.join(NUMPY_MEMMAP_DIR, f"{mode}.npy")
    memmap_meta = os.path.join(NUMPY_MEMMAP_DIR, f"{mode}.json")
    has_memmap = (not resume_loop2_from_existing) and os.path.isfile(memmap_path) and os.path.isfile(memmap_meta)

    if resume_loop2_from_existing:
        print(f"[RESUME] Skipping {mode} stack loading. Will collect existing STL meshes and run LOOP2 only.")
        files = []
        paths = []
        z = 0
        firstimg = np.zeros((1,1), dtype=np.uint8)
    else:
        if has_memmap:
            try:
                with open(memmap_meta, 'r') as f:
                    meta = json.load(f)
                mm_shape = tuple(int(v) for v in meta.get("shape", []))
                mm_dtype = np.dtype(meta.get("dtype", "uint8"))
                assert len(mm_shape) == 3, "memmap shape must be [Y,X,Z]"
                # open read-only memory-mapped .npy
                stack_arr = np.load(memmap_path, mmap_mode='r')
                if stack_arr.shape != mm_shape:
                    print(f"[FAST I/O] Warning: JSON shape {mm_shape} != file shape {stack_arr.shape}; using file metadata")
                y, x, z = stack_arr.shape
                firstimg = stack_arr[:, :, 0]
                print(f"[FAST I/O] Using memmapped npy stack for {mode}: {memmap_path} | shape={stack_arr.shape} dtype={stack_arr.dtype}")
            except Exception as e:
                print(f"[FAST I/O] Failed to open memmap for {mode}: {e}. Falling back to per-file reading.")
                has_memmap = False
        if not has_memmap:
            files = [f for f in os.listdir(folder) if f.lower().endswith(".bmp")]
            # numeric sort by the largest integer in filename
            def key(f): 
                m = re.findall(r"\d+", f)
                return int(m[-1]) if m else -1
            paths = [os.path.join(folder, f) for f in sorted(files, key=key)]
            z = len(paths)
            firstimg = skio.imread(paths[0], as_gray=True)
    # # plot simple histogram of the first image, in ONLY y log axis, side by side the image in question
    # plt.figure(figsize=(12,5))
    # plt.subplot(1,3,1)
    # plt.imshow(firstimg, cmap='gray')
    # plt.title("First Image in Stack")
    # plt.axis('off')
    # plt.subplot(1,3,2)
    # plt.hist(firstimg.ravel(), bins=256, range=(0, 1), color='black')
    # plt.yscale('log')
    # plt.title("Histogram of First Image")
    # plt.xlabel("Intensity")
    # plt.ylabel("Frequency")
    # plt.subplot(1,3,3)
    # # plot binarized image with threshold t = 25/255 defined below
    # t = 43/255
    # plt.imshow(firstimg > t, cmap='gray')
    # # colorbar
    # plt.colorbar()
    # plt.title("Binarized Image")
    # plt.axis('off')
    # plt.show()
    if not resume_loop2_from_existing:
        y, x = firstimg.shape
        print(f"Image size: {x} x {y} x {z}")

    T0 = time.time()

    if not resume_loop2_from_existing:
        volume_shape = [y, x, z]

    # %%
    ## snapshot of 0%, 20%, 40%, 60%, 80%, 100% of the stack
    # fig, ax = plt.subplots(1, 6, figsize=(15, 5))
    # for i, perc in enumerate([0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    #     idx = int((z-1) * perc)
    #     img = cv2.imread(paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
    #     ax[i].imshow(img, cmap='gray')
    #     ax[i].set_title(f"Slice {idx}")
    #     ax[i].axis('off')
    # plt.tight_layout()
    # plt.show()


    print(f"Generating {n_samples} RVEs of size {size_rve}x{size_rve}x{size_rve} um from {mode} data in {folder}")
    # Resume mode: collect existing STL meshes and decide whether to run LOOP1
    if resume_loop2_from_existing:
        base_dir = os.path.join(npz_base_path, f"{size_rve}um")
        collected = 0
        if os.path.isdir(base_dir):
            for root, dirs, files_in in os.walk(base_dir):
                if "triang_after_dedup.stl" in files_in:
                    bin_anchor = os.path.join(root, "binary_seg.npz")
                    # We only need the folder path in LOOP2; the file doesn't need to be readable here.
                    RVEs_binary_paths.append(bin_anchor)
                    collected += 1
        print(f"[RESUME] Found {collected} existing triang_after_dedup.stl meshes. Skipping LOOP1.")
        print(f"[RESUME] LOOP2 will process {len(RVEs_binary_paths)} items.")
    run_loop1 = not resume_loop2_from_existing

    while run_loop1 and n_generated_samples < n_samples:
        # Progress flag for current iteration (1-based index out of total)
        iter_tag = f"[{n_generated_samples+1}/{n_samples}]"
        print(f"[LOOP1] START {iter_tag}")

        ## extraction and saving 
        # whole image inside original macroRVE 
        sample = np.array([0,0,0])
        # project x and y coordinates in a circle contained in the image
        center_xy = np.array([volume_shape[0]//2, volume_shape[1]//2])
        maxR = min(volume_shape[0]//2, volume_shape[1]//2) - size_rve//2
        radius_xy = np.random.uniform(0, maxR - 5)  # leave 5px margin

        angle = np.random.uniform(0, 2*np.pi)
        sample[0] = int(center_xy[0] + radius_xy * np.cos(angle))
        sample[1] = int(center_xy[1] + radius_xy * np.sin(angle))
        sample[2] = np.random.randint(size_rve//2, volume_shape[2] - size_rve//2)

        curr_folder = os.path.join(npz_base_path, f"{size_rve}um/{sample[0]}_{sample[1]}_{sample[2]}")
        print("\n============================================================")
        print(f"[RVE GEN] curr_folder: {curr_folder} | sample xyz: {tuple(int(v) for v in sample)} | size: {size_rve}")

        # if folder does not exist, create it
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)

        out_name = os.path.join(curr_folder, "rve.npz")
        # If STL from LOOP1 already exists for this folder, skip image processing
        stl_exists_path = os.path.join(curr_folder, "triang_after_dedup.stl")
        if os.path.exists(stl_exists_path):
            bin_anchor = os.path.join(curr_folder, "binary_seg.npz")
            RVEs_binary_paths.append(bin_anchor)
            # Count as accepted for progress accounting
            samples[n_generated_samples] = sample
            n_generated_samples += 1
            print(f"[LOOP1] {iter_tag} SKIP: found existing triang_after_dedup.stl → reusing and skipping image pipeline")
            gc.collect()
            continue

        # make rve img and save
        try:
            data = np.load(out_name)
            vol = data['vol']
            print(f"[reload] Volume shape: {vol.shape}; min/max: {vol.min()}/{vol.max()}")
            if tuple(vol.shape) != (size_rve, size_rve, size_rve):
                print(f"[reload] REJECT: volume shape mismatch {vol.shape} != ({size_rve},{size_rve},{size_rve}); deleting {curr_folder}")
                try:
                    os.remove(out_name)
                except Exception:
                    pass
                shutil.rmtree(curr_folder, ignore_errors=True)
                del vol
                gc.collect()
                continue
            if vol.max() == 0 or not np.isfinite(vol).all():
                print("[reload] REJECT: invalid volume (all zeros or non-finite); deleting folder")
                try:
                    os.remove(out_name)
                except Exception:
                    pass
                shutil.rmtree(curr_folder, ignore_errors=True)
                del vol
                gc.collect()
                continue
        except Exception:
            imgs: List[np.ndarray] = []
            if has_memmap:
                # Slice directly from memmap; axis order in memmap is [Y,X,Z], we need [Z,Y,X]
                z0 = int(sample[2])
                z1 = int(sample[2] + size_rve)
                y0 = int(sample[1])
                y1 = int(sample[1] + size_rve)
                x0 = int(sample[0])
                x1 = int(sample[0] + size_rve)
                sub = stack_arr[y0:y1, x0:x1, z0:z1]
                if sub.shape != (size_rve, size_rve, size_rve):
                    print(f"[LOOP1] REJECT: memmap slice shape {sub.shape} != ({size_rve},{size_rve},{size_rve})")
                    shutil.rmtree(curr_folder, ignore_errors=True)
                    gc.collect()
                    continue
                # reorder to [Z,Y,X]
                vol_u8 = np.transpose(sub, (2,0,1)).astype(np.uint8, copy=False)
                print(f"[LOOP1][FAST I/O] stacked volume (uint8) shape: {vol_u8.shape}; min/max: {vol_u8.min()}/{vol_u8.max()}")
                # Save a location preview from the first slice
                full_img = stack_arr[:, :, z0]
                fig_loc, ax_loc = plt.subplots(figsize=(6, 6))
                ax_loc.imshow(full_img, cmap='gray')
                ax_loc.add_patch(plt.Rectangle((sample[0], sample[1]), size_rve, size_rve, edgecolor='red', facecolor='none', lw=2))
                ax_loc.set_title(f"RVE location: {sample[0]},{sample[1]} ({size_rve}x{size_rve})")
                ax_loc.axis('off')
                fig_loc.tight_layout()
                fig_loc.savefig(os.path.join(curr_folder, "rve_location.png"), dpi=300)
                plt.close(fig_loc)
                del full_img
            else:
                for i, p in enumerate(tqdm(paths[sample[2]:sample[2]+size_rve], desc=f"[LOOP1] {iter_tag} Reading BMP stack")):
                    # Read ROI as uint8 to minimize memory; cast later only once
                    roi_full = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    roi = roi_full[sample[1]:sample[1]+size_rve, sample[0]:sample[0]+size_rve].copy()
                    del roi_full
                    if roi is None:
                        print(f"[LOOP1] ERROR: Failed to read image at path: {p}")
                        shutil.rmtree(curr_folder, ignore_errors=True)
                        imgs = []
                        break
                    if i == 0:
                        full_img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                        print(f"[LOOP1] first ROI shape: {roi.shape}; dtype: {roi.dtype}; min/max: {roi.min()}/{roi.max()}")
                        fig_loc, ax_loc = plt.subplots(figsize=(6, 6))
                        ax_loc.imshow(full_img, cmap='gray')
                        # Rectangle expects (x, y); ROI slicing above uses [y, x]
                        # so the correct top-left corner is (sample[0], sample[1])
                        ax_loc.add_patch(plt.Rectangle((sample[0], sample[1]), size_rve, size_rve, edgecolor='red', facecolor='none', lw=2))
                        ax_loc.set_title(f"RVE location: {sample[0]},{sample[1]} ({size_rve}x{size_rve})")
                        ax_loc.axis('off')
                        fig_loc.tight_layout()
                        fig_loc.savefig(os.path.join(curr_folder, "rve_location.png"), dpi=300)
                        plt.close(fig_loc)
                        del full_img
                    # Ensure uint8 to reduce memory; many BMPs are 8-bit
                    if roi.dtype != np.uint8:
                        try:
                            roi = roi.astype(np.uint8, copy=False)
                        except Exception:
                            roi = (roi / roi.max() * 255.0).astype(np.uint8) if roi.max() > 0 else roi.astype(np.uint8)
                    imgs.append(roi)
                if not imgs:
                    # Already cleaned up; skip this sample
                    gc.collect()
                    continue
                vol_u8 = np.stack(imgs).astype(np.uint8, copy=False)  # [Z,Y,X]
                # free list container
                del imgs
                print(f"[LOOP1] stacked volume (uint8) shape: {vol_u8.shape}; min/max: {vol_u8.min()}/{vol_u8.max()}")
            # Enforce exact shape 80x80x80
            if tuple(vol_u8.shape) != (size_rve, size_rve, size_rve):
                print(f"[LOOP1] REJECT: volume shape {vol_u8.shape} != ({size_rve},{size_rve},{size_rve})")
                shutil.rmtree(curr_folder, ignore_errors=True)
                del vol_u8
                gc.collect()
                continue
            if vol_u8.max() == 0:
                print("[LOOP1] REJECT: volume is all zeros")
                shutil.rmtree(curr_folder, ignore_errors=True)
                del vol_u8
                gc.collect()
                continue
            # Normalize to float32
            vmax = float(vol_u8.max())
            vol = (vol_u8.astype(np.float32) / vmax) if vmax > 0 else vol_u8.astype(np.float32)
            del vol_u8
            if not np.isfinite(vol).all():
                print("[LOOP1] REJECT: non-finite values after normalization")
                shutil.rmtree(curr_folder, ignore_errors=True)
                del vol
                gc.collect()
                continue
            print(f"[LOOP1] volume (float32) min/max: {vol.min()}/{vol.max()}")

            np.savez_compressed(out_name, vol=vol)

        # calculate porosity from simple thresholding, 
        thr = 30/255
        critical_porosity = 0.95
        critical_porosity_lower = 0.50
        porosity = 1.0 - (np.sum(vol) / vol.size)
        porosity_binary = np.mean(vol < thr)
        print(f"[LOOP1] Porosity = {porosity:.4f}")
        print(f"[LOOP1] Binary porosity (thr={thr:.3f}): {porosity_binary*100:.4f}%")

        # check if porosity is above critical threshold
        if porosity > critical_porosity or porosity < critical_porosity_lower:
            print(f"[LOOP1] REJECT: anomalous porosity ({porosity*100:.4f}%) -> deleting {curr_folder}")
            try:
                os.remove(out_name)
            except Exception:
                pass
            shutil.rmtree(curr_folder, ignore_errors=True)
            del vol
            gc.collect()
            continue

        # no overlapping
        if any((sample[0] >= b[0][0] and sample[0] <= b[1][0]) and 
            (sample[1] >= b[0][1] and sample[1] <= b[1][1]) and 
            (sample[2] >= b[0][2] and sample[2] <= b[1][2]) for b in RVEs_bounds):
            print(f"[LOOP1] REJECT: sample {tuple(int(v) for v in sample)} overlaps existing RVE -> deleting {curr_folder}")
            try:
                os.remove(out_name)
            except Exception:
                pass
            shutil.rmtree(curr_folder, ignore_errors=True)
            del vol
            gc.collect()
            continue

        # store RVE bounds and path
        RVEs_bounds.append(([sample[0], sample[1], sample[2]], [sample[0]+size_rve, sample[1]+size_rve, sample[2]+size_rve]))
        RVEs_img_paths.append(out_name)
        RVEs_folders.append(curr_folder)

        ## processing
        # previsualization
        figg = show_four_panel_volume(vol, title_prefix="original")
        # savefig
        figg.savefig(os.path.join(curr_folder, "rve_original.png"), dpi=300)
        plt.close(figg)
        del figg
        
        
        # resizing
        z, y, x = vol.shape
        new_shape = (int(round(z*resize_factor)), int(round(y*resize_factor)), int(round(x*resize_factor)))
        out = resize(
                vol,
                new_shape,
                order=3, 
                anti_aliasing=True, 
                preserve_range=True
            ).astype(vol.dtype)
        figg = show_four_panel_volume(out, title_prefix="resized")
        # savefig
        figg.savefig(os.path.join(curr_folder, "rve_resized.png"), dpi=300)
        plt.close(figg)
        del figg
        porosity_out = 1.0 - (np.sum(out) / out.size)
        print(f"[resize] new porosity: {porosity_out:.4f}")
        print(f"[resize] new shape: {out.shape}")

        volume_np = out.astype(np.float32, copy=False)
        volume_sitk = sitk.GetImageFromArray(volume_np)

        voxel_size = int(1*resize_factor)  # in um, assuming original voxel size is 1 um

        slice_index_z_for_plot = size_rve // 2
        print("starting filtering pipeline")
        #!!!!!! must make this absolute in the image intensity range, since noise is being misinterpreted as alveolar walls. 
        ## 1. Denoising: use Curvature Anisotropic Diffusion to denoise while preserving edges
        t0 = time.time()
        denoised = sitk.CurvatureAnisotropicDiffusion(volume_sitk, timeStep=0.0625, conductanceParameter=10, numberOfIterations=30)
        print(f"denoised with Curvature Anisotropic Diffusion, time: {time.time()-t0:.2f} s")
        t0 = time.time()
        clipped_image = sitk.Threshold(denoised, lower=30/255, upper=255/255, outsideValue=0)
        print(f"thresholded to remove low-valued pixels, time: {time.time()-t0:.2f} s")
        t0 = time.time()
        clahe = sitk.AdaptiveHistogramEqualization(clipped_image, alpha=0.5, beta=0.7)
        print(f"contrast enhanced with Adaptive Histogram Equalization, time: {time.time()-t0:.2f} s")
        ## 3. Thresholding: use the mean of the image as a global threshold
        t0 = time.time()
        binary = sitk.BinaryThreshold(clahe, lowerThreshold=0.202696, upperThreshold=255/255, outsideValue=0, insideValue=1) #!!! np.mean(clahe)
        print(f"thresholded to binary image with new param, time: {time.time()-t0:.2f} s")
        ## Convert back to NumPy for scikit-image postprocessing
        t0 = time.time()
        binary_np = sitk.GetArrayFromImage(binary).astype(bool, copy=False)
        print(f"converted to NumPy array, time: {time.time()-t0:.2f} s")
        ## 4. Postprocessing: thin the segmentation and remove small unconnected objects
        t0 = time.time()
        binary_cleaned = skm.remove_small_objects(binary_np, min_size=(20/voxel_size)**3)
        print(f"removed small objects, time: {time.time()-t0:.2f} s")

        t0 = time.time()
        max_num_iter=2
        thinned_3d = []
        for i in range(binary_cleaned.shape[0]):
            thinned = skm.thin(binary_cleaned[i, :, :], max_num_iter=max_num_iter)
            thinned_3d.append(thinned)
        img_thinned = np.stack(thinned_3d)
        print(f"thinned slice-by-slice, time: {time.time()-t0:.2f} s")

        t0 = time.time()
        thin_cleaned   = skm.remove_small_objects(img_thinned, min_size=(20/voxel_size)**3)
        print(f"removed small objects after thinning, time: {time.time()-t0:.2f} s")
        t0 = time.time()
        img_segmented    = skm.remove_small_holes(thin_cleaned, 125000)
        print(f"removed small holes, time: {time.time()-t0:.2f} s")

        print(f"final porosity segmented: {1.0 - (np.sum(img_segmented) / img_segmented.size):.4f}")

        figg = show_four_panel_volume(img_segmented, title_prefix="final segmented", is_binary=True)
        # fig.show()
        # savefig
        figg.savefig(os.path.join(curr_folder, "rve_segmented.png"), dpi=300)
        plt.close(figg)
        del figg

        # save npz segmented
        out_seg_name = os.path.join(curr_folder, "binary_seg.npz")
        np.savez_compressed(out_seg_name, vol=img_segmented)
        RVEs_binary_paths.append(out_seg_name)

        stl_temp_path = os.path.join(curr_folder, "triang_after_dedup.stl")
        print(f"[LOOP1] segmented shape: {img_segmented.shape}; min/max: {img_segmented.min()}/{img_segmented.max()}; porosity={1.0 - (np.sum(img_segmented) / img_segmented.size):.4f}")

        print("---------------- Surface meshing (iso2mesh) ----------------")
        ## generate mesh with iso2mesh
        no, el, fc = i2m.v2m(img_segmented, 0.5, opts, 1e12, "cgalmesh")
        print(no.shape, fc.shape)
        if fc.shape[0] == 0:
            print(f"[LOOP1] REJECT: iso2mesh produced 0 faces -> deleting {curr_folder}")
            # cleanup
            try:
                os.remove(out_name)
            except Exception:
                pass
            shutil.rmtree(curr_folder, ignore_errors=True)
            # free memory of this sample
            del vol, out, volume_np, volume_sitk, denoised, clipped_image, clahe, binary, binary_np, binary_cleaned
            del thinned_3d, img_thinned, thin_cleaned, img_segmented
            gc.collect()
            continue
        # view_surface(no[:,:3], fc[:,:3] - 1, title="initial mesh from iso2mesh")
        fc_unique = remove_duplicate_faces(no, fc)
        print(f"[LOOP1] fc_unique shape: {fc_unique.shape}")
        i2m.savestl(*matlab_to_python_conv(no, fc_unique), stl_temp_path)
        # free largest arrays of LOOP1 before proceeding
        del fc_unique, no, el, fc
        # also free heavy volume intermediates no longer needed
        del out, volume_np, volume_sitk, denoised, clipped_image, clahe, binary, binary_np, binary_cleaned
        del thinned_3d, img_thinned, thin_cleaned, img_segmented
        gc.collect()

        # no duplicates
        if not any(np.all(sample == s) for s in samples[:n_generated_samples]):
            samples[n_generated_samples] = sample
            n_generated_samples += 1
            print(f"[LOOP1] END [{n_generated_samples}/{n_samples}] -> accepted")
        # free remaining small arrays from LOOP1
        try:
            del vol
        except NameError:
            pass
        gc.collect()

    if run_loop1:
        print(f"FINISHED GENERATING {n_generated_samples} SAMPLES")


    # %%
    print(f"[LOOP2] Items to process: {len(RVEs_binary_paths)}")
    completed_meshes = 0

    for i, bin_path in enumerate(RVEs_binary_paths):
        print("\n============================================================")
        loop2_tag = f"[{i+1}/{len(RVEs_binary_paths)}]"
        print(f"[LOOP2] START {loop2_tag}")
        out_folder_name = os.path.dirname(bin_path)

        # If the folder was renamed previously (e.g., bad_*, no_tetra_*), redirect to it
        if not os.path.isdir(out_folder_name):
            parent_dir = os.path.dirname(out_folder_name)
            base_name = os.path.basename(out_folder_name)
            for prefix in ("bad_", "no_tetra_"):
                cand = os.path.join(parent_dir, f"{prefix}{base_name}")
                if os.path.isdir(cand):
                    print(f"[LOOP2] {loop2_tag} Redirecting to renamed folder: {cand}")
                    out_folder_name = cand
                    bin_path = os.path.join(out_folder_name, "binary_seg.npz")
                    break

        # Minimal resume: if final stitched mesh already exists in this folder, skip
        final_stitched_path = os.path.join(out_folder_name, "final_stitched.stl")
        if os.path.exists(final_stitched_path):
            print(f"[LOOP2] {loop2_tag} SKIP: found existing final_stitched.stl → skipping welding/tetra for {out_folder_name}")
            gc.collect()
            continue

        stl_temp_path = os.path.join(out_folder_name, "triang_after_dedup.stl")
        ## read mesh
        mesh = meshio.read(stl_temp_path)
        nodes = mesh.points
        cells_dict = mesh.cells_dict
        print(f"[LOOP2] meshio read: points={nodes.shape}, triangles={cells_dict.get('triangle', []).__len__()}")
        elems = mesh.cells_dict['triangle']
        print(f"[LOOP2] nodes shape: {nodes.shape}; elems shape: {elems.shape}")
        # free meshio container early
        del mesh, cells_dict
        gc.collect()

        print("---------------- Initial clean (PyMeshFix) ----------------")
        ## pymeshfix 
        clean_nodes, clean_elems = pfix.clean_from_arrays(nodes, elems)
        print(f"cleaned\nold shapes: nodes {nodes.shape}, elems {elems.shape}\nnew shapes: clean_nodes {clean_nodes.shape}, clean_elems {clean_elems.shape}")
        # Checks whether original mesh was already clean
        is_nodes_unchanged = bool(np.array_equal(nodes, clean_nodes))
        is_faces_unchanged = bool(np.array_equal(elems, clean_elems))
        print(f"original already clean? nodes: {is_nodes_unchanged}, faces: {is_faces_unchanged}")

        ## quick initial report
        m = trimesh.Trimesh(nodes, elems)
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(nodes, elems))
        print("[ORIGINAL]")
        quick_mesh_report(ms, i=0)
        del ms

        print("---------------- Optional resample ----------------")
        ## resample step (optional)
        pct_to_keep = 0.8
        clean_nodes, clean_elems = i2m.meshresample(clean_nodes, clean_elems+1, pct_to_keep)

        # pymeshfix cleaning
        print("[PYMESHFIX CLEANING]")
        clean_nodes, clean_elems = pfix.clean_from_arrays(nodes, elems)
        m = trimesh.Trimesh(clean_nodes, clean_elems)
        quick_mesh_report(m, i=1)
        # safe to free original nodes/elems now
        del nodes, elems
        gc.collect()

        # reorient and heal (see if needed)
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(clean_nodes, clean_elems))
        ms.meshing_invert_face_orientation()
        quick_mesh_report(ms, i=0)
        heal(ms)

        # smoothing
        print(f"[SMOOTHING]")
        trimesh_mesh = trimesh.Trimesh(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
        smoothed_mesh = trimesh.smoothing.filter_mut_dif_laplacian(trimesh_mesh, lamb=0.5, iterations=10)
        quick_mesh_report(smoothed_mesh, i=0)
        del ms, trimesh_mesh
        gc.collect()

        # pymeshfix cleaning again
        print(f"[PYMESHFIX CLEANING AGAIN]")
        clean_nodes2, clean_elems2 = pfix.clean_from_arrays(smoothed_mesh.vertices, smoothed_mesh.faces)
        # smoothed_mesh no longer needed beyond this point
        del smoothed_mesh
        gc.collect()
        ms0 = ml.MeshSet()
        ms0.add_mesh(ml.Mesh(clean_nodes2, clean_elems2))
        quick_mesh_report(ms0, i=0)
        del ms0

        # normalize nodes 
        print(f"[NORMALIZING]")
        normalize_vertices_inplace(clean_nodes2)
        ms0 = ml.MeshSet()
        ms0.add_mesh(ml.Mesh(clean_nodes2, clean_elems2))
        # quick_mesh_report(ms0, i=0)
        del ms0

        print("---------------- Plane cutting & capping ----------------")
        # plane cutting
        print("[PLANE CUTTING]")
        d = 0.01

        x_plane_min_normal = (1,0,0)
        x_plane_min_point = (d,0,0)
        x_plane_max_normal = (-1,0,0)
        x_plane_max_point = (1-d,0,0)

        y_plane_min_normal = (0,1,0)
        y_plane_min_point = (0,d,0)
        y_plane_max_normal = (0,-1,0)
        y_plane_max_point = (0,1-d,0)   

        z_plane_min_normal = (0,0,1)
        z_plane_min_point = (0,0,d)
        z_plane_max_normal = (0,0,-1)
        z_plane_max_point = (0,0,1-d)

        normalized_mesh = trimesh.Trimesh(clean_nodes2, clean_elems2)
        # free arrays after constructing trimesh
        del clean_nodes2, clean_elems2
        gc.collect()

        engine = "triangle" # "triangle" # "manifold", "earcut"

        # consecutive slicing and capping
        normalized_mesh_sliced_x_min = normalized_mesh.slice_plane(x_plane_min_point, x_plane_min_normal, cap=True, engine=engine)
        # print(f"new vertices after x min slice: {normalized_mesh_sliced_x_min.vertices.shape}")
        normalized_mesh_sliced_x_max = normalized_mesh_sliced_x_min.slice_plane(x_plane_max_point, x_plane_max_normal, cap=True, engine=engine)
        # print(f"new vertices after x max slice: {normalized_mesh_sliced_x_max.vertices.shape}")
        normalized_mesh_sliced_y_min = normalized_mesh_sliced_x_max.slice_plane(y_plane_min_point, y_plane_min_normal, cap=True, engine=engine)
        # print(f"new vertices after y min slice: {normalized_mesh_sliced_y_min.vertices.shape}")
        normalized_mesh_sliced_y_max = normalized_mesh_sliced_y_min.slice_plane(y_plane_max_point, y_plane_max_normal, cap=True, engine=engine)
        # print(f"new vertices after y max slice: {normalized_mesh_sliced_y_max.vertices.shape}")
        normalized_mesh_sliced_z_min = normalized_mesh_sliced_y_max.slice_plane(z_plane_min_point, z_plane_min_normal, cap=True, engine=engine)
        # print(f"new vertices after z min slice: {normalized_mesh_sliced_z_min.vertices.shape}")
        normalized_mesh_sliced_all = normalized_mesh_sliced_z_min.slice_plane(z_plane_max_point, z_plane_max_normal, cap=True, engine=engine)
        # print(f"new vertices after z max slice: {normalized_mesh_sliced_all.vertices.shape}")

        # free slicing intermediates ASAP
        del normalized_mesh_sliced_x_min, normalized_mesh_sliced_x_max
        del normalized_mesh_sliced_y_min, normalized_mesh_sliced_y_max
        del normalized_mesh_sliced_z_min
        gc.collect()

        ## normalize again
        print(f"[NORMALIZING]")
        normalize_vertices_inplace(normalized_mesh_sliced_all.vertices)

        # mesh_report(normalized_mesh_sliced_all, name="after slicing and capping", plot=True)
        ms0 = ml.MeshSet()
        ms0.add_mesh(ml.Mesh(normalized_mesh_sliced_all.vertices, normalized_mesh_sliced_all.faces))
        # quick_mesh_report(ms0, i=0)
        del ms0

        # view_surface(normalized_mesh_sliced_all.vertices, normalized_mesh_sliced_all.faces, title="normalized")

        print("---------------- Isotropic remeshing ----------------")
        ## isotropic remeshing
        h_cap = 0.018
        iterations = 6  
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(normalized_mesh_sliced_all.vertices, normalized_mesh_sliced_all.faces))
        # no longer need the pre-remesh mesh
        del normalized_mesh_sliced_all
        gc.collect()
        print(f"[ISOTROPIC REMESHING to target h={h_cap}]")
        ms.meshing_isotropic_explicit_remeshing(targetlen=ml.PercentageValue(h_cap*100),
                                                iterations=iterations,
                                                adaptive=False,
                                                reprojectflag=True
                                                )

        heal(ms)
        normalized_mesh = trimesh.Trimesh(np.asarray(ms.current_mesh().vertex_matrix()), np.asarray(ms.current_mesh().face_matrix()))
        del ms
        gc.collect()

        # save to stl 
        normalized_mesh.export(os.path.join(out_folder_name, "triang_remeshed_normalized.stl"))

        print("---------------- Reflections & welding ----------------")
        ## reflection
        normalized_mesh_reflected_x = normalized_mesh.copy()
        normalized_mesh_reflected_x.vertices[:,0] = 2 - normalized_mesh.vertices[:,0]
        ms_reflected = ml.MeshSet()
        ms_reflected.add_mesh(ml.Mesh(normalized_mesh_reflected_x.vertices, normalized_mesh_reflected_x.faces))
        heal(ms_reflected)
        quick_mesh_report(ms_reflected, i=0)
        normalized_mesh_reflected_x = trimesh.Trimesh(np.asarray(ms_reflected.current_mesh().vertex_matrix()), np.asarray(ms_reflected.current_mesh().face_matrix()), process=False)
        del ms_reflected
        gc.collect()

        # view_cropped(normalized_mesh.vertices, normalized_mesh.faces, title="original", point_on_plane=(0.5,0.5,0.5), normal=(1,0,0), mode="surface")
        # view_cropped(normalized_mesh_reflected_x.vertices, normalized_mesh_reflected_x.faces, title="reflected", point_on_plane=(0.5,0.5,0.5), normal=(1,0,0), mode="surface")

        ### union
        ###### ALGORITHM FOR UNION (A) ######

        '''
        ## x
        print("[UNION IN X]")
        V0 = np.asarray(normalized_mesh.vertices, float)
        F0 = np.asarray(normalized_mesh.faces,    int)

        # print(f"bounding box: min={V0.min(axis=0)}, max={V0.max(axis=0)}")

        shift_x = np.array([2,0,0], float)
        V0x = np.stack((2 - V0[:,0], V0[:,1], V0[:,2]), axis=-1)
        # print(f"bounding box after shift in x: min={V0x.min(axis=0)}, max={V0x.max(axis=0)}")
        assert V0.shape == V0x.shape

        ms_reflected = ml.MeshSet()
        ms_reflected.add_mesh(ml.Mesh(V0x, F0))
        ms_reflected.meshing_invert_face_orientation()
        quick_mesh_report(ms_reflected, i=0)

        orig_trimesh_mesh = trimesh.Trimesh(V0, F0)
        x_trimesh_mesh = trimesh.Trimesh(ms_reflected.current_mesh().vertex_matrix(), ms_reflected.current_mesh().face_matrix())
        orig_x_trimesh_mesh = trimesh.boolean.union([orig_trimesh_mesh, x_trimesh_mesh])
        # print("fused in x direction")
        # mesh_report(orig_x_trimesh_mesh, name="after fusing in x - blender", plot=True)
        orig_x_ms = ml.MeshSet()
        orig_x_ms.add_mesh(ml.Mesh(orig_x_trimesh_mesh.vertices, orig_x_trimesh_mesh.faces))
        quick_mesh_report(orig_x_ms, i=0)
        heal(orig_x_ms, manifold_method=1)
        # quick_mesh_report(orig_x_ms, i=0)

        # print(f"bounding box after fusing in x: min={orig_x_trimesh_mesh.vertices.min(axis=0)}, max={orig_x_trimesh_mesh.vertices.max(axis=0)}")

        # view_surface(orig_x_ms.current_mesh().vertex_matrix(), orig_x_ms.current_mesh().face_matrix(), title="after fusing in x - remeshlab", plane=True, planepoint=[1,0.5,0.5], planenormal=[1,0,0])
        # view_cropped(orig_x_ms.current_mesh().vertex_matrix(), orig_x_ms.current_mesh().face_matrix(), title="after fusing in x - remeshlab cropped", point_on_plane=(0.5,0.5,0.5), normal=(0,1,0), mode="surface")

        ## tetrahedralize the mesh 

        VV = np.asarray(orig_x_ms.current_mesh().vertex_matrix())
        FF = np.asarray(orig_x_ms.current_mesh().face_matrix())
        print("[TETGEN FOR ORIG+X MESH]")
        quick_mesh_report(orig_x_ms, i=0)

        tg = tetgen.TetGen(VV, FF)
        tg.tetrahedralize(order=1, mindihedral=10, minratio=1.5, steinerleft=0, maxvolume=0.001)
        nodes_tet = tg.node
        elems_tet = tg.elem
        surf = tg.f
        print(f"tetrahedral mesh: {nodes_tet.shape}, {elems_tet.shape}, surface: {surf.shape}")
        # print(f"min node index: {elems_tet.min()}, max node index: {elems_tet.max()}")
        # view_surface(nodes_tet, surf, title="tet mesh surface")

        # now in y 
        print("[UNION IN Y]")
        V0 = np.asarray(nodes_tet, float) #np.asarray(orig_x_ms.current_mesh().vertex_matrix(), float)
        F0 = np.asarray(surf) # np.asarray(orig_x_ms.current_mesh().face_matrix(), int)

        # V0 = normalize_vertices(V0)

        # print(f"bounding box: min={V0.min(axis=0)}, max={V0.max(axis=0)}")

        shift_y = np.array([0,2,0], float)
        V0y = np.stack((V0[:,0], 2 - V0[:,1], V0[:,2]), axis=-1)
        # print(f"bounding box after shift in y: min={V0y.min(axis=0)}, max={V0y.max(axis=0)}")
        assert V0.shape == V0y.shape

        ms_reflected = ml.MeshSet()
        ms_reflected.add_mesh(ml.Mesh(V0y, F0))
        ms_reflected.meshing_invert_face_orientation()
        print("[after reflecting in y]")
        quick_mesh_report(ms_reflected, i=0)

        orig_trimesh_mesh = trimesh.Trimesh(V0, F0)
        y_trimesh_mesh = trimesh.Trimesh(ms_reflected.current_mesh().vertex_matrix(), ms_reflected.current_mesh().face_matrix())
        orig_y_trimesh_mesh = trimesh.boolean.union([orig_trimesh_mesh, y_trimesh_mesh])
        # print("fused in y direction")
        # mesh_report(orig_y_trimesh_mesh, name="after fusing in y - blender", plot=True)
        orig_y_ms = ml.MeshSet()
        orig_y_ms.add_mesh(ml.Mesh(orig_y_trimesh_mesh.vertices, orig_y_trimesh_mesh.faces))
        quick_mesh_report(orig_y_ms, i=0)
        heal(orig_y_ms, manifold_method=0)
        # quick_mesh_report(orig_y_ms, i=0)
        print("[REMOVING SMALL CONNECTED COMPONENTS]")
        orig_y_ms.meshing_remove_connected_component_by_diameter()
        quick_mesh_report(orig_y_ms, i=0)

        # print(f"bounding box after fusing in y: min={orig_y_trimesh_mesh.vertices.min(axis=0)}, max={orig_y_trimesh_mesh.vertices.max(axis=0)}")

        # view_surface(orig_y_ms.current_mesh().vertex_matrix(), orig_y_ms.current_mesh().face_matrix(), title="after fusing in y - remeshlab", plane=False, planepoint=[1,0.5,0.5], planenormal=[1,0,0])
        # view_cropped(orig_y_ms.current_mesh().vertex_matrix(), orig_y_ms.current_mesh().face_matrix(), title="after fusing in y - remeshlab cropped", point_on_plane=(0.5,0.5,0.5), normal=(0,0,1), mode="surface")

        # now in z
        print("[UNION IN Z]")
        V0 = np.asarray(orig_y_ms.current_mesh().vertex_matrix(), float)
        F0 = np.asarray(orig_y_ms.current_mesh().face_matrix(), int)

        # V0 = normalize_vertices(V0)

        # print(f"bounding box: min={V0.min(axis=0)}, max={V0.max(axis=0)}")

        shift_z = np.array([0,0,2], float)
        V0z = np.stack((V0[:,0], V0[:,1], 2 - V0[:,2] - 1e-3), axis=-1)
        # print(f"bounding box after shift in z: min={V0z.min(axis=0)}, max={V0z.max(axis=0)}")
        assert V0.shape == V0z.shape

        ms_reflected = ml.MeshSet()
        ms_reflected.add_mesh(ml.Mesh(V0z, F0))
        ms_reflected.meshing_invert_face_orientation()
        print("[after reflecting in z]")
        quick_mesh_report(ms_reflected, i=0)

        orig_trimesh_mesh = trimesh.Trimesh(V0, F0)
        z_trimesh_mesh = trimesh.Trimesh(ms_reflected.current_mesh().vertex_matrix(), ms_reflected.current_mesh().face_matrix())
        orig_z_trimesh_mesh = trimesh.boolean.union([orig_trimesh_mesh, z_trimesh_mesh])
        print("fused in z direction")
        # mesh_report(orig_z_trimesh_mesh, name="after fusing in z - blender", plot=True)
        orig_z_ms = ml.MeshSet()
        orig_z_ms.add_mesh(ml.Mesh(orig_z_trimesh_mesh.vertices, orig_z_trimesh_mesh.faces))
        quick_mesh_report(orig_z_ms, i=0)
        heal(orig_z_ms, manifold_method=0)
        # print(f"bounding box after fusing in z: min={orig_z_trimesh_mesh.vertices.min(axis=0)}, max={orig_z_trimesh_mesh.vertices.max(axis=0)}")
        print("[MESHING REPAIR NON MANIFOLD VERTICES]")
        orig_z_ms.meshing_repair_non_manifold_vertices()
        '''

        ##### ALGORITHM B #####

        #### X
        print("[UNION IN X]")
        print("[WELDING]")
        tri_welded = vtk_weld_two(normalized_mesh, normalized_mesh_reflected_x, tol=1e-8)
        quick_mesh_report(tri_welded, i=0)
        # view_surface(tri_welded.vertices, tri_welded.faces, title="VTK welded") 
        # free originals used for the weld
        del normalized_mesh, normalized_mesh_reflected_x
        gc.collect()
        print(f"[PYMESHFIX CLEANING WELDED X]")
        clean_nodes3, clean_elems3 = pfix.clean_from_arrays(tri_welded.vertices, tri_welded.faces)
        m = trimesh.Trimesh(clean_nodes3, clean_elems3)
        x_report = quick_mesh_report(m, i=1)
        if not x_report["is_watertight"]:
            print(f"[PYMESHFIX CLEANING WELDED AGAIN]")
            clean_nodes_3, clean_elems_3 = pfix.clean_from_arrays(tri_welded.vertices, tri_welded.faces)
            m = trimesh.Trimesh(clean_nodes_3, clean_elems_3)
            quick_mesh_report(m, i=0)
        # no longer need the welded pair; we have m now
        del tri_welded
        gc.collect()

        ms_x = ml.MeshSet()
        ms_x.add_mesh(ml.Mesh(m.vertices, m.faces))
        heal(ms_x)
        tri_welded_x = trimesh.Trimesh(np.asarray(ms_x.current_mesh().vertex_matrix()), np.asarray(ms_x.current_mesh().face_matrix()), process=False)
        print("[FINAL X REPORT]")
        quick_mesh_report(tri_welded_x, i=0)
        del ms_x
        gc.collect()

        #### Y    
        print("[UNION IN Y]")
        normalized_mesh_reflected_y = tri_welded_x.copy()
        normalized_mesh_reflected_y.vertices[:,1] = 2 - tri_welded_x.vertices[:,1]
        ms_reflected_y = ml.MeshSet()
        ms_reflected_y.add_mesh(ml.Mesh(normalized_mesh_reflected_y.vertices, normalized_mesh_reflected_y.faces))
        heal(ms_reflected_y)
        normalized_mesh_reflected_y = trimesh.Trimesh(np.asarray(ms_reflected_y.current_mesh().vertex_matrix()), np.asarray(ms_reflected_y.current_mesh().face_matrix()), process=False)
        del ms_reflected_y
        gc.collect()
        print(f"[WELDING]")
        tri_welded_y = vtk_weld_two(tri_welded_x, normalized_mesh_reflected_y, tol=1e-8)
        quick_mesh_report(tri_welded_y, i=0)
        # free inputs for Y weld early
        del normalized_mesh_reflected_y
        del tri_welded_x
        gc.collect()
        print(f"[PYMESHFIX CLEANING WELDED Y]")
        clean_nodes4, clean_elems4 = pfix.clean_from_arrays(tri_welded_y.vertices, tri_welded_y.faces)
        m = trimesh.Trimesh(clean_nodes4, clean_elems4)
        y_report = quick_mesh_report(m, i=1)
        if not y_report["is_watertight"]:
            print(f"[PYMESHFIX CLEANING WELDED AGAIN]")
            clean_nodes_4, clean_elems_4 = pfix.clean_from_arrays(tri_welded_y.vertices, tri_welded_y.faces)
            m = trimesh.Trimesh(clean_nodes_4, clean_elems_4)
            quick_mesh_report(m, i=0)

        ms_y = ml.MeshSet()
        ms_y.add_mesh(ml.Mesh(m.vertices, m.faces))
        heal(ms_y)
        tri_welded_y = trimesh.Trimesh(np.asarray(ms_y.current_mesh().vertex_matrix()), np.asarray(ms_y.current_mesh().face_matrix()), process=False)
        print("[FINAL Y REPORT]")
        quick_mesh_report(tri_welded_y, i=1)
        del ms_y
        gc.collect()
        # view_surface(tri_welded_y.vertices, tri_welded_y.faces, title="VTK welded y")

        #### Z
        print("[UNION IN Z]")
        normalized_mesh_reflected_z = tri_welded_y.copy()
        normalized_mesh_reflected_z.vertices[:,2] = 2 - tri_welded_y.vertices[:,2]
        ms_reflected_z = ml.MeshSet()
        ms_reflected_z.add_mesh(ml.Mesh(normalized_mesh_reflected_z.vertices, normalized_mesh_reflected_z.faces))
        heal(ms_reflected_z)
        normalized_mesh_reflected_z = trimesh.Trimesh(np.asarray(ms_reflected_z.current_mesh().vertex_matrix()), np.asarray(ms_reflected_z.current_mesh().face_matrix()), process=False)
        del ms_reflected_z
        gc.collect()
        print(f"[WELDING]")
        quick_mesh_report(normalized_mesh_reflected_z, i=0)
        tri_welded_z = vtk_weld_two(tri_welded_y, normalized_mesh_reflected_z, tol=1e-8)
        quick_mesh_report(tri_welded_z, i=0) 
        print(f"[PYMESHFIX CLEANING WELDED Z]")
        clean_nodes4, clean_elems4 = pfix.clean_from_arrays(tri_welded_z.vertices, tri_welded_z.faces)
        m = trimesh.Trimesh(clean_nodes4, clean_elems4)
        z_report = quick_mesh_report(m, i=1)
        if not z_report["is_watertight"]:
            print(f"[PYMESHFIX CLEANING WELDED Z AGAIN]")
            clean_nodes4, clean_elems4 = pfix.clean_from_arrays(m.vertices, m.faces)
            m = trimesh.Trimesh(clean_nodes4, clean_elems4)
            quick_mesh_report(m, i=1)

        ms_z = ml.MeshSet()
        ms_z.add_mesh(ml.Mesh(m.vertices, m.faces))
        heal(ms_z)
        tri_welded_z = trimesh.Trimesh(np.asarray(ms_z.current_mesh().vertex_matrix()), np.asarray(ms_z.current_mesh().face_matrix()), process=False)
        print("[FINAL Z REPORT]")
        quick_mesh_report(tri_welded_z, i=1)
        # view_surface(tri_welded_z.vertices, tri_welded_z.faces, title="VTK welded z")

        ### Final normalization
        print(f"[NORMALIZING FINAL]")
        normalize_vertices_inplace(tri_welded_z.vertices)
        orig_z_ms = ml.MeshSet()
        orig_z_ms.add_mesh(ml.Mesh(tri_welded_z.vertices, tri_welded_z.faces))
        quick_mesh_report(orig_z_ms, i=0)

        VVV = orig_z_ms.current_mesh().vertex_matrix()
        FFF = orig_z_ms.current_mesh().face_matrix()
        
        #######
        print("---------------- Preliminary report ----------------")
        preliminary_report = quick_mesh_report(orig_z_ms, i=0)

        if preliminary_report["n_nonmanifold_edges"] > 0 or preliminary_report["n_nonmanifold_vertices"] > 0:
            orig_z_ms.meshing_repair_non_manifold_edges(method=0)
            orig_z_ms.meshing_repair_non_manifold_vertices()
            orig_z_ms.meshing_remove_null_faces()
            orig_z_ms.meshing_remove_unreferenced_vertices()
        if preliminary_report["connected_components"] > 1:
            print("[REMOVING SMALL CONNECTED COMPONENTS]")
            orig_z_ms.meshing_remove_connected_component_by_diameter()
        
        print("---------------- Final report ----------------")
        final_report = quick_mesh_report(orig_z_ms, i=0)
        
        if not final_report["is_watertight"]:
            print("WARNING: final stitched mesh is NOT WATERTIGHT!")
            flag_watertight = False
        else:
            flag_watertight = True

        if final_report["n_nonmanifold_edges"] > 0 or final_report["n_nonmanifold_vertices"] > 0:
            print("WARNING: final stitched mesh has NON-MANIFOLD edges or vertices!")
            flag_manifold = False
        else:
            flag_manifold = True

        if final_report["connected_components"] > 1:
            print("WARNING: final stitched mesh has MORE THAN ONE connected component!")
            flag_connected = False
        else:
            flag_connected = True

        tesselated_mesh = trimesh.Trimesh(np.asarray(orig_z_ms.current_mesh().vertex_matrix()), np.asarray(orig_z_ms.current_mesh().face_matrix()), process=False)

        # print("---------------- Visual check ----------------")
        # view_surface(orig_z_ms.current_mesh().vertex_matrix(), orig_z_ms.current_mesh().face_matrix(), title="after fusing in z - remeshlab", plane=False, planepoint=[1,0.5,0.5], planenormal=[1,0,0])
        # view_cropped(orig_z_ms.current_mesh().vertex_matrix(), orig_z_ms.current_mesh().face_matrix(), title="after fusing in z - remeshlab cropped", point_on_plane=(0.5,0.5,0.5), normal=(0,0,1), mode="surface")

        print("[FINAL STITCHED MESH]")
        quick_mesh_report(tesselated_mesh, i=1)

        # always save the final surface STL, even if tetrahedralization fails later
        final_stl_path = os.path.join(out_folder_name, "final_stitched.stl") 
        mesh = meshio.Mesh(points=orig_z_ms.current_mesh().vertex_matrix(), cells=[("triangle", orig_z_ms.current_mesh().face_matrix())])
        mesh.write(final_stl_path)
        print(f"saved final 2x2x2 RVE stl to: {final_stl_path}")
        del mesh

        # tetrahedralization with retry-after-clean fallback
        print("---------------- Tetrahedralization ----------------")
        tet_success = False
        volume = None
        pct_diff_vol = None
        tg = None
        try:
            tg = tetgen.TetGen(tesselated_mesh.vertices, tesselated_mesh.faces)
            tg.tetrahedralize(order=1, minratio=1.1, nobisect=False, mindihedral=10, steinerleft=0)
            print(f"tetgen output: nodes {tg.node.shape}, elems {tg.elem.shape}, faces {tg.f.shape}")
            tet_success = True
        except Exception as e:
            print(f"[TETGEN] First tetrahedralization failed: {e}")
            try:
                print("[TETGEN] Fallback: pymeshfix clean_from_arrays then retry")
                orig_n = int(tesselated_mesh.vertices.shape[0])
                Vc, Fc = pfix.clean_from_arrays(tesselated_mesh.vertices, tesselated_mesh.faces)
                if Vc is not None and Vc.shape[0] >= max(1, int(0.95 * orig_n)):
                    try:
                        tg = tetgen.TetGen(Vc, Fc)
                        tg.tetrahedralize(order=1, minratio=1.1, nobisect=False, mindihedral=10, steinerleft=0)
                        print(f"tetgen output (after fallback clean): nodes {tg.node.shape}, elems {tg.elem.shape}, faces {tg.f.shape}")
                        tet_success = True
                    except Exception as e2:
                        print(f"[TETGEN] Retry after pymeshfix also failed: {e2}")
                else:
                    print(f"[TETGEN] Fallback clean reduced vertices too much ({0 if Vc is None else Vc.shape[0]}/{orig_n}); skipping retry")
            except Exception as eclean:
                print(f"[TETGEN] Fallback clean_from_arrays failed: {eclean}")

        # if tet succeeded, save tet, compute volume, and compare with STL volume
        if tet_success and tg is not None:
            # save tet mesh as vtu and xdmf
            vtu_name = os.path.join(out_folder_name, "stitched_tetra.vtu")
            tg.write(vtu_name)
            vtumsh = meshio.read(vtu_name)
            print(vtumsh.points.shape)
            print(vtumsh.cells_dict)
            xdmf_name = os.path.join(out_folder_name, "stitched_tetra.xdmf")
            vtumsh.write(xdmf_name)
            del vtumsh

            with io.XDMFFile(MPI.COMM_WORLD, xdmf_name, "r") as xdmf:
                xdmfmesh = xdmf.read_mesh(name="Grid")
                dim = xdmfmesh.topology.dim
                fdim = dim - 1
            
            ### VOLUME COMPARING ###
            # integrate 1 to compare volumes
            V = fem.functionspace(xdmfmesh, ("Lagrange", 1))
            one = fem.Constant(xdmfmesh, PETSc.ScalarType(1.0))
            volume = fem.assemble_scalar(fem.form(one*ufl.dx))
            print(f"tet mesh volume: {volume}")

            pct_diff_vol = 100.0*(volume - tesselated_mesh.volume)/tesselated_mesh.volume
            print(f"% difference between stl and tetgen volume: {pct_diff_vol:.4f}%")
            # drop dolfinx objects holding references to xdmfmesh so it can be freed later
            del V, one
            gc.collect()
        else:
            # rename output folder to mark no tetra case and continue
            try:
                parent_dir = os.path.dirname(out_folder_name)
                base_name = os.path.basename(out_folder_name)
                if not base_name.startswith("no_tetra_"):
                    new_folder = os.path.join(parent_dir, f"no_tetra_{base_name}")
                    shutil.move(out_folder_name, new_folder)
                    # os.rename(out_folder_name, new_folder)
                    print(f"[TETGEN] Rename folder due to tetra failure: {out_folder_name} -> {new_folder}")
                    out_folder_name = new_folder
            except Exception as erename:
                print(f"[TETGEN] Failed to rename folder after tetra failure: {erename}")


        ### PERIODIC CHECKING ###

        _, _, n_ok_x, n_tot_x = preview_periodic_pairs(VVV, 'x', tol=1e-6)
        _, _, n_ok_y, n_tot_y = preview_periodic_pairs(VVV, 'y', tol=1e-6)
        _, _, n_ok_z, n_tot_z = preview_periodic_pairs(VVV, 'z', tol=1e-6)

        pct_all = 100.0*(n_ok_x + n_ok_y + n_ok_z) / (n_tot_x + n_tot_y + n_tot_z)
        if pct_all < 95.0: 
            print(f"WARNING: only {pct_all:.2f}% of periodic pairs are within tolerance!")
            flag_periodic = False
        else:
            flag_periodic = True


        ### AREA CHECKING ###
        area_x = area_of_flat_faces(VVV, FFF, 'x', 0.0, tol=1e-9)
        area_y = area_of_flat_faces(VVV, FFF, 'y', 0.0, tol=1e-9)
        area_z = area_of_flat_faces(VVV, FFF, 'z', 0.0, tol=1e-9)
        print(f"[LOOP2] area_x: {area_x}, area_y: {area_y}, area_z: {area_z}")

        area_ref_sq = 1.0
        area_xyz_mean = (area_x + area_y + area_z) / 3.0
        # ensure all areas are within +- 25% of the mean area
        if (abs(area_x - area_xyz_mean) > 0.25*area_ref_sq or
            abs(area_y - area_xyz_mean) > 0.25*area_ref_sq or
            abs(area_z - area_xyz_mean) > 0.25*area_ref_sq):
            print(f"WARNING: one of the areas is more than 25% different from the mean area {area_xyz_mean}!")
            flag_areas = False
        else:
            flag_areas = True
        
        # Save JSON summaries before potential cleanup
        print("---------------- Saving JSON summaries ----------------")
        options_dict = {
            "radbound": float(radbound),
            "resize_factor": float(resize_factor),
            "size_rve_um": int(size_rve),
            "slicing_engine": str(engine),
            "plane_cut_offset_d": float(d),
            "isotropic_h_cap": float(h_cap),
            "isotropic_iterations": int(iterations),
            "smoothing": {"lamb": 0.5, "iterations": 10},
            "tetgen": {"order": 1, "minratio": 1.1, "nobisect": False, "mindihedral": 10, "steinerleft": 0},
        }
        mesh_props = {
            "final_surface": {
                "n_verts": int(final_report["n_verts"]),
                "n_faces": int(final_report["n_faces"]),
                "area": float(final_report["area"]),
                "volume_trimesh": float(tesselated_mesh.volume) if tesselated_mesh.is_watertight else None,
                "watertight": bool(final_report["is_watertight"]),
                "winding_consistent": bool(final_report["is_winding_consistent"]),
                "nonmanifold_edges": int(final_report["n_nonmanifold_edges"]),
                "nonmanifold_vertices": int(final_report["n_nonmanifold_vertices"]),
                "connected_components": int(final_report["connected_components"])
            },
            "periodic_pairs": {
                "x": {"matched": int(n_ok_x), "total": int(n_tot_x)},
                "y": {"matched": int(n_ok_y), "total": int(n_tot_y)},
                "z": {"matched": int(n_ok_z), "total": int(n_tot_z)},
                "percent_all": float(pct_all)
            },
            "flat_face_areas": {"x0": float(area_x), "y0": float(area_y), "z0": float(area_z), "mean": float(area_xyz_mean)},
            "tet_volume": None if volume is None else float(volume),
            "tet_vs_stl_percent_diff": None if pct_diff_vol is None else float(pct_diff_vol),
            "flags": {
                "watertight": bool(flag_watertight),
                "manifold": bool(flag_manifold),
                "connected": bool(flag_connected),
                "periodic": bool(flag_periodic),
                "areas_ok": bool(flag_areas)
            }
        }
        save_json(os.path.join(out_folder_name, "meshing_options.json"), options_dict)
        save_json(os.path.join(out_folder_name, "mesh_properties.json"), mesh_props)

        ### FINAL CHECKING: check all flags are true, else delete folder and contents ###
        # only delete failed cases when tetra succeeded; for no-tetra, keep renamed folder for auditing
        if tet_success and not (flag_watertight and flag_manifold and flag_connected and flag_periodic and flag_areas):
            print("WARNING: one of the checks failed, prepending bad_ to the folder name")
            # shutil.rmtree(out_folder_name)
            # add prefix "bad_" to the folder name
            parent_dir = os.path.dirname(out_folder_name)
            base_name = os.path.basename(out_folder_name)
            if not base_name.startswith("bad_"):
                new_folder = os.path.join(parent_dir, f"bad_{base_name}")
                # os.rename(out_folder_name, new_folder)
                shutil.move(out_folder_name, new_folder)
                print(f"Rename bad folder: {out_folder_name} -> {new_folder}")
                out_folder_name = new_folder
        else:
            if tet_success:
                print("All checks passed!")
                completed_meshes += 1
            else:
                print("[TETGEN] Tetrahedralization failed; kept surface outputs and renamed folder (no_tetra_*)")
        print(f"[LOOP2] END {loop2_tag}")
        print("============================================================\n")
        # free per-RVE heavy objects (guard in case they were freed earlier)
        try:
            del tri_welded
        except Exception:
            pass
        try:
            del m, clean_nodes, clean_elems
        except Exception:
            pass
        try:
            del normalized_mesh, normalized_mesh_reflected_x
        except Exception:
            pass
        try:
            del tri_welded_x, normalized_mesh_reflected_y, tri_welded_y
        except Exception:
            pass
        try:
            del normalized_mesh_reflected_z, tri_welded_z
        except Exception:
            pass
        # free remaining analysis buffers
        try:
            del clean_nodes3, clean_elems3, clean_nodes_3, clean_elems_3
        except Exception:
            pass
        try:
            del clean_nodes4, clean_elems4, clean_nodes_4, clean_elems_4
        except Exception:
            pass
        try:
            del VVV, FFF, preliminary_report, final_report
        except Exception:
            pass
        try:
            del orig_z_ms, tesselated_mesh, tg, xdmfmesh
        except Exception:
            pass
        gc.collect()

        # ### PLOTTING ###
        # # Scale mesh to [0,1]³ for periodic boundary conditions
        # x = xdmfmesh.geometry.x
        # min_coords = np.min(x, axis=0)
        # max_coords = np.max(x, axis=0)

        # # Scale to [0,1]³
        # for i in range(3):
        #     x[:, i] = (x[:, i] - min_coords[i]) / (max_coords[i] - min_coords[i])

        # grid = tg.grid
        # grid.plot(show_edges=True)
        # cells = grid.cells.reshape(-1, 5)[:, 1:]
        # cell_center = grid.points[cells].mean(1)

        # mask = cell_center[:, 1] < 0.5
        # cell_ind = mask.nonzero()[0]
        # subgrid = grid.extract_cells(cell_ind)

        # # advanced plotting
        # plotter = pv.Plotter()
        # plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
        # plotter.add_mesh(tg.mesh, 'red', 'wireframe', opacity=0.2) # style = 'wireframe'
        # plotter.add_legend([[' Input Mesh ', 'r'],
        #                     [' tetrahedralize mesh ', 'black']])
        # plotter.show()

    _elapsed = time.time() - T0
    mins_elapsed, secs_elapsed = divmod(int(round(_elapsed)), 60)
    print(f"COMPLETED {completed_meshes} OUT OF {n_generated_samples} MESHES SUCCESSFULLY")
    print(f"execution time: {mins_elapsed}m {secs_elapsed}s")