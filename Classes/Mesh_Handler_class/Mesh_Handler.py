import trimesh
from trimesh.graph import connected_components
from pathlib import Path
import numpy as np

class Mesh_Handler:
    
    def __init__(self):
        pass


    def rescale_meshes(self, meshes_path, real_size, target_size):
        meshes_path = Path(meshes_path)

        size_y = real_size
        target_y = target_size
        scale = target_y / size_y

        # New folder, same parent as mesh_path
        new_folder = meshes_path.parent / (meshes_path.name + "_rescaled")
        new_folder.mkdir(exist_ok=True)

        for mesh_file in meshes_path.iterdir():
            if not mesh_file.is_file():
                continue

            mesh = trimesh.load(str(mesh_file))
            path = mesh_file

            new_path = new_folder / path.name

            if new_path.exists():
                continue

            # Rescale based on known diameter to wanted diameter
            mesh.apply_scale(scale)

            mesh.export(new_path)

        return new_folder
    
    def center_meshes(self, meshes):
        meshes = Path(meshes)

        fibre_meshes = [trimesh.load(m) for m in meshes.iterdir() if m.is_file()]

        base_path = meshes
        moved_folder = base_path.with_name(base_path.name + "_moved")
        moved_folder.mkdir(exist_ok=True)

        all_vertices = np.vstack([m.vertices for m in fibre_meshes])

        min_bounds = all_vertices.min(axis=0)
        max_bounds = all_vertices.max(axis=0)

        group_center = (min_bounds + max_bounds) / 2
        translation = -group_center

        i = 1
        for m in fibre_meshes:
            m.apply_translation(translation)
            m.export(moved_folder / f"fibre_{i}.stl")
            i += 1
        
        return moved_folder

    def split_mesh(self, mesh_path, components_name="component", skip=0):
        mesh = trimesh.load(mesh_path)
        path = Path(mesh_path)
        # 1. Center entire model (bbox)
        mesh.apply_translation(-mesh.bounding_box.centroid)

        # 2. Compute face-connected components
        adjacency = mesh.face_adjacency
        components = connected_components(edges=adjacency,
                                        nodes=np.arange(len(mesh.faces)),
                                        min_len=1)

        submeshes = []
        for comp in components:
            faces = mesh.faces[comp]
            # Use slicing, which keeps original coordinates
            sub = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                                faces=faces)
            submeshes.append(sub)

        # 3. Save each component
        for i, comp in enumerate(submeshes):
            if i > skip:
                new_path = path.parent / f"{components_name}_{i}.stl"
                if new_path.exists() != True:
                    comp.export(new_path)
    
     