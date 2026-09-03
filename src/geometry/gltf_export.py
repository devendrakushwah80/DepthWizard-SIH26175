"""
DepthWizard (SIH26175) — GLTF 2.0 Binary (.glb) & OBJ Exporter
Player 3: 3D Reconstruction & Visualization Lead

Exports calibrated 3D textured triangular meshes to compact web-ready .glb
and universal .obj formats with accompanying metadata JSON.
"""

import os
import json
import numpy as np
import trimesh
from PIL import Image
from src.geometry.raster_to_mesh import Mesh3D

def export_mesh_glb(mesh: Mesh3D, output_path: str, texture_quality: int = 85) -> dict:
    """
    Exports a Mesh3D to standard GLTF 2.0 Binary (.glb) format with embedded RGB texture.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Prepare texture image
    if mesh.texture_image is not None:
        tex = mesh.texture_image
        if tex.mode != 'RGB':
            tex = tex.convert('RGB')
    else:
        # Fallback default white texture
        tex = Image.new('RGB', (256, 256), color=(200, 200, 200))

    # Mesh3D stores Wavefront/OpenGL-style UVs (V=1 at image row 0).  Trimesh
    # converts that convention to glTF's upper-left origin when it serializes
    # the raw TEXCOORD_0 accessor.  Do not pre-flip here or the GLB would be
    # vertically mirrored in standards-compliant glTF viewers.
    visual = trimesh.visual.TextureVisuals(
        uv=mesh.uvs,
        image=tex
    )
    
    # Create Trimesh object
    tri_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.normals,
        visual=visual,
        process=False,
        validate=False
    )
    
    # Export to GLB binary
    glb_bytes = tri_mesh.export(file_type='glb')
    with open(output_path, 'wb') as f:
        f.write(glb_bytes)
        
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    # Metadata summary
    meta = dict(mesh.metadata)
    meta.update({
        'export_format': 'GLB (GLTF 2.0 Binary)',
        'file_path': output_path,
        'file_size_mb': file_size_mb,
        'texture_resolution': f"{tex.width}x{tex.height}",
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.faces)
    })
    
    # Save accompanying metadata JSON
    json_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(meta, jf, indent=2)
        
    return meta

def export_mesh_obj(mesh: Mesh3D, output_path: str) -> dict:
    """
    Exports a Mesh3D to standard Wavefront .obj and .mtl with PNG texture for CAD/GIS tools.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = os.path.dirname(os.path.abspath(output_path))
    
    tex_filename = f"{base_name}_texture.jpg"
    mtl_filename = f"{base_name}.mtl"
    
    tex_path = os.path.join(out_dir, tex_filename)
    mtl_path = os.path.join(out_dir, mtl_filename)
    
    # Save Texture
    if mesh.texture_image is not None:
        mesh.texture_image.save(tex_path, "JPEG", quality=90)
    else:
        Image.new('RGB', (256, 256), color=(200, 200, 200)).save(tex_path, "JPEG")
        
    # Save MTL
    with open(mtl_path, 'w', encoding='utf-8') as f:
        f.write(f"# DepthWizard SIH26175 Material\n")
        f.write(f"newmtl Material_0\n")
        f.write(f"Ka 1.000 1.000 1.000\n")
        f.write(f"Kd 1.000 1.000 1.000\n")
        f.write(f"Ks 0.000 0.000 0.000\n")
        f.write(f"d 1.0\n")
        f.write(f"illum 1\n")
        f.write(f"map_Kd {tex_filename}\n")
        
    # Save OBJ
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# DepthWizard SIH26175 3D Mesh\n")
        f.write(f"mtllib {mtl_filename}\n")
        
        # Vertices
        for v in mesh.vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        # Texture Coords
        for uv in mesh.uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            
        # Normals
        for vn in mesh.normals:
            f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
            
        # Faces (1-indexed: v/vt/vn)
        f.write("usemtl Material_0\n")
        f.write("s 1\n")
        for face in mesh.faces:
            i0, i1, i2 = face + 1
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
            
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return {
        'export_format': 'Wavefront OBJ',
        'file_path': output_path,
        'mtl_path': mtl_path,
        'tex_path': tex_path,
        'file_size_mb': file_size_mb,
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.faces)
    }
