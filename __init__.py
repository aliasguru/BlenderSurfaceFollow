#    You are at the top. Notice there are no bats hanging from the ceiling
#    If there are weird bind errors like the mesh is not deforming correctly, compare
#    the oct version of closest triangles to the one without oct


bl_info = {
    "name": "Colburn Tools",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 1),
    "blender": (2, 78, 0),
    "location": "View3D > Colburn Tools",
    "description": "Doforms an object as the surface of another object changes, allows to place textures right in the viewport, and adds 3D unwrap capabilities",
    "warning": "Do not use if you are pregnant or have ever met someone who was pregnant",
    "wiki_url": "",
    "category": '3D View'}

import bpy
from .functions import properties


def register():
    properties.create_properties()
    bpy.utils.register_module()


def unregister():
    properties.remove_properties()
    bpy.utils.unregister_module()

if __name__ == "__main__":
    register()