'''
Created on 08.01.2017

@author: renderman
'''


bl_info = {
    "name": "Colburn Tools",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 1),
    "blender": (2, 78, 0),
    "location": "View3D > Extended Tools > Colburn Tools",
    "description": "Doforms an object as the surface of another object changes",
    "warning": "Do not use if you are pregnant or have ever met someone who was pregnant",
    "wiki_url": "",
    "category": '3D View'}


if 'bpy' in locals():
	import importlib
	for m in [Gizmo, Properties, SurfaceFollow, TextureHack, UVShape, UI]:    #    @UndefinedVariable
		importlib.reload(m)

else:
	import bpy
	from . import Properties, UI



def register():
	Properties.create_properties()
	bpy.utils.register_module(__name__)

def unregister():
	Properties.remove_properties()
	bpy.utils.unregister_module(__name__)

if __name__ == '__main__':
	register()
