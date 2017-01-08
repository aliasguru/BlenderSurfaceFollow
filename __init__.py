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

import bpy



def register():
	bpy.utils.register_module()

def unregister():
	bpy.utils.unregister_module()

if __name__ == '__main__':
	register()
