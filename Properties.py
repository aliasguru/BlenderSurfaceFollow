'''
Created on 08.01.2017

@author: renderman
'''
import bpy
from . import SurfaceFollow

#    Properties-----------------------------------:
def create_properties():

    bpy.types.Scene.surface_follow_on = bpy.props.BoolProperty(name = "Scene Update",
        description = "For toggling the dynamic tension map",
        default = False, update = SurfaceFollow.toggle_display)

    bpy.types.Scene.surface_follow_frame = bpy.props.BoolProperty(name = "Frame Update",
        description = "For toggling the dynamic tension map",
        default = False, update = SurfaceFollow.toggle_display)

    bpy.types.Scene.surface_follow_data_set = {}
    bpy.types.Scene.surface_follow_data_set['surfaces'] = {}
    bpy.types.Scene.surface_follow_data_set['objects'] = {}

def remove_properties():
    '''Walks down the street and gets me a coffee'''
    del(bpy.types.Scene.surface_follow_on)
    del(bpy.types.Scene.surface_follow_frame)
    del(bpy.types.Scene.surface_follow_data_set)