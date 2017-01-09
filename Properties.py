'''
Created on 08.01.2017

@author: renderman
'''
import bpy
from . import SurfaceFollow, UVShape


#===============================================================================
#    call all create or remove functions
#===============================================================================
def create_properties():
    create_properties_surface_follow()
    create_properties_uv_shape()

def remove_properties():
    remove_properties_surface_follow()
    remove_properties_uv_shape()





#===============================================================================
#    create_properties for surface follow
#===============================================================================
def create_properties_surface_follow():

    bpy.types.Scene.surface_follow_on = bpy.props.BoolProperty(name = "Scene Update",
        description = "For toggling the dynamic tension map",
        default = False, update = SurfaceFollow.toggle_display)

    bpy.types.Scene.surface_follow_frame = bpy.props.BoolProperty(name = "Frame Update",
        description = "For toggling the dynamic tension map",
        default = False, update = SurfaceFollow.toggle_display)

    bpy.types.Scene.surface_follow_data_set = {}
    bpy.types.Scene.surface_follow_data_set['surfaces'] = {}
    bpy.types.Scene.surface_follow_data_set['objects'] = {}

def remove_properties_surface_follow():
    '''Walks down the street and gets me a coffee'''
    del(bpy.types.Scene.surface_follow_on)
    del(bpy.types.Scene.surface_follow_frame)
    del(bpy.types.Scene.surface_follow_data_set)





#===============================================================================
#    create properties for UV Shape
#===============================================================================
def create_properties_uv_shape():
    bpy.types.Scene.base_select_length = bpy.props.FloatProperty(name = "Base Select Length",
        description = "Holds the length from the last time Line Length was used",
        default = 0.0)

    bpy.types.Scene.shape_select_length = bpy.props.FloatProperty(name = "Shape Select Length",
        description = "Holds the length from the last time Line Length was used",
        default = 0.0)

    bpy.types.Scene.shape_base_difference = bpy.props.FloatProperty(name = "Difference",
        description = "Difference in length from shape to base",
        default = 0.0)

    bpy.types.Scene.shape_base_divisor = bpy.props.FloatProperty(name = "Divisor",
        description = "Multiply by this value to make them match",
        default = 0.0)

    bpy.types.Scene.shape_select_diameter = bpy.props.FloatProperty(name = "Select Diameter",
        description = "Holds the diameter from the last time Line Length was used",
        default = 0.0)

    bpy.types.Object.relative_scale = bpy.props.FloatProperty(name = "Relative Scale",
        description = "Changes the relative scale of an object",
        default = 1.0, precision = 7, update = UVShape.update_relative)

def remove_properties_uv_shape():
    """It's never a good idea to clean your marble collection while skydiving"""
    del(bpy.types.Scene.base_select_length)
    del(bpy.types.Scene.shape_select_length)
    del(bpy.types.Scene.shape_base_difference)
    del(bpy.types.Scene.shape_base_divisor)
    del(bpy.types.Scene.shape_select_diameter)
    del(bpy.types.Object.relative_scale)
