'''
Created on 08.01.2017

@author: renderman
'''
import bpy
from ..functions import surface_follow as f
from ..functions import properties

class BindToSurface(bpy.types.Operator):
    '''Bind To Surface'''
    bl_idname = "scene.bind_to_surface"
    bl_label = "bind to surface"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        x = f.multi_bind()
        if x == -1:
            self.report({'ERROR'}, 'Select at least two objects')
        return {'FINISHED'}

class ToggleSurfaceFollow(bpy.types.Operator):
    '''Toggle Surface Follow Update'''
    bl_idname = "scene.toggle_surface_follow"
    bl_label = "surface follow updater"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        properties.toggle_display()
        return {'FINISHED'}

class UpdateOnce(bpy.types.Operator):
    '''Surface Update'''
    bl_idname = "scene.surface_update_once"
    bl_label = "update surface one time"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        properties.run_handler(None, True)
        return {'FINISHED'}

class SurfaceFollowPanel(bpy.types.Panel):
    """Surface Follow Panel"""
    bl_label = "Surface Follow Panel"
    bl_idname = "Surface Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"
    #    gt_show = True

    def draw(self, context):
        layout = self.layout
        col = layout.column(align = True)
        col.label(text = "Surface Follow")
        col.operator("scene.bind_to_surface", text = "Bind to Surface")
        col.operator("scene.surface_update_once", text = "Update Once", icon = 'RECOVER_AUTO')
        if not bpy.context.scene.surface_follow_frame:    #    @UndefinedVariable
            col.prop(bpy.context.scene , "surface_follow_on", text = "Scene Update", icon = 'SCENE_DATA')
        if not bpy.context.scene.surface_follow_on:    #    @UndefinedVariable
            col.prop(bpy.context.scene , "surface_follow_frame", text = "Frame Update", icon = 'PLAY')

