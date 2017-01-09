'''
Created on 09.01.2017

@author: r.trummer
'''
from bpy.types import Panel
from . import SurfaceFollow

class SurfaceFollowPanel(Panel):
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
        col.operator(SurfaceFollow.BindToSurface.bl_idname, text = "Bind to Surface")
        col.operator(SurfaceFollow.UpdateOnce.bl_idname, text = "Update Once", icon = 'RECOVER_AUTO')
        if not context.scene.surface_follow_frame:
            col.prop(context.scene , "surface_follow_on", text = "Scene Update", icon = 'SCENE_DATA')
        if not context.scene.surface_follow_on:
            col.prop(context.scene , "surface_follow_frame", text = "Frame Update", icon = 'PLAY')



class Print3DTools(Panel):
    """Creates a new tab with physics UI"""
    bl_label = "3D Print Tools"
    bl_idname = "3D Print Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text = "UV Tools")
        col.operator("object.shape_from_uv", text = "Create UV Shape", icon = 'SHAPEKEY_DATA')
        #    col.operator("object.update_line_lengths", text="Update Measurements", icon='FILE_REFRESH')
        #    col.prop(bpy.context.scene, "base_select_length", text="Base Select Length", icon='FORCE_HARMONIC')
        #    col.prop(bpy.context.scene, "shape_select_length", text="Shape Select Length", icon='FORCE_HARMONIC')
        #    col.prop(bpy.context.scene, "shape_base_difference", text="Difference", icon='FORCE_HARMONIC')
        #    col.prop(bpy.context.scene, "shape_base_divisor", text="Divisor", icon='FORCE_HARMONIC')
        #    if bpy.context.object != None:
            #    col.prop(bpy.context.object, "relative_scale", text="Relative Scale", icon='FORCE_HARMONIC')