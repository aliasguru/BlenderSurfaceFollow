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


