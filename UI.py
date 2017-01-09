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


class MoveTexturePanel(Panel):
    """Creates Buttons for Line Tools"""
    bl_label = "Adjust Active Texture"
    bl_idname = "drag_textures"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"

    def draw(self, context):
        layout = self.layout
        #    col = layout.column(align = True)
        #    col.operator("object.reset_all", text="Reset All Adjustments", icon='RECOVER_LAST')
        col = layout.column(align = True)

        move_text = "Move Texture"
        if context.scene.tex_move_alert:
            move_text = 'Right click to exit'
            col.alert = True

        col.operator("view3d.texture_move_modal", text = move_text, icon = 'HAND')
        col.operator("object.reset_move", text = "Reset Move", icon = 'RECOVER_LAST')
        col.label(text = "Left Shift for X Y")
        col = layout.column(align = True)

        scale_text = "Scale Texture"
        if context.scene.tex_scale_alert:
            scale_text = 'Right click to exit'
            col.alert = True
        #    col.label(text="Right Click To Exit")
        col.operator("view3d.texture_scale_modal", text = scale_text, icon = 'MOD_ARRAY')
        col.prop(context.scene , "scale_strength", text = "Scale Strength", slider = True)
        col.operator("object.reset_scale", text = "Reset Scale", icon = 'RECOVER_LAST')
        col.label(text = "Left Shift for X Y")
        col.label(text = "Left Ctrl for Free")
        #    col.label(text="Right Click To Exit")
        col = layout.column(align = True)

        rotate_text = "Rotate Texture"
        if context.scene.tex_rotate_alert:
            rotate_text = 'Right click to exit'
            col.alert = True
        col.operator("view3d.texture_rotate_modal", text = rotate_text, icon = 'FILE_REFRESH')
        #    col.operator("object.reset_rotation", text="Reset Rotation", icon='RECOVER_LAST')
        col.label(text = "Left Shift For Precise")
        col.label(text = "Left Ctrl for increment")
        #    col.label(text="Right Click To Exit")
