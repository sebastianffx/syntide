"""Syntide convergence sculpture, authored in Blender 5.1.

Run: blender -b --python generate_convergence.py
Synthetic trajectories are illustrative, not real measurements.
"""
from pathlib import Path
import math
import bpy
from mathutils import Vector

OUT = Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Scene-referred color values derived from the Syntide explorer palette.
def linear_hex(value):
    v = [int(value[i:i+2],16)/255 for i in (0,2,4)]
    return tuple(c/12.92 if c < .04045 else ((c+.055)/1.055)**2.4 for c in v) + (1,)

def material(name, color, metallic=.0, roughness=.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = linear_hex(color)
    p.inputs['Metallic'].default_value = metallic
    p.inputs['Roughness'].default_value = roughness
    return m

ink = material('Deep ink | #192e3b', '192e3b', .25, .38)
real = material('Real signal | teal', '21c8cb', .58, .22)
p = real.node_tree.nodes.get('Principled BSDF')
p.inputs['Emission Color'].default_value = linear_hex('087f8c')
p.inputs['Emission Strength'].default_value = .18

synthetic = material('Synthetic trajectories | amber into teal', 'bd6a20', .72, .25)
nodes = synthetic.node_tree.nodes
links = synthetic.node_tree.links
geo = nodes.new('ShaderNodeNewGeometry')
sep = nodes.new('ShaderNodeSeparateXYZ')
mapn = nodes.new('ShaderNodeMapRange')
mapn.inputs['From Min'].default_value = -1.6
mapn.inputs['From Max'].default_value = 5.6
ramp = nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].position = .0
ramp.color_ramp.elements[0].color = linear_hex('d99850')
ramp.color_ramp.elements[1].position = 1.0
ramp.color_ramp.elements[1].color = linear_hex('21c8cb')
e = ramp.color_ramp.elements.new(.43)
e.color = linear_hex('cfbb91')
links.new(geo.outputs['Position'],sep.inputs[0])
links.new(sep.outputs['X'],mapn.inputs['Value'])
links.new(mapn.outputs['Result'],ramp.inputs[0])
links.new(ramp.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color'])

# A complex but smooth, non-periodic reference is shared by all trajectories.
def reference(x):
    return (.76*math.sin(.93*x + .2) + .34*math.sin(1.71*x-.9)
            + .19*math.sin(2.87*x+.3)
            + .50*math.exp(-((x-4.50)/.40)**2)
            - .32*math.exp(-((x-5.12)/.28)**2))

def tube(name, coords, radius, mat):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2
    curve.bevel_depth = radius
    curve.bevel_resolution = 5
    curve.resolution_u = 2
    curve.fill_mode = 'FULL'
    spl = curve.splines.new('POLY')
    spl.points.add(len(coords)-1)
    for p, c in zip(spl.points, coords):
        p.co = (*c,1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj

count = 15
for i in range(count):
    u = (i-(count-1)/2)/((count-1)/2)
    phase = i*.55
    coords=[]
    for j in range(1001):
        t = j/1000
        x = -8.0 + 16*t
        # Smoothly suppress shape and phase error while keeping a tiny separation.
        decay = (1-t)**2.15
        err = 1.9*u + .32*math.sin(1.1*x+phase) + .13*math.cos(2.1*x-phase)
        z = reference(x) + decay*err + .016*u
        y = .42 + decay*(.40 + .64*math.cos(i*.63)) + .055*u
        coords.append((x,y,z))
    tube('Synthetic %02d | converging trajectory'%(i+1), coords, .027, synthetic)

coords = [(x, -.13, reference(x)) for x in [-8+16*j/1400 for j in range(1401)]]
tube('Real signal | continuous reference',coords,.071,real)

# A low uninterrupted plinth gives the sculpture a physical shadow without chart axes.
bpy.ops.mesh.primitive_plane_add(size=200, location=(0,0,-3.10))
plane=bpy.context.object
plane.name='Seamless deep ink ground'
plane.data.materials.append(ink)

world=bpy.data.worlds.new('Ink studio')
bpy.context.scene.world=world
world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Color'].default_value = linear_hex('607e89')
world.node_tree.nodes['Background'].inputs['Strength'].default_value = .20

def area(name, location, energy, color, size, target=(0,0,0), size_y=None):
    data=bpy.data.lights.new(name,'AREA')
    data.energy=energy
    data.color=tuple(int(color[i:i+2],16)/255 for i in (0,2,4))
    data.shape='RECTANGLE'
    data.size=size
    data.size_y=size_y if size_y is not None else size
    obj=bpy.data.objects.new(name,data)
    bpy.context.collection.objects.link(obj)
    obj.location=location
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
    return obj

area('Long softbox | pearl',(0,-4,8),1900,'e3f8ff',13,size_y=5)
area('Copper edge | left',(-8,1,4),1050,'ffc685',7,size_y=5)
area('Teal edge | right',(8,3,5),1500,'73f2ec',7,size_y=4)
area('Front reflection',(0,-8,2),650,'c4f2ff',10,size_y=2)

bpy.ops.object.camera_add(location=(1.2,-18,9.0))
cam=bpy.context.object
cam.name='Syntide hero camera'
cam.rotation_euler=(Vector((0,.25,-.30))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO'
cam.data.ortho_scale=19.8
scene=bpy.context.scene
scene.camera=cam
scene.render.engine='CYCLES'
scene.cycles.samples=64
scene.cycles.use_denoising=True
scene.render.resolution_x=2400
scene.render.resolution_y=1350
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGB'
scene.render.image_settings.color_depth='8'
scene.render.filepath=str(OUT/'syntide-convergence.png')
scene.view_settings.view_transform='AgX'
scene.view_settings.look='AgX - Medium High Contrast'
scene.render.film_transparent=False
scene['brand']='Syntide'
scene['concept']='Illustrative synthetic trajectories converge in shape toward a continuous real-data reference.'
scene['palette']='Ink #192e3b, real teal #087f8c, synthetic amber #bd6a20; illuminated metallic variants.'
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'syntide-convergence.blend'))
bpy.ops.render.render(write_still=True)
