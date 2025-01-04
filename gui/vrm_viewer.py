# gui/vrm_viewer.py
import logging
import numpy as np
import json
import ctypes
from PyQt5 import QtCore  
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPointF
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GLU import *
from utils.model_parser import ModelParser
import os

class VRMViewer(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vrm_data = None
        self.rotation = 0.0
        self.model_data = None
        self.textures = {}
        self.vbos = []
        
        self.zoom_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        self.debug = True

        # Initialize animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_rotation)
        self.timer.start(30)  

        # Camera parameters
        self.camera_distance = 10.0
        self.camera_height = 0.0
        self.camera_rotation = 0.0

        # Mouse state
        self.mouse_drag = False
        self.last_x = 0
        self.last_y = 0
        self.rotation_x = 0
        self.rotation_y = 0
        self.translation_z = 0

        # Initialize zoom factor
        self.zoom_factor = 1.0
        
        self.materials = {}
        self.mesh_materials = {}  # Map meshes to materials
        
    def parse_materials(self, gltf_json):
        """Extract material data from GLTF/VRM file with enhanced texture and shader support"""
        materials = {}
        print("Parsing materials...")
        
        if 'materials' not in gltf_json:
            print("No materials found in file, using default")
            materials['default'] = {
                'baseColor': [0.8, 0.8, 0.8, 1.0],
                'metallicFactor': 0.0,
                'roughnessFactor': 1.0,
                'emissiveFactor': [0, 0, 0],
                'alphaMode': 'OPAQUE'
            }
            return materials

        for idx, material in enumerate(gltf_json['materials']):
            print(f"\nParsing material {idx}: {material.get('name', 'unnamed')}")
            mat_data = {}
            
            # Store material name for debugging
            mat_data['name'] = material.get('name', f'material_{idx}')
            
            # Handle VRM extensions first as they might override standard properties
            if 'extensions' in material:
                if 'VRM' in material['extensions']:
                    vrm_mat = material['extensions']['VRM']
                    
                    # Handle VRM shader properties
                    if 'shader' in vrm_mat:
                        mat_data['shader'] = vrm_mat['shader']
                    
                    # Handle VRM color properties
                    if 'vectorProperties' in vrm_mat:
                        vec_props = vrm_mat['vectorProperties']
                        
                        # Main color override
                        if '_Color' in vec_props:
                            mat_data['baseColor'] = vec_props['_Color']
                            print(f"VRM main color: {mat_data['baseColor']}")
                        
                        # Additional color properties
                        if '_ShadeColor' in vec_props:
                            mat_data['shadeColor'] = vec_props['_ShadeColor']
                            print(f"VRM shade color: {mat_data['shadeColor']}")
                            
                        if '_RimColor' in vec_props:
                            mat_data['rimColor'] = vec_props['_RimColor']
                            print(f"VRM rim color: {mat_data['rimColor']}")
                    
                    # Handle VRM texture properties
                    if 'textureProperties' in vrm_mat:
                        tex_props = vrm_mat['textureProperties']
                        for tex_name, tex_idx in tex_props.items():
                            mat_data[f'vrm_{tex_name}'] = {'index': tex_idx}
                            print(f"VRM texture {tex_name}: {tex_idx}")
                    
                    # Handle VRM float properties
                    if 'floatProperties' in vrm_mat:
                        float_props = vrm_mat['floatProperties']
                        for prop_name, value in float_props.items():
                            mat_data[f'vrm_{prop_name}'] = value
                            print(f"VRM float property {prop_name}: {value}")

            # Handle standard PBR properties if not overridden by VRM
            if 'pbrMetallicRoughness' in material and 'baseColor' not in mat_data:
                pbr = material['pbrMetallicRoughness']
                
                # Base color and texture
                mat_data['baseColor'] = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
                print(f"Base color: {mat_data['baseColor']}")
                
                if 'baseColorTexture' in pbr:
                    tex_info = pbr['baseColorTexture']
                    mat_data['baseColorTexture'] = {
                        'index': tex_info['index'],
                        'texCoord': tex_info.get('texCoord', 0)
                    }
                    
                    # Handle texture transform if present
                    if 'extensions' in tex_info and 'KHR_texture_transform' in tex_info['extensions']:
                        transform = tex_info['extensions']['KHR_texture_transform']
                        mat_data['baseColorTexture']['transform'] = {
                            'offset': transform.get('offset', [0, 0]),
                            'rotation': transform.get('rotation', 0),
                            'scale': transform.get('scale', [1, 1])
                        }
                    print(f"Base color texture: {mat_data['baseColorTexture']}")
                
                # Metallic/Roughness
                mat_data['metallicFactor'] = pbr.get('metallicFactor', 1.0)
                mat_data['roughnessFactor'] = pbr.get('roughnessFactor', 1.0)
                
                if 'metallicRoughnessTexture' in pbr:
                    mat_data['metallicRoughnessTexture'] = {
                        'index': pbr['metallicRoughnessTexture']['index'],
                        'texCoord': pbr['metallicRoughnessTexture'].get('texCoord', 0)
                    }

            # Normal map
            if 'normalTexture' in material:
                mat_data['normalTexture'] = {
                    'index': material['normalTexture']['index'],
                    'scale': material['normalTexture'].get('scale', 1.0),
                    'texCoord': material['normalTexture'].get('texCoord', 0)
                }
                print(f"Normal map: {mat_data['normalTexture']}")

            # Occlusion
            if 'occlusionTexture' in material:
                mat_data['occlusionTexture'] = {
                    'index': material['occlusionTexture']['index'],
                    'strength': material['occlusionTexture'].get('strength', 1.0),
                    'texCoord': material['occlusionTexture'].get('texCoord', 0)
                }

            # Emissive properties
            if 'emissiveTexture' in material:
                mat_data['emissiveTexture'] = {
                    'index': material['emissiveTexture']['index'],
                    'texCoord': material['emissiveTexture'].get('texCoord', 0)
                }
            mat_data['emissiveFactor'] = material.get('emissiveFactor', [0, 0, 0])
            print(f"Emissive factor: {mat_data['emissiveFactor']}")

            # Alpha properties
            mat_data['alphaMode'] = material.get('alphaMode', 'OPAQUE')
            if mat_data['alphaMode'] == 'MASK':
                mat_data['alphaCutoff'] = material.get('alphaCutoff', 0.5)
            print(f"Alpha mode: {mat_data['alphaMode']}")

            # Additional properties
            mat_data['doubleSided'] = material.get('doubleSided', False)
            mat_data['unlit'] = 'KHR_materials_unlit' in material.get('extensions', {})

            materials[str(idx)] = mat_data

        print(f"\nTotal materials parsed: {len(materials)}")
        return materials
        
    def cleanup_buffers(self):
        """Clean up OpenGL buffers"""
        try:
            if self.vbos:
                print(f"Cleaning up {len(self.vbos)} VBO sets")
                for mesh_vbos in self.vbos:
                    for _, vbo in mesh_vbos:
                        gl.glDeleteBuffers(1, [vbo])
                self.vbos = []
        except Exception as e:
            print(f"Error cleaning up buffers: {str(e)}")
            
    def initialize_buffers(self):
        """Initialize OpenGL buffers with validation"""
        print("Starting buffer initialization...")
        self.cleanup_buffers() 
        
        try:
            for mesh_idx, mesh in enumerate(self.model_data):
                mesh_vbos = []
                print(f"Processing mesh {mesh_idx}")

                # Vertex buffer
                if 'vertices' in mesh:
                    vbo = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
                    vertices = np.array(mesh['vertices'], dtype=np.float32)
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
                    mesh_vbos.append(('vertices', vbo))
                    print(f"Created vertex buffer: {vbo}")

                # Normal buffer
                if 'normals' in mesh:
                    normal_vbo = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, normal_vbo)
                    normals = np.array(mesh['normals'], dtype=np.float32)
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STATIC_DRAW)
                    mesh_vbos.append(('normals', normal_vbo))
                    print(f"Created normal buffer: {normal_vbo}")

                # Index buffer
                if 'indices' in mesh:
                    ibo = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
                    indices = np.array(mesh['indices'], dtype=np.uint32)
                    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
                    mesh_vbos.append(('indices', ibo))
                    print(f"Created index buffer: {ibo}")

                self.vbos.append(mesh_vbos)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            print("Buffer initialization complete")

        except Exception as e:
            print(f"Error initializing buffers: {str(e)}")
            self.cleanup_buffers()
            raise


    def wheelEvent(self, event):
        # Get scroll direction and amount
        delta = event.angleDelta().y() / 120  # Normalize wheel delta
        
        # Use a larger zoom factor for more dramatic zoom
        zoom_step = 1.2 if delta > 0 else 0.8333  # 1/1.2 for smooth zoom out
        
        # Apply zoom with cursor position as focal point
        cursor_pos = event.position()
        
        # Store pre-zoom viewport coordinates
        pre_zoom_pos = self.screen_to_world(cursor_pos)
        
        # Apply zoom
        self.zoom_factor *= zoom_step
        
        # what do your elf eyes see
        self.zoom_factor = max(0.01, min(self.zoom_factor, 100.0))
        
        # Adjust view to keep cursor point stable
        post_zoom_pos = self.screen_to_world(cursor_pos)
        
        # Adjust offset to maintain cursor position
        self.offset_x += (post_zoom_pos.x() - pre_zoom_pos.x())
        self.offset_y += (post_zoom_pos.y() - pre_zoom_pos.y())
        
        # Optional: Add smooth animation for zoom
        if hasattr(self, 'zoom_animation'):
            self.zoom_animation.stop()
        
        # Trigger redraw
        self.updateGL()
        
        # Emit zoom level for status updates if needed
        if hasattr(self, 'zoom_changed'):
            self.zoom_changed.emit(self.zoom_factor)

    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        x = (screen_pos.x() - self.width() / 2) / self.zoom_factor + self.offset_x
        y = (screen_pos.y() - self.height() / 2) / self.zoom_factor + self.offset_y
        return QPointF(x, y)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_drag = True
            self.last_x = event.x()
            self.last_y = event.y()
            self.is_right_click = False  
        elif event.button() == QtCore.Qt.RightButton:
            self.mouse_drag = True
            self.last_x = event.x()
            self.last_y = event.y()
            self.is_right_click = True  

    def mouseMoveEvent(self, event):
        if not self.mouse_drag:
            return

        dx = event.x() - self.last_x
        dy = event.y() - self.last_y

        if self.is_right_click:  
            self.rotation_x += dy * 0.5  
            self.rotation_y += dx * 0.5
        else:  
            self.translation_z += dy * 0.05  

        self.last_x = event.x()
        self.last_y = event.y()

        self.update()  

    def mouseReleaseEvent(self, event):
        if event.button() in [QtCore.Qt.LeftButton, QtCore.Qt.RightButton]:
            self.mouse_drag = False


    def load_vrm(self, vrm_path):
        """Load a VRM file with debug output"""
        try:
            print(f"Loading VRM file: {vrm_path}")
            with open(vrm_path, 'rb') as f:
                # Read GLB/VRM header
                magic = f.read(4)
                if magic != b'glTF':
                    raise ValueError("Invalid GLB/VRM file format")

                version = int.from_bytes(f.read(4), 'little')
                length = int.from_bytes(f.read(4), 'little')
                print(f"VRM version: {version}, length: {length}")

                # Read JSON 
                json_chunk_length = int.from_bytes(f.read(4), 'little')
                json_chunk_type = f.read(4)
                if json_chunk_type != b'JSON':
                    raise ValueError("Missing JSON chunk")

                json_data = f.read(json_chunk_length)
                print(f"JSON chunk size: {json_chunk_length}")

                # Read binary 
                bin_chunk_length = int.from_bytes(f.read(4), 'little')
                bin_chunk_type = f.read(4)
                if bin_chunk_type != b'BIN\x00':
                    raise ValueError("Missing BIN chunk")

                bin_data = f.read(bin_chunk_length)
                print(f"Binary chunk size: {bin_chunk_length}")

                # Parse JSON 
                gltf_json = json.loads(json_data.decode('utf-8'))
                self.model_data = self._parse_meshes(gltf_json, bin_data)
                print(f"Parsed {len(self.model_data)} meshes")

                # Initialize OpenGL buffers
                self.initialize_buffers()
                print("Buffers initialized successfully")

            self.updateGL()
            return True

        except Exception as e:
            print(f"Error loading VRM: {str(e)}")
            logging.error(f"Failed to load VRM model: {e}")
            raise

    def load_gltf(self, gltf_path):
        """Load a GLTF file with associated binary data"""
        try:
            print(f"Loading GLTF file: {gltf_path}")
            
            # Read GLTF JSON
            with open(gltf_path, 'r') as f:
                gltf_json = json.load(f)
                print("Successfully loaded GLTF JSON")

            # Find and load binary file
            bin_path = None
            if 'buffers' in gltf_json:
                for buffer in gltf_json['buffers']:
                    if 'uri' in buffer:
                        # Handle both relative and absolute paths
                        bin_uri = buffer['uri']
                        if bin_uri.startswith('data:'):
                            print("Embedded binary data not supported yet")
                            continue
                        
                        bin_path = os.path.join(os.path.dirname(gltf_path), bin_uri)
                        if os.path.exists(bin_path):
                            print(f"Found binary file: {bin_path}")
                            break

            if not bin_path:
                raise ValueError("Binary file not found")

            # Load binary data
            with open(bin_path, 'rb') as f:
                bin_data = f.read()
                print(f"Loaded binary data: {len(bin_data)} bytes")

            # Parse meshes
            if 'meshes' not in gltf_json:
                raise ValueError("No meshes found in GLTF file")

            self.model_data = self._parse_meshes(gltf_json, bin_data)
            print(f"Parsed {len(self.model_data)} meshes")

            # Load textures if available (they fucking should be)
            if 'images' in gltf_json:
                self.textures = self._load_textures(gltf_json, bin_data, os.path.dirname(gltf_path))
                print(f"Loaded {len(self.textures)} textures")

            # Initialize OpenGL buffers
            self.initialize_buffers()
            print("Buffers initialized successfully")

            self.updateGL()
            return True

        except Exception as e:
            print(f"Error loading GLTF: {str(e)}")
            logging.error(f"Failed to load GLTF model: {e}")
            raise

    def _parse_meshes(self, gltf_json, bin_data):
        """Updated mesh parsing with debug logging for material assignments"""
        meshes = []
        print("\nParsing materials first...")
        self.materials = self.parse_materials(gltf_json)
        
        print("\nParsing meshes...")
        for mesh_idx, mesh in enumerate(gltf_json['meshes']):
            print(f"\nProcessing mesh {mesh_idx}: {mesh.get('name', 'unnamed')}")
            
            for prim_idx, primitive in enumerate(mesh['primitives']):
                print(f"Processing primitive {prim_idx}")
                mesh_data = {}
                
                # Parse vertex data
                if 'POSITION' in primitive['attributes']:
                    vertices = self._get_buffer_data(gltf_json, bin_data, primitive['attributes']['POSITION'])
                    mesh_data['vertices'] = vertices
                    print(f"Vertex count: {len(vertices)}")

                if 'NORMAL' in primitive['attributes']:
                    normals = self._get_buffer_data(gltf_json, bin_data, primitive['attributes']['NORMAL'])
                    mesh_data['normals'] = normals

                if 'indices' in primitive:
                    indices = self._get_buffer_data(gltf_json, bin_data, primitive['indices'], dtype=np.uint32)
                    mesh_data['indices'] = indices
                    print(f"Index count: {len(indices)}")

                # Store material reference
                material_idx = str(primitive.get('material', 'default'))
                mesh_data['material'] = material_idx
                print(f"Assigned material index: {material_idx}")
                
                # If this primitive has VRM extensions, check for material overrides
                if 'extensions' in primitive and 'VRM' in primitive['extensions']:
                    vrm_data = primitive['extensions']['VRM']
                    if 'materialProperties' in vrm_data:
                        mat_props = vrm_data['materialProperties']
                        if '_Color' in mat_props:
                            print(f"Found VRM color override in primitive")
                            self.materials[material_idx]['baseColor'] = mat_props['_Color']
                
                meshes.append(mesh_data)

        print(f"\nTotal meshes parsed: {len(meshes)}")
        return meshes
    
    def apply_material(self, material_idx):
        """Apply material properties with comprehensive texture and model-specific material support"""
        if material_idx not in self.materials:
            print(f"Material {material_idx} not found, using default")
            material_idx = 'default'
        
        material = self.materials[material_idx]
        print(f"Applying material {material_idx}")
        
        # Reset all texture states first
        for i in range(5): 
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glDisable(gl.GL_TEXTURE_2D)
        
        # Enable/disable lighting based on unlit status
        if material.get('unlit', False):
            gl.glDisable(gl.GL_LIGHTING)
        else:
            gl.glEnable(gl.GL_LIGHTING)
        
        # Handle material colors and properties
        base_color = material.get('baseColor', [1.0, 1.0, 1.0, 1.0])
        
        # Set material properties
        ambient_intensity = material.get('ambientFactor', 0.2)
        ambient_color = [c * ambient_intensity for c in base_color[:3]] + [base_color[3]]
        diffuse_color = base_color
        
        # Apply core material properties
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, ambient_color)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, diffuse_color)
        
        # Handle base color texture
        if 'baseColorTexture' in material:
            texture_info = material['baseColorTexture']
            if texture_info['index'] in self.textures:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glEnable(gl.GL_TEXTURE_2D)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_info['index']])
                
                # Set up texture environment to properly blend with material color
                gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_COMBINE)
                gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_COMBINE_RGB, gl.GL_MODULATE)
                gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_SOURCE0_RGB, gl.GL_TEXTURE)
                gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_SOURCE1_RGB, gl.GL_PREVIOUS)
                
                # Handle texture transforms if present
                if 'transform' in texture_info:
                    gl.glMatrixMode(gl.GL_TEXTURE)
                    gl.glLoadIdentity()
                    transform = texture_info['transform']
                    gl.glTranslatef(transform.get('offset', [0, 0])[0], transform.get('offset', [0, 0])[1], 0)
                    gl.glRotatef(transform.get('rotation', 0), 0, 0, 1)
                    gl.glScalef(transform.get('scale', [1, 1])[0], transform.get('scale', [1, 1])[1], 1)
                    gl.glMatrixMode(gl.GL_MODELVIEW)
        
        # Handle PBR properties
        metallic = material.get('metallicFactor', 0.0)
        roughness = material.get('roughnessFactor', 1.0)
        
        # Convert PBR parameters to traditional specular lighting
        specular_intensity = (1.0 - roughness) * (metallic + 0.5) 
        specular_color = [specular_intensity] * 3 + [1.0]
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, specular_color)
        gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, (1.0 - roughness) * 128.0)
        
        # Handle normal mapping
        if 'normalTexture' in material:
            texture_info = material['normalTexture']
            if texture_info['index'] in self.textures:
                gl.glActiveTexture(gl.GL_TEXTURE1)
                gl.glEnable(gl.GL_TEXTURE_2D)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_info['index']])
                gl.glEnable(gl.GL_NORMALIZE)
                
                # Handle normal map intensity
                scale = texture_info.get('scale', 1.0)
                gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_COMBINE)
                gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_RGB_SCALE, scale)
        
        # Handle alpha properties
        alpha_mode = material.get('alphaMode', 'OPAQUE')
        if alpha_mode == 'BLEND':
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        elif alpha_mode == 'MASK':
            gl.glEnable(gl.GL_ALPHA_TEST)
            gl.glAlphaFunc(gl.GL_GEQUAL, material.get('alphaCutoff', 0.5))
        else:  # OPAQUE
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_ALPHA_TEST)
        
        # Handle double-sided rendering
        if material.get('doubleSided', False):
            gl.glDisable(gl.GL_CULL_FACE)
        else:
            gl.glEnable(gl.GL_CULL_FACE)
        
        # Reset to default texture unit and matrix mode
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def _load_textures(self, gltf_json, bin_data, base_path):
        """Load textures from glTF file."""
        textures = {}

        if 'images' in gltf_json:
            for idx, image in enumerate(gltf_json['images']):
                if 'uri' in image:
                    image_path = os.path.join(base_path, image['uri'])
                    if os.path.exists(image_path):
                        img = QImage(image_path).convertToFormat(QImage.Format_RGBA8888)
                        textures[idx] = self._upload_texture(img)
                elif 'bufferView' in image:
                    bufview = gltf_json['bufferViews'][image['bufferView']]
                    offset = bufview.get('byteOffset', 0)
                    length = bufview['byteLength']
                    img_data = bin_data[offset:offset + length]
                    img = QImage.fromData(img_data).convertToFormat(QImage.Format_RGBA8888)
                    textures[idx] = self._upload_texture(img)

        return textures

    def _upload_texture(self, img):
        """Upload a QImage as an OpenGL texture."""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        img_data = bytes(img.bits().asarray(img.byteCount()))

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.width(), img.height(),
            0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data
        )

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        return texture_id

    def _get_buffer_data(self, gltf_json, bin_data, accessor_idx, shape=None, dtype=np.float32):
        """Extract buffer data using accessor information"""
        try:
            accessor = gltf_json['accessors'][accessor_idx]
            bufview = gltf_json['bufferViews'][accessor['bufferView']]
            
            # Calculate offsets and stride
            offset = (bufview.get('byteOffset', 0) + accessor.get('byteOffset', 0))
            stride = bufview.get('byteStride', 0)
            count = accessor['count']
            
            # Determine component type and size
            component_types = {
                5120: np.int8, 5121: np.uint8,
                5122: np.int16, 5123: np.uint16,
                5124: np.int32, 5125: np.uint32,
                5126: np.float32
            }
            
            component_type = component_types.get(accessor['componentType'], dtype)
            
            # Determine number of components per element
            type_sizes = {
                'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
                'MAT2': 4, 'MAT3': 9, 'MAT4': 16
            }
            num_components = type_sizes[accessor['type']]
            
            # Extract data
            element_size = num_components * np.dtype(component_type).itemsize
            
            if stride == 0:
                # Some numpy nonsense from claude
                data = np.frombuffer(
                    bin_data[offset:offset + count * element_size],
                    dtype=component_type
                )
            else:
                # Something about strided data
                data = np.empty(count * num_components, dtype=component_type)
                for i in range(count):
                    element_offset = offset + i * stride
                    element_data = np.frombuffer(
                        bin_data[element_offset:element_offset + element_size],
                        dtype=component_type,
                        count=num_components
                    )
                    data[i * num_components:(i + 1) * num_components] = element_data
            
            # Reshape if needed
            if shape:
                data = data.reshape(-1, *shape)
            elif num_components > 1:
                data = data.reshape(-1, num_components)
            
            print(f"Extracted buffer data: {data.shape} elements of type {component_type}")
            return data

        except Exception as e:
            print(f"Error extracting buffer data: {str(e)}")
            raise
    
    def parse_model_hierarchy(self, gltf_json):
        """Parse complete model hierarchy including nodes, skins, and materials"""
        try:
            print("Parsing complete model hierarchy...")
            
            # Parse materials first(makes sense, right guys? guys?)
            self.materials = self.parse_materials(gltf_json)
            
            # Parse nodes and build hierarchy
            nodes = []
            if 'nodes' in gltf_json:
                for idx, node in enumerate(gltf_json['nodes']):
                    node_data = {
                        'name': node.get('name', f'node_{idx}'),
                        'children': node.get('children', []),
                        'mesh': node.get('mesh'),
                        'skin': node.get('skin'),
                        'matrix': node.get('matrix'),
                        'translation': node.get('translation', [0, 0, 0]),
                        'rotation': node.get('rotation', [0, 0, 0, 1]),
                        'scale': node.get('scale', [1, 1, 1])
                    }
                    
                    # Handle mesh instances
                    if node_data['mesh'] is not None:
                        mesh = gltf_json['meshes'][node_data['mesh']]
                        node_data['primitives'] = []
                        
                        for prim in mesh['primitives']:
                            primitive_data = {
                                'attributes': prim['attributes'],
                                'indices': prim.get('indices'),
                                'material': str(prim.get('material', 'default')),
                                'mode': prim.get('mode', 4)  # TRIANGLES!?!??!
                            }
                            
                            # Parse primitive extensions (e.g., for VRM)
                            if 'extensions' in prim:
                                if 'VRM' in prim['extensions']:
                                    vrm_data = prim['extensions']['VRM']
                                    if 'materialProperties' in vrm_data:
                                        mat_props = vrm_data['materialProperties']
                                        primitive_data['vrm_materials'] = mat_props
                            
                            node_data['primitives'].append(primitive_data)
                    
                    # Handle skins
                    if node_data['skin'] is not None:
                        skin = gltf_json['skins'][node_data['skin']]
                        node_data['joints'] = skin.get('joints', [])
                        node_data['inverseBindMatrices'] = skin.get('inverseBindMatrices')
                        node_data['skeleton'] = skin.get('skeleton')
                    
                    nodes.append(node_data)
        
            # Pass, sorry parse skins and joints
            skins = []
            if 'skins' in gltf_json:
                for idx, skin in enumerate(gltf_json['skins']):
                    skin_data = {
                        'name': skin.get('name', f'skin_{idx}'),
                        'joints': skin['joints'],
                        'inverseBindMatrices': self._get_buffer_data(
                            gltf_json,
                            self.bin_data,
                            skin['inverseBindMatrices']
                        ) if 'inverseBindMatrices' in skin else None,
                        'skeleton': skin.get('skeleton')
                    }
                    skins.append(skin_data)
            
            # Build vertex skinning data(peeling the potato)
            if 'meshes' in gltf_json:
                for mesh_idx, mesh in enumerate(gltf_json['meshes']):
                    for prim in mesh['primitives']:
                        if 'attributes' in prim:
                            attrs = prim['attributes']
                            if 'JOINTS_0' in attrs and 'WEIGHTS_0' in attrs:
                                joints = self._get_buffer_data(
                                    gltf_json,
                                    self.bin_data,
                                    attrs['JOINTS_0'],
                                    dtype=np.uint16
                                )
                                weights = self._get_buffer_data(
                                    gltf_json,
                                    self.bin_data,
                                    attrs['WEIGHTS_0']
                                )
                                prim['skinning'] = {
                                    'joints': joints,
                                    'weights': weights
                                }
            
            return {
                'nodes': nodes,
                'skins': skins
            }
        except Exception as e:
            logging.error(f"Failed to parse_model_hierarchy: {e}")
            return 1.0 

    def parse_morphs(self, gltf_json):
        """Parse morph targets if present"""
        morphs = []
        
        if 'meshes' in gltf_json:
            for mesh_idx, mesh in enumerate(gltf_json['meshes']):
                for prim in mesh['primitives']:
                    if 'targets' in prim:
                        for target_idx, target in enumerate(prim['targets']):
                            morph_data = {
                                'mesh': mesh_idx,
                                'primitive': prim,
                                'target_idx': target_idx,
                                'positions': self._get_buffer_data(
                                    gltf_json,
                                    self.bin_data,
                                    target['POSITION']
                                ) if 'POSITION' in target else None,
                                'normals': self._get_buffer_data(
                                    gltf_json,
                                    self.bin_data,
                                    target['NORMAL']
                                ) if 'NORMAL' in target else None,
                                'tangents': self._get_buffer_data(
                                    gltf_json,
                                    self.bin_data,
                                    target['TANGENT']
                                ) if 'TANGENT' in target else None
                            }
                            morphs.append(morph_data)
        
        return morphs
    
    def draw_model(self):
        """Updated draw_model with material support"""
        if not self.model_data or not self.vbos:
            return

        try:
            gl.glPushMatrix()
            gl.glTranslatef(0, 0, -self.camera_distance)
            gl.glTranslatef(0, self.translation_z, 0)
            gl.glRotatef(self.rotation_x, 1, 0, 0)
            gl.glRotatef(self.rotation_y, 0, 1, 0)

            # Apply scale
            max_dim = max(np.max(np.abs(mesh['vertices'])) for mesh in self.model_data if 'vertices' in mesh)
            scale = 2.0 / max(max_dim, 1.0) * self.zoom_factor
            gl.glScalef(scale, scale, scale)

            # Enable material properties
            gl.glEnable(gl.GL_COLOR_MATERIAL)
            gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
            
            # Draw each mesh with its material
            for mesh_idx, mesh_vbos in enumerate(self.vbos):
                # mesh the mesh the mesh???
                if 'material' in self.model_data[mesh_idx]:
                    self.apply_material(self.model_data[mesh_idx]['material'])
                
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

                # Set up vertices
                vertex_vbo = next(vbo for name, vbo in mesh_vbos if name == 'vertices')
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_vbo)
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)

                # Set up normals if available
                normal_vbo = next((vbo for name, vbo in mesh_vbos if name == 'normals'), None)
                if normal_vbo:
                    gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, normal_vbo)
                    gl.glNormalPointer(gl.GL_FLOAT, 0, None)

                # Draw with indices if available
                index_vbo = next((vbo for name, vbo in mesh_vbos if name == 'indices'), None)
                if index_vbo:
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index_vbo)
                    num_indices = len(self.model_data[mesh_idx]['indices'])
                    gl.glDrawElements(gl.GL_TRIANGLES, num_indices, gl.GL_UNSIGNED_INT, None)
                else:
                    num_vertices = len(self.model_data[mesh_idx]['vertices'])
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, num_vertices)

                if normal_vbo:
                    gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            gl.glPopMatrix()

        except Exception as e:
            print(f"Error drawing model: {str(e)}")
            raise

    def update_model_matrices(self):
        """Update transformation matrices for all nodes"""
        for node in self.model_hierarchy['nodes']:
            if 'matrix' in node:
                # Use provided matrix
                node['worldMatrix'] = np.array(node['matrix']).reshape(4, 4)
            else:
                # Build matrix from TRS
                translation = np.array(node['translation'])
                rotation = np.array(node['rotation'])
                scale = np.array(node['scale'])
                
                # Convert quaternion to matrix
                qx, qy, qz, qw = rotation
                rot_matrix = np.array([
                    [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, 0],
                    [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw, 0],
                    [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy, 0],
                    [0, 0, 0, 1]
                ])
                
                # Build transformation matrix
                transform = np.eye(4)
                transform[:3, 3] = translation
                transform = np.dot(transform, rot_matrix)
                transform[:3, :3] *= scale[:, None]
                
                node['worldMatrix'] = transform
    
    
    def draw_node(self, node_idx, parent_matrix=None):
        """Recursively draw nodes and their children with proper transformations"""
        node = self.model_hierarchy['nodes'][node_idx]
        
        gl.glPushMatrix()
        
        # Apply node's transformation
        if parent_matrix is not None:
            gl.glMultMatrixf(parent_matrix.T)
        
        if 'worldMatrix' in node:
            gl.glMultMatrixf(node['worldMatrix'].T)
        
        # Draw mesh if present
        if 'primitives' in node:
            for prim in node['primitives']:
                # Apply material
                self.apply_material(prim['material'])
                
                # Set up vertex attributes
                if 'vertices' in prim:
                    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                    gl.glVertexPointer(3, gl.GL_FLOAT, 0, prim['vertices'])
                
                if 'normals' in prim:
                    gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                    gl.glNormalPointer(gl.GL_FLOAT, 0, prim['normals'])
                
                # Apply skinning if present
                if 'skinning' in prim:
                    self.apply_skinning(prim['skinning'], node)
                
                # Draw primitive
                if 'indices' in prim:
                    gl.glDrawElements(
                        prim['mode'],
                        len(prim['indices']),
                        gl.GL_UNSIGNED_INT,
                        prim['indices']
                    )
                else:
                    gl.glDrawArrays(prim['mode'], 0, len(prim['vertices']) // 3)
                
                # Clean up state
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        
        # Recursively draw the things I'll never have
        for child_idx in node.get('children', []):
            self.draw_node(child_idx, node.get('worldMatrix', np.eye(4)))
        
        gl.glPopMatrix()

    def apply_skinning(self, skinning_data, node):
        """Apply vertex skinning for animated meshes"""
        if 'joints' not in node or 'inverseBindMatrices' not in node:
            return
        
        joints = skinning_data['joints']
        weights = skinning_data['weights']
        
        # Enable matrix palette skinning
        gl.glEnable(gl.GL_MATRIX_PALETTE_ARB)
        
        # Set up joint matrices
        for i, joint_idx in enumerate(node['joints']):
            joint_node = self.model_hierarchy['nodes'][joint_idx]
            inverse_bind = node['inverseBindMatrices'][i]
            joint_matrix = np.dot(joint_node['worldMatrix'], inverse_bind)
            
            gl.glCurrentPaletteMatrixARB(i)
            gl.glMatrixMode(gl.GL_MATRIX0_ARB + i)
            gl.glLoadMatrixf(joint_matrix.T)
        
        # Set up vertex attributes for skinning
        gl.glVertexAttribPointer(1, 4, gl.GL_UNSIGNED_SHORT, False, 0, joints)
        gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, False, 0, weights)
        
        gl.glEnableVertexAttribArray(1)
        gl.glEnableVertexAttribArray(2)
        
        
    def calculate_max_dimension(self):
        """Calculate maximum dimension of the model"""
        try:
            max_dim = 0
            for mesh in self.model_data:
                if 'vertices' in mesh:
                    vertices = np.array(mesh['vertices'])
                    max_dim = max(max_dim, np.max(np.abs(vertices)))
            return max(max_dim, 1.0)  # Fucking guy, you cost me hours I'll never get back(get better at loggin hannah). /0 is banned
        except Exception as e:
            logging.error(f"Failed to calculate model dimensions: {e}")
            return 1.0

    def closeEvent(self, event):
        """Clean up resources when closing"""
        try:
            self.cleanup_buffers()
            super().closeEvent(event)
        except Exception as e:
            logging.error(f"Close event failed: {e}")

    def update_rotation(self):
        self.rotation += 1.0
        self.updateGL()

    def initializeGL(self):
        try:
            gl.glClearColor(0.2, 0.2, 0.2, 1.0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_COLOR_MATERIAL)
            
            # Set up light position and properties
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
            
            logging.info("OpenGL initialization successful")
        except Exception as e:
            logging.error(f"OpenGL initialization failed: {e}")
            raise

    def resizeGL(self, width, height):
        try:
            # one does not simply divide by zero
            if height == 0:
                height = 1
                
            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(GL_PROJECTION)
            gl.glLoadIdentity()
            
            aspect = width / float(height)
            # Adjusted frustum to see more of the scene
            gl.glFrustum(-aspect, aspect, -1.0, 1.0, 1.0, 1000.0)
            
            logging.info(f"GL resize successful: {width}x{height}")
        except Exception as e:
            logging.error(f"GL resize failed: {e}")
            raise

    def paintGL(self):
        try:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            
            # Use gluLookAt if available
            if gluLookAt:
                gluLookAt(
                    self.camera_distance * np.sin(np.radians(self.camera_rotation)),
                    self.camera_height,
                    self.camera_distance * np.cos(np.radians(self.camera_rotation)),
                    0, 0, 0,
                    0, 1, 0
                )
            else:
                logging.error("gluLookAt is not available.")
                gl.glTranslatef(0, 0, -self.camera_distance) 
            
            self.draw_axes()
            # Changed condition here
            if self.model_data:
                self.draw_model()
        except Exception as e:
            logging.error(f"GL paint failed: {e}")
            raise



    def draw_axes(self):
        gl.glBegin(gl.GL_LINES)
        # X axis (red)
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(1.0, 0.0, 0.0)
        # Y axis (green)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 1.0, 0.0)
        # Z axis (blue)
        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 1.0)
        gl.glEnd()


    def close_viewer(self):
        try:
            if self.vrm_data:
                self.vrm_data = None
                self.update()
            logging.info("Viewer resources cleaned up")
        except Exception as e:
            logging.error(f"Failed to close viewer: {e}")
