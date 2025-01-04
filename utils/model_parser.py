# utils/model_parser.py

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class ModelFormat(Enum):
    """Supported 3D model formats"""
    VRM = "vrm"
    GLTF = "gltf"
    GLB = "glb"

@dataclass
class MaterialData:
    """Material data structure"""
    name: str
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 0.0
    roughness_factor: float = 1.0
    emissive_factor: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    textures: Dict[str, Dict] = None
    vrm_properties: Dict = None

@dataclass
class MeshData:
    """Mesh data structure"""
    name: str
    vertices: np.ndarray
    indices: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    material_id: Optional[str] = None
    primitives: List[Dict] = None

class ModelParser:
    """Enhanced model parser with support for VRM and GLTF formats"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse VRM/GLTF/GLB file and return structured data"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")

            model_format = self.detect_format(file_path)
            self.logger.info(f"Detected format: {model_format.value}")

            if model_format in [ModelFormat.VRM, ModelFormat.GLB]:
                return self.parse_binary_format(file_path)
            elif model_format == ModelFormat.GLTF:
                return self.parse_gltf(file_path)
            else:
                raise ValueError(f"Unsupported format: {model_format}")

        except Exception as e:
            self.logger.error(f"Error parsing file: {e}")
            raise

    def detect_format(self, file_path: Path) -> ModelFormat:
        """Detect file format based on extension and content"""
        ext = file_path.suffix.lower()
        
        if ext == '.vrm':
            return ModelFormat.VRM
        elif ext == '.gltf':
            return ModelFormat.GLTF
        elif ext == '.glb':
            return ModelFormat.GLB
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def parse_binary_format(self, file_path: Path) -> Dict[str, Any]:
        """Parse VRM/GLB binary format"""
        try:
            with open(file_path, 'rb') as f:
                # Validate magic number
                magic = f.read(4)
                if magic != b'glTF':
                    raise ValueError("Invalid binary format: missing glTF magic number")

                # Read header
                version = int.from_bytes(f.read(4), 'little')
                length = int.from_bytes(f.read(4), 'little')
                
                self.logger.debug(f"Binary format version: {version}, length: {length}")

                # Parse chunks
                json_data = self._parse_json_chunk(f)
                bin_data = self._parse_bin_chunk(f)

                # Process data
                materials = self.parse_materials(json_data)
                meshes = self.parse_meshes(json_data, bin_data)
                model_info = self.parse_model_info(json_data)
                
                return {
                    'format': 'binary',
                    'version': version,
                    'materials': materials,
                    'meshes': meshes,
                    'model_info': model_info,
                    'raw_json': json_data
                }

        except Exception as e:
            self.logger.error(f"Error parsing binary format: {e}")
            raise

    def _parse_json_chunk(self, file_handle) -> Dict:
        """Parse JSON chunk from binary file"""
        chunk_length = int.from_bytes(file_handle.read(4), 'little')
        chunk_type = file_handle.read(4)
        
        if chunk_type != b'JSON':
            raise ValueError("Invalid chunk structure: missing JSON chunk")
        
        json_data = file_handle.read(chunk_length)
        return json.loads(json_data)

    def _parse_bin_chunk(self, file_handle) -> bytes:
        """Parse BIN chunk from binary file"""
        chunk_length = int.from_bytes(file_handle.read(4), 'little')
        chunk_type = file_handle.read(4)
        
        if chunk_type != b'BIN\x00':
            raise ValueError("Invalid chunk structure: missing BIN chunk")
        
        return file_handle.read(chunk_length)

    def parse_gltf(self, file_path: Path) -> Dict[str, Any]:
        """Parse GLTF JSON format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Find and load associated binary file
            bin_data = None
            if 'buffers' in json_data:
                bin_data = self._load_binary_data(file_path, json_data['buffers'])

            materials = self.parse_materials(json_data)
            meshes = self.parse_meshes(json_data, bin_data)
            model_info = self.parse_model_info(json_data)

            return {
                'format': 'gltf',
                'materials': materials,
                'meshes': meshes,
                'model_info': model_info,
                'raw_json': json_data
            }

        except Exception as e:
            self.logger.error(f"Error parsing GLTF: {e}")
            raise

    def _load_binary_data(self, gltf_path: Path, buffers: List[Dict]) -> Optional[bytes]:
        """Load binary data associated with GLTF file"""
        for buffer in buffers:
            if 'uri' in buffer:
                if buffer['uri'].startswith('data:'):
                    self.logger.warning("Embedded binary data not supported yet")
                    continue

                bin_path = gltf_path.parent / buffer['uri']
                if bin_path.exists():
                    with open(bin_path, 'rb') as f:
                        return f.read()

        return None

    def parse_materials(self, json_data: Dict) -> Dict[str, MaterialData]:
        """Parse material data with enhanced texture and shader support"""
        materials = {}
        
        if 'materials' not in json_data:
            self.logger.info("No materials found, using default")
            materials['default'] = MaterialData(
                name="default",
                base_color=(0.8, 0.8, 0.8, 1.0)
            )
            return materials

        for idx, material in enumerate(json_data['materials']):
            material_data = self._parse_single_material(material, idx)
            materials[str(idx)] = material_data

        return materials

    def _parse_single_material(self, material: Dict, idx: int) -> MaterialData:
        """Parse a single material entry"""
        name = material.get('name', f'material_{idx}')
        mat_data = MaterialData(name=name)

        # Parse VRM extensions
        if 'extensions' in material and 'VRM' in material['extensions']:
            self._parse_vrm_material_extensions(material['extensions']['VRM'], mat_data)

        # Parse standard PBR properties
        if 'pbrMetallicRoughness' in material:
            self._parse_pbr_properties(material['pbrMetallicRoughness'], mat_data)

        # Parse additional properties
        mat_data.alpha_mode = material.get('alphaMode', 'OPAQUE')
        mat_data.double_sided = material.get('doubleSided', False)
        mat_data.emissive_factor = material.get('emissiveFactor', (0, 0, 0))

        if mat_data.alpha_mode == 'MASK':
            mat_data.alpha_cutoff = material.get('alphaCutoff', 0.5)

        return mat_data

    def _parse_vrm_material_extensions(self, vrm_data: Dict, mat_data: MaterialData):
        """Parse VRM-specific material extensions"""
        mat_data.vrm_properties = {}

        if 'vectorProperties' in vrm_data:
            vec_props = vrm_data['vectorProperties']
            if '_Color' in vec_props:
                mat_data.base_color = tuple(vec_props['_Color'])
            mat_data.vrm_properties.update(vec_props)

        if 'textureProperties' in vrm_data:
            if not mat_data.textures:
                mat_data.textures = {}
            for name, idx in vrm_data['textureProperties'].items():
                mat_data.textures[f'vrm_{name}'] = {'index': idx}

    def _parse_pbr_properties(self, pbr: Dict, mat_data: MaterialData):
        """Parse standard PBR material properties"""
        mat_data.base_color = tuple(pbr.get('baseColorFactor', (1.0, 1.0, 1.0, 1.0)))
        mat_data.metallic_factor = pbr.get('metallicFactor', 1.0)
        mat_data.roughness_factor = pbr.get('roughnessFactor', 1.0)

        if 'baseColorTexture' in pbr:
            if not mat_data.textures:
                mat_data.textures = {}
            mat_data.textures['baseColor'] = self._parse_texture_info(pbr['baseColorTexture'])

    def _parse_texture_info(self, tex_info: Dict) -> Dict:
        """Parse texture information including transforms"""
        result = {
            'index': tex_info['index'],
            'texCoord': tex_info.get('texCoord', 0)
        }

        if 'extensions' in tex_info and 'KHR_texture_transform' in tex_info['extensions']:
            transform = tex_info['extensions']['KHR_texture_transform']
            result['transform'] = {
                'offset': transform.get('offset', [0, 0]),
                'rotation': transform.get('rotation', 0),
                'scale': transform.get('scale', [1, 1])
            }

        return result

    def parse_meshes(self, json_data: Dict, bin_data: Optional[bytes]) -> List[MeshData]:
        """Parse mesh data with support for primitives and attributes"""
        meshes = []

        if 'meshes' not in json_data:
            self.logger.warning("No meshes found in file")
            return meshes

        for mesh_idx, mesh in enumerate(json_data['meshes']):
            mesh_name = mesh.get('name', f'mesh_{mesh_idx}')
            primitives_data = []

            for prim in mesh['primitives']:
                primitive = self._parse_primitive(prim, json_data, bin_data)
                primitives_data.append(primitive)

            # Create MeshData object
            mesh_data = MeshData(
                name=mesh_name,
                vertices=np.concatenate([p['vertices'] for p in primitives_data if 'vertices' in p]),
                indices=np.concatenate([p['indices'] for p in primitives_data if 'indices' in p]) 
                    if all('indices' in p for p in primitives_data) else None,
                normals=np.concatenate([p['normals'] for p in primitives_data if 'normals' in p])
                    if all('normals' in p for p in primitives_data) else None,
                material_id=str(primitives_data[0].get('material', 'default')),
                primitives=primitives_data
            )
            meshes.append(mesh_data)

        return meshes

    def _parse_primitive(self, primitive: Dict, json_data: Dict, bin_data: Optional[bytes]) -> Dict:
        """Parse a single mesh primitive"""
        result = {}

        # Parse attributes
        if 'attributes' in primitive:
            attrs = primitive['attributes']
            
            if 'POSITION' in attrs:
                result['vertices'] = self._get_accessor_data(
                    json_data, bin_data, attrs['POSITION']
                )
            
            if 'NORMAL' in attrs:
                result['normals'] = self._get_accessor_data(
                    json_data, bin_data, attrs['NORMAL']
                )
            
            if 'TEXCOORD_0' in attrs:
                result['uvs'] = self._get_accessor_data(
                    json_data, bin_data, attrs['TEXCOORD_0']
                )

        # Parse indices
        if 'indices' in primitive:
            result['indices'] = self._get_accessor_data(
                json_data, bin_data, primitive['indices'], dtype=np.uint32
            )

        # Store material reference
        if 'material' in primitive:
            result['material'] = primitive['material']

        return result

    def _get_accessor_data(
        self, 
        json_data: Dict, 
        bin_data: bytes, 
        accessor_idx: int, 
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """Extract data from buffer using accessor information"""
        accessor = json_data['accessors'][accessor_idx]
        bufview = json_data['bufferViews'][accessor['bufferView']]
        
        # Calculate offsets and sizes
        offset = (bufview.get('byteOffset', 0) + accessor.get('byteOffset', 0))
        stride = bufview.get('byteStride', 0)
        count = accessor['count']
        
        # Determine data type and size
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
            # Tightly packed data
            data = np.frombuffer(
                bin_data[offset:offset + count * element_size],
                dtype=component_type
            )
        else:
            # Handle strided data
            data = np.empty(count * num_components, dtype=component_type)
            for i in range(count):
                element_offset = offset + i * stride
                element_data = np.frombuffer(
                    bin_data[element_offset:element_offset + element_size],
                    dtype=component_type,
                    count=num_components
                )
                data[i * num_components:(i + 1) * num_components] = element_data
        
        # Reshape array if needed
        if num_components > 1:
            data = data.reshape(-1, num_components)
        
        # Apply normalization if specified
        if 'normalized' in accessor and accessor['normalized']:
            data = self._normalize_data(data, component_type)
        
        return data