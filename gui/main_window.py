# gui/main_window.py

import os
import json
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt

from .vrm_viewer import VRMViewer
from .webcam_tracker import WebcamTracker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTuber Motion Tracker & VRM Handler")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.init_ui()
        self.vrm_path = None
        self.webcam_tracker = WebcamTracker(self)

    def init_ui(self):
        main_widget = QWidget()
        self.layout = QVBoxLayout(main_widget)
        
        self.label = QLabel("Upload a VTuber model or process a VRM file")
        self.layout.addWidget(self.label)
        
        self.create_buttons()
        
        self.vrm_viewer = VRMViewer()
        self.vrm_viewer.setMinimumSize(640, 480)
        self.layout.addWidget(self.vrm_viewer)
        
        self.webcam_label = QLabel()
        self.webcam_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.webcam_label)
        
        self.setCentralWidget(main_widget)

    def create_buttons(self):
        buttons_data = [
            ("Upload VRM/GLTF Model", self.upload_model),
            ("Validate Model", self.validate_model),
            ("Preview Model", self.preview_model),
            ("Convert VRM to GLTF", self.convert_vrm_to_gltf),
            ("Start Webcam Tracking", self.toggle_webcam_tracking)
        ]
        
        for text, slot in buttons_data:
            button = QPushButton(text)
            button.clicked.connect(slot)
            self.layout.addWidget(button)
            if text != "Upload VRM/GLTF Model" and text != "Start Webcam Tracking":
                button.setEnabled(False)
            setattr(self, text.lower().replace(" ", "_").replace("/", "_") + "_button", button)

    def upload_model(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Upload Model", 
                "", 
                "3D Models (*.vrm *.gltf);;VRM Files (*.vrm);;GLTF Files (*.gltf);;All Files (*.*)"
            )
            if file_path:
                self.model_path = file_path
                self.model_type = os.path.splitext(file_path)[1].lower()
                self.label.setText(f"Loaded model: {os.path.basename(self.model_path)}")
                self.validate_model_button.setEnabled(True)
                self.preview_model_button.setEnabled(True)
                self.convert_vrm_to_gltf_button.setEnabled(self.model_type == '.vrm')
                logging.info(f"Successfully loaded {self.model_type} file: {file_path}")
        except Exception as e:
            error_msg = f"Failed to upload file: {str(e)}"
            self.label.setText(error_msg)
            logging.error(f"Upload error: {e}")
            QMessageBox.critical(self, "Error", error_msg)

    def validate_model(self):
        if not hasattr(self, 'model_path'):
            self.label.setText("No model uploaded!")
            return

        try:
            # Read file and determine type
            with open(self.model_path, 'rb') as f:
                magic = f.read(4)
                f.seek(0)
                
                validation_report = []
                
                # VRM/GLB binary format
                if magic == b'glTF':
                    self._validate_vrm_glb(f, validation_report)
                # GLTF JSON format
                else:
                    self._validate_gltf_json(f, validation_report)
                
                # Display validation results
                report_text = "\n".join(validation_report)
                self.label.setText(report_text)
                logging.info(f"Validation completed successfully for {self.model_path}")
                
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            self.label.setText(error_msg)
            logging.error(f"Validation error: {e}")
            QMessageBox.critical(self, "Error", error_msg)

    def _validate_vrm_glb(self, file_handle, validation_report):
        """Helper method to validate VRM/GLB format"""
        # Read header
        header = file_handle.read(12)
        version = int.from_bytes(header[4:8], 'little')
        total_length = int.from_bytes(header[8:12], 'little')
        
        # Read JSON chunk
        chunk_length = int.from_bytes(file_handle.read(4), 'little')
        chunk_type = file_handle.read(4)
        
        if chunk_type != b'JSON':
            raise ValueError("Invalid GLB structure: Missing JSON chunk")
        
        json_data = file_handle.read(chunk_length)
        metadata = json.loads(json_data)
        
        # Build VRM validation report
        validation_report.extend([
            "VRM File Structure Report:",
            f"File type: Binary VRM/GLB",
            f"GLB Version: {version}",
            f"File size: {total_length} bytes",
            "\nStructure Check:"
        ])
        
        self._check_common_properties(metadata, validation_report)
        self._check_vrm_properties(metadata, validation_report)

    def _validate_gltf_json(self, file_handle, validation_report):
        """Helper method to validate GLTF JSON format"""
        try:
            gltf_data = json.load(file_handle)
            
            validation_report.extend([
                "GLTF File Structure Report:",
                f"File type: GLTF JSON",
                "\nStructure Check:"
            ])
            
            self._check_common_properties(gltf_data, validation_report)
            self._check_binary_references(gltf_data, validation_report)
            
        except json.JSONDecodeError:
            raise ValueError("Invalid GLTF JSON format")

    def _check_common_properties(self, data, validation_report):
        """Check common GLTF properties"""
        for prop in ['asset', 'scenes', 'nodes', 'meshes']:
            status = "✓" if prop in data else "❌"
            validation_report.append(f"{status} {prop} data")

    def _check_vrm_properties(self, metadata, validation_report):
        """Check VRM-specific properties"""
        if 'extensionsUsed' in metadata and 'VRM' in metadata['extensionsUsed']:
            validation_report.append("✓ VRM extension declared")
            
            if 'extensions' in metadata and 'VRM' in metadata['extensions']:
                vrm_data = metadata['extensions']['VRM']
                validation_report.append("\nVRM Metadata:")
                for key in ['title', 'version', 'author', 'contactInformation', 'reference']:
                    value = vrm_data.get(key, 'Not found')
                    validation_report.append(f"{key}: {value}")
            else:
                validation_report.append("❌ No VRM metadata found")

    def _check_binary_references(self, gltf_data, validation_report):
        """Check binary file references in GLTF"""
        if 'asset' in gltf_data and 'version' in gltf_data['asset']:
            validation_report.append(f"GLTF Version: {gltf_data['asset']['version']}")
        
        if 'buffers' in gltf_data:
            validation_report.append("\nBinary Files:")
            for idx, buffer in enumerate(gltf_data['buffers']):
                if 'uri' in buffer:
                    bin_path = os.path.join(os.path.dirname(self.model_path), buffer['uri'])
                    if os.path.exists(bin_path):
                        bin_size = os.path.getsize(bin_path)
                        validation_report.append(f"✓ Binary file {idx+1} found: {buffer['uri']} ({bin_size} bytes)")
                    else:
                        validation_report.append(f"❌ Binary file {idx+1} missing: {buffer['uri']}")
                else:
                    validation_report.append(f"❌ Buffer {idx+1} has no URI specified")
        else:
            validation_report.append("\n❌ No buffer data found")

    def preview_model(self):
        if not hasattr(self, 'model_path'):
            self.label.setText("No model uploaded!")
            return

        try:
            file_ext = os.path.splitext(self.model_path)[1].lower()
            
            if file_ext == '.vrm':
                self.vrm_viewer.load_vrm(self.model_path)
                model_type = "VRM"
            elif file_ext == '.gltf':
                self.vrm_viewer.load_gltf(self.model_path)
                model_type = "GLTF"
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            self.label.setText(f"Previewing {model_type} model... Use mouse to rotate view\n"
                            "Left click + drag: Move up/down\n"
                            "Right click + drag: Rotate\n"
                            "Mouse wheel: Zoom in/out")
            
            logging.info(f"Preview started for {model_type} model: {self.model_path}")
        
        except Exception as e:
            error_msg = f"Failed to preview model: {str(e)}"
            self.label.setText(error_msg)
            logging.error(f"Preview error: {e}")
            QMessageBox.critical(self, "Error", error_msg)

    def convert_vrm_to_gltf(self):
        if not self.vrm_path:
            self.label.setText("No VRM model uploaded!")
            return

        try:
            with open(self.vrm_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'glTF':
                    raise ValueError("Not a valid GLB/VRM file")
                
                # Process file chunks
                file_data = self._read_vrm_file_chunks(f)
                
                # Convert and save files
                success_msg = self._save_converted_files(file_data)
                
                # Display success message
                self.label.setText(success_msg)
                logging.info(f"Successfully converted {self.vrm_path} to GLTF")
                QMessageBox.information(self, "Success", success_msg)

        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            self.label.setText(error_msg)
            logging.error(f"Conversion error: {e}")
            QMessageBox.critical(self, "Error", error_msg)

    def _read_vrm_file_chunks(self, file_handle):
        """Read and validate VRM file chunks"""
        # Read header
        file_handle.seek(0)
        header = file_handle.read(12)
        
        # Read JSON chunk
        chunk_length = int.from_bytes(file_handle.read(4), 'little')
        chunk_type = file_handle.read(4)
        
        if chunk_type != b'JSON':
            raise ValueError("Invalid GLB structure: Missing JSON chunk")
        
        json_data = file_handle.read(chunk_length)
        
        # Read BIN chunk
        bin_length = int.from_bytes(file_handle.read(4), 'little')
        bin_type = file_handle.read(4)
        
        if bin_type != b'BIN\x00':
            raise ValueError("Invalid GLB structure: Missing BIN chunk")
        
        bin_data = file_handle.read(bin_length)
        
        return {
            'json_data': json_data,
            'bin_data': bin_data
        }

    def _save_converted_files(self, file_data):
        """Save converted GLTF and binary files"""
        # Create output directory
        converted_models_dir = os.path.join(os.getcwd(), 'convertedModels')
        if not os.path.exists(converted_models_dir):
            os.makedirs(converted_models_dir)

        # Setup file paths
        base_name = os.path.splitext(os.path.basename(self.vrm_path))[0]
        gltf_path = os.path.join(converted_models_dir, f"{base_name}.gltf")
        bin_path = os.path.join(converted_models_dir, f"{base_name}.bin")

        # Process and save GLTF JSON
        gltf_json = json.loads(file_data['json_data'])
        if 'buffers' in gltf_json and len(gltf_json['buffers']) > 0:
            gltf_json['buffers'][0]['uri'] = os.path.basename(bin_path)
        
        with open(gltf_path, 'w', encoding='utf-8') as gltf_file:
            json.dump(gltf_json, gltf_file, indent=2)
        
        # Save binary data
        with open(bin_path, 'wb') as bin_file:
            bin_file.write(file_data['bin_data'])
        
        return f"Conversion successful:\nGLTF: {gltf_path}\nBIN: {bin_path}"

    def toggle_webcam_tracking(self):
        try:
            if not self.webcam_tracker.is_tracking:
                self.webcam_tracker.start()
                self.track_button.setText("Stop Webcam Tracking")
                self.label.setText("Webcam tracking active. Face detection enabled.")
            else:
                self.webcam_tracker.stop()
                self.track_button.setText("Start Webcam Tracking")
                self.label.setText("Webcam tracking stopped.")
                self.webcam_label.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Webcam error: {str(e)}")
            logging.error(f"Webcam toggle error: {e}")

    def closeEvent(self, event):
        try:
            if self.webcam_tracker:
                self.webcam_tracker.stop()
            if self.vrm_viewer:
                self.vrm_viewer.close_viewer()
            logging.info("Application closed properly")
            event.accept
        except Exception as e:
            logging.error(f"Error during application closure: {e}")
            event.accept()