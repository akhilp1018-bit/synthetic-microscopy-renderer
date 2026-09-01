"""
Interactive 3D GUI for the Synthetic Microscopy Renderer.

This GUI provides a visual interface for preparing and launching synthetic
microscopy renders. It can load a single mesh or multiple component meshes,
display and orient them in 3D, select a fixed rendering subvolume, export the
renderer configuration, and launch or queue renders.

Mesh coordinates loaded through the GUI are expected to be in nanometres.
Multiple meshes must share the same XYZ coordinate system.

Installation:
    pip install -r requirements.txt

Run from the repository root:
    python gui/app.py

The GUI launches:
    python scripts/render.py --config gui/configs/<exported_config.yaml>

Main workflow:
    1. Load a single mesh or multiple-mesh folder.
    2. Rotate/orient the mesh in the 3D viewer.
    3. Set output shape and spatial sampling.
    4. Add and position the rendering ROI.
    5. Export the YAML configuration.
    6. Run immediately or add the render to the queue.

See gui/README.md for detailed usage and configuration information.
"""

import copy
import os
import sys
from pathlib import Path

import pyvista as pv
import yaml
from PySide6.QtCore import QProcess
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor


MESH_UNITS_PER_UM = 1000.0

DEFAULT_CONFIG = {
    "input": {
        "mode": "single_mesh",
        "mesh_path": "",
        "scale_to_nm": 1.0,
        "recenter": False,
    },
    "output": {
        "output_dir": "outputs/gui_results",
        "output_name": "gui_render",
    },
    "grid": {
        "xy_um_per_px": 0.094,
        "z_step_um": 0.5,
        "use_roi": False,
        "margin": 0.05,
        "shape_mode": "fixed",
        "output_shape_zyx": [64, 128, 128],
    },
    "renderer": {
        "method": "voxel_grid",
        "labeling_mode": "membrane",
        "spacing_nm": 200,
        "batch_faces": 2048,
        "pseudofill_sigma_zyx": [2.0, 2.5, 2.5],
        "density_smooth_sigma_zyx": [0.5, 0.8, 0.8],
        "density_normalize_sum": True,
    },
    "splatting": {
        "spacing_nm": 200,
        "sigma_zyx": [0.5, 0.8, 0.8],
        "apply_psf": True,
        "seed": 0,
        "points_per_batch": 50000,
    },
    "psf": {
        "mode": "gaussian_2p",
        "path": "psfs/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif",
        "two_photon_like": True,
        "shape_zyx": [13, 65, 65],
        "lambda_nm": 488.0,
        "na": 1.0,
        "refractive_index": 1.33,
        "sigma_scale_xy": 1.0,
        "sigma_scale_z": 1.0,
    },
    "noise": {
        "enabled": False,
        "peak_photons": 500.0,
        "read_noise_std": 5.0,
        "seed": 0,
        "gaussian_chunk_slices": 16,
    },
    "masks": {
        "object_rel_threshold": 0.1,
        "spine_rel_threshold": 0.1,
        "dendrite_rel_threshold": 0.1,
        "save_individual_spine_masks": False,
        "save_clean_component_images": False,
    },
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Mesh Renderer")
        self.resize(1400, 900)

        self.mesh = None
        self.mesh_path = None
        self.input_mode = None
        self.labelled_dir = None
        self.dendrite_path = None
        self.spine_paths = []
        self.roi_actor = None
        self.roi_center_nm = None
        self.roi_bounds = None
        self.exported_config_path = None

        self.render_queue = []
        self.current_queue_job = None
        self.queue_running = False

        self._build_ui()
        self._create_render_process()
        self._connect_setting_signals()

        self.update_volume_label()
        self.update_queue_display()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_widget = QWidget()
        control_widget.setMaximumWidth(390)
        self.left_layout = QVBoxLayout(control_widget)

        self._build_mesh_controls()
        self._build_shape_controls()
        self._build_sampling_controls()
        self._build_roi_controls()
        self._build_render_controls()
        self._build_queue_controls()
        self._build_log_controls()

        self.left_layout.addStretch()
        main_layout.addWidget(control_widget)

        self.plotter = QtInteractor(central_widget)
        main_layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("white")
        self.plotter.add_axes()

    def _section_label(self, text):
        self.left_layout.addWidget(QLabel(f"\n{text}"))

    def _build_mesh_controls(self):
        self._section_label("Mesh Input")

        self.load_button = QPushButton("Load Single Mesh (nm)")
        self.load_button.clicked.connect(self.load_mesh)
        self.left_layout.addWidget(self.load_button)

        self.load_labelled_button = QPushButton(
            "Load Multiple Meshes Folder (nm)"
        )
        self.load_labelled_button.clicked.connect(
            self.load_labelled_components
        )
        self.left_layout.addWidget(self.load_labelled_button)

        self.mesh_label = QLabel("No mesh loaded")
        self.mesh_label.setWordWrap(True)
        self.left_layout.addWidget(self.mesh_label)

    def _build_shape_controls(self):
        self._section_label("Output Shape [Z, Y, X]")

        layout = QHBoxLayout()
        self.z_spin = self._make_spin_box(64)
        self.y_spin = self._make_spin_box(128)
        self.x_spin = self._make_spin_box(128)

        for label, widget in (
            ("Z", self.z_spin),
            ("Y", self.y_spin),
            ("X", self.x_spin),
        ):
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)

        self.left_layout.addLayout(layout)

    def _build_sampling_controls(self):
        self._section_label("Sampling")

        self.xy_spin = self._make_double_spin_box(
            value=0.094,
            step=0.001,
        )
        self.z_sampling_spin = self._make_double_spin_box(
            value=0.5,
            step=0.1,
        )

        self.left_layout.addLayout(
            self._labeled_row("XY µm/pixel", self.xy_spin)
        )
        self.left_layout.addLayout(
            self._labeled_row("Z µm/slice", self.z_sampling_spin)
        )

        self.volume_label = QLabel()
        self.volume_label.setWordWrap(True)
        self.left_layout.addWidget(self.volume_label)

    def _build_roi_controls(self):
        self._section_label("Rendering ROI")

        layout = QHBoxLayout()

        self.add_roi_button = QPushButton("Add Rendering ROI")
        self.add_roi_button.clicked.connect(self.add_roi)

        self.move_roi_button = QPushButton("Move ROI")
        self.move_roi_button.clicked.connect(self.enable_roi_movement)

        layout.addWidget(self.add_roi_button)
        layout.addWidget(self.move_roi_button)
        self.left_layout.addLayout(layout)

        self.roi_label = QLabel("No ROI selected")
        self.roi_label.setWordWrap(True)
        self.left_layout.addWidget(self.roi_label)

    def _build_render_controls(self):
        self._section_label("Render Configuration")

        layout = QHBoxLayout()

        self.export_button = QPushButton("Export Config")
        self.export_button.clicked.connect(self.export_config)

        self.run_button = QPushButton("Run Render")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_render)

        layout.addWidget(self.export_button)
        layout.addWidget(self.run_button)
        self.left_layout.addLayout(layout)

    def _build_queue_controls(self):
        layout = QHBoxLayout()

        self.add_queue_button = QPushButton("Add to Queue")
        self.add_queue_button.setEnabled(False)
        self.add_queue_button.clicked.connect(self.add_to_queue)

        self.run_queue_button = QPushButton("Run Queue")
        self.run_queue_button.setEnabled(False)
        self.run_queue_button.clicked.connect(self.run_queue)

        layout.addWidget(self.add_queue_button)
        layout.addWidget(self.run_queue_button)
        self.left_layout.addLayout(layout)

        self._section_label("Render Queue")

        self.queue_display = QTextEdit()
        self.queue_display.setReadOnly(True)
        self.queue_display.setPlaceholderText("No renders queued.")
        self.queue_display.setMaximumHeight(140)
        self.left_layout.addWidget(self.queue_display)

    def _build_log_controls(self):
        self._section_label("Renderer Log")

        self.render_log = QTextEdit()
        self.render_log.setReadOnly(True)
        self.render_log.setPlaceholderText(
            "Renderer output will appear here..."
        )
        self.render_log.setMinimumHeight(170)
        self.left_layout.addWidget(self.render_log)

    @staticmethod
    def _make_spin_box(value):
        box = QSpinBox()
        box.setRange(1, 4096)
        box.setValue(value)
        return box

    @staticmethod
    def _make_double_spin_box(value, step):
        box = QDoubleSpinBox()
        box.setDecimals(4)
        box.setRange(0.001, 100.0)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    @staticmethod
    def _labeled_row(text, widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(text))
        layout.addWidget(widget)
        return layout

    # ------------------------------------------------------------------
    # Process setup
    # ------------------------------------------------------------------

    def _create_render_process(self):
        self.render_process = QProcess(self)
        self.render_process.readyReadStandardOutput.connect(
            self.read_render_output
        )
        self.render_process.readyReadStandardError.connect(
            self.read_render_error
        )
        self.render_process.finished.connect(self.render_finished)

    def _connect_setting_signals(self):
        for widget in (
            self.z_spin,
            self.y_spin,
            self.x_spin,
            self.xy_spin,
            self.z_sampling_spin,
        ):
            widget.valueChanged.connect(self.settings_changed)

    # ------------------------------------------------------------------
    # Repository and path helpers
    # ------------------------------------------------------------------

    def get_repo_root(self):
        return str(Path(__file__).resolve().parent.parent)

    def get_render_script(self):
        return os.path.join(
            self.get_repo_root(),
            "scripts",
            "render.py",
        )

    def make_repo_relative_path(self, file_path):
        try:
            path = os.path.relpath(
                os.path.abspath(file_path),
                self.get_repo_root(),
            )
        except ValueError:
            path = os.path.abspath(file_path)

        return path.replace("\\", "/")

    # ------------------------------------------------------------------
    # Mesh loading
    # ------------------------------------------------------------------

    def _reset_loaded_input(self):
        """Clear input-specific state before loading new geometry."""
        self.mesh = None
        self.mesh_path = None
        self.input_mode = None
        self.labelled_dir = None
        self.dendrite_path = None
        self.spine_paths = []

        try:
            self.plotter.disable_picking()
        except Exception:
            pass

        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.add_axes()

        self.roi_actor = None
        self.roi_center_nm = None
        self.roi_bounds = None
        self.roi_label.setText("No ROI selected")
        self.invalidate_exported_config()

    def load_mesh(self):
        """Load one complete mesh for single-mesh rendering."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Single Mesh",
            self.get_repo_root(),
            "Mesh Files (*.ply *.stl *.obj *.vtk *.vtp);;All Files (*)",
        )
        if not file_path:
            return

        try:
            mesh = pv.read(file_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Mesh Error",
                f"Could not load mesh:\n\n{exc}",
            )
            return

        self._reset_loaded_input()

        self.mesh = mesh
        self.mesh_path = os.path.abspath(file_path)
        self.input_mode = "single_mesh"

        self.plotter.add_mesh(
            self.mesh,
            show_edges=False,
        )
        self.plotter.reset_camera()

        self.mesh_label.setText(
            "Input mode: single mesh\n"
            f"{self.make_repo_relative_path(self.mesh_path)}"
        )

    def load_labelled_components(self):
        """
        Load a folder containing one dendrite mesh and individual spine meshes.

        Expected naming:
            dendrite*.ply
            spine*.ply
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Load Multiple Mesh Folder (nm)",
            self.get_repo_root(),
        )
        if not folder:
            return

        folder_path = Path(folder)
        dendrite_paths = sorted(folder_path.glob("dendrite*.ply"))
        spine_paths = sorted(folder_path.glob("spine*.ply"))

        if len(dendrite_paths) == 0:
            QMessageBox.warning(
                self,
                "No Dendrite Mesh",
                "No dendrite*.ply file was found in the selected folder.",
            )
            return

        if len(spine_paths) == 0:
            QMessageBox.warning(
                self,
                "No Spine Meshes",
                "No spine*.ply files were found in the selected folder.",
            )
            return

        dendrite_path = dendrite_paths[0]

        try:
            dendrite_mesh = pv.read(str(dendrite_path))
            spine_meshes = [pv.read(str(path)) for path in spine_paths]

            # One combined geometry object is stored for bounds and picking.
            # The individual components are still displayed separately.
            combined_mesh = pv.merge(
                [dendrite_mesh, *spine_meshes],
                merge_points=False,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Mesh Error",
                f"Could not load labelled components:\n\n{exc}",
            )
            return

        self._reset_loaded_input()

        self.input_mode = "labelled_components"
        self.labelled_dir = os.path.abspath(folder)
        self.dendrite_path = os.path.abspath(str(dendrite_path))
        self.spine_paths = [
            os.path.abspath(str(path)) for path in spine_paths
        ]
        self.mesh = combined_mesh

        # Display dendrite and spines as visually distinct components.
        self.plotter.add_mesh(
            dendrite_mesh,
            show_edges=False,
            color="lightgray",
        )
        for spine_mesh in spine_meshes:
            self.plotter.add_mesh(
                spine_mesh,
                show_edges=False,
                color="lightblue",
            )

        self.plotter.reset_camera()

        self.mesh_label.setText(
            "Input mode: multiple meshes (nm)\n"
            f"Folder: {self.make_repo_relative_path(self.labelled_dir)}\n"
            f"Dendrite: {dendrite_path.name}\n"
            f"Spines: {len(self.spine_paths)}"
        )

    # ------------------------------------------------------------------
    # Rendering volume and ROI
    # ------------------------------------------------------------------

    def get_render_volume_um(self):
        return (
            self.x_spin.value() * self.xy_spin.value(),
            self.y_spin.value() * self.xy_spin.value(),
            self.z_spin.value() * self.z_sampling_spin.value(),
        )

    def update_volume_label(self):
        size_x, size_y, size_z = self.get_render_volume_um()
        self.volume_label.setText(
            "Physical rendering volume:\n"
            f"{size_x:.3f} × {size_y:.3f} × {size_z:.3f} µm"
        )

    def settings_changed(self):
        self.update_volume_label()
        self.invalidate_exported_config()

        if self.roi_center_nm is not None:
            self.create_roi_at_center(tuple(self.roi_center_nm))

    def _require_mesh(self):
        """Return True when mesh geometry is loaded, otherwise show a warning."""
        if self.mesh is not None:
            return True

        QMessageBox.warning(
            self,
            "No Mesh",
            "Please load a mesh or labelled components first.",
        )
        return False

    def add_roi(self):
        if not self._require_mesh():
            return

        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        )
        self.create_roi_at_center(center)

    def create_roi_at_center(self, center):
        size_x_um, size_y_um, size_z_um = self.get_render_volume_um()
        size_x_nm = size_x_um * MESH_UNITS_PER_UM
        size_y_nm = size_y_um * MESH_UNITS_PER_UM
        size_z_nm = size_z_um * MESH_UNITS_PER_UM

        cx, cy, cz = (float(v) for v in center)
        bounds = (
            cx - size_x_nm / 2.0,
            cx + size_x_nm / 2.0,
            cy - size_y_nm / 2.0,
            cy + size_y_nm / 2.0,
            cz - size_z_nm / 2.0,
            cz + size_z_nm / 2.0,
        )

        if self.roi_actor is not None:
            try:
                self.plotter.remove_actor(self.roi_actor)
            except Exception:
                pass

        self.roi_actor = self.plotter.add_mesh(
            pv.Box(bounds=bounds),
            style="wireframe",
            color="orange",
            line_width=3,
        )
        self.roi_center_nm = (cx, cy, cz)
        self.roi_bounds = bounds

        self.roi_label.setText(
            "ROI center XYZ nm:\n"
            f"[{cx:.1f}, {cy:.1f}, {cz:.1f}]"
        )

        self.plotter.render()
        self.invalidate_exported_config()

    def enable_roi_movement(self):
        """Enable the original working ROI surface-picking interaction."""
        if self.mesh is None:
            return

        self.roi_label.setText(
            "Click on the mesh to place the ROI centre."
        )

        # This intentionally follows the original working GUI logic:
        # normal left-click surface picking with the cell picker.
        self.plotter.enable_surface_point_picking(
            callback=self.move_roi_to_point,
            show_message=False,
            show_point=False,
            left_clicking=True,
            picker="cell",
        )

    def move_roi_to_point(self, point):
        """Move the fixed-size ROI to the selected mesh surface point."""
        if point is None:
            return

        center = (
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

        self.create_roi_at_center(center)

    # ------------------------------------------------------------------
    # Configuration export
    # ------------------------------------------------------------------

    def invalidate_exported_config(self):
        self.exported_config_path = None

        if hasattr(self, "run_button"):
            self.run_button.setEnabled(False)

        if hasattr(self, "add_queue_button"):
            self.add_queue_button.setEnabled(False)

    def build_config(self):
        config = copy.deepcopy(DEFAULT_CONFIG)

        if self.input_mode == "labelled_components":
            config["input"] = {
                "mode": "labelled_components",
                "labelled_dir": self.make_repo_relative_path(
                    self.labelled_dir
                ),
                "dendrite_pattern": "dendrite*.ply",
                "spine_pattern": "spine*.ply",
                "scale_to_nm": 1.0,
                "recenter": False,
            }
        else:
            config["input"] = {
                "mode": "single_mesh",
                "mesh_path": self.make_repo_relative_path(
                    self.mesh_path
                ),
                "scale_to_nm": 1.0,
                "recenter": False,
            }

        config["grid"]["xy_um_per_px"] = float(self.xy_spin.value())
        config["grid"]["z_step_um"] = float(
            self.z_sampling_spin.value()
        )
        config["grid"]["output_shape_zyx"] = [
            int(self.z_spin.value()),
            int(self.y_spin.value()),
            int(self.x_spin.value()),
        ]
        config["grid"]["center_xyz_nm"] = [
            float(v) for v in self.roi_center_nm
        ]

        xmin, xmax, ymin, ymax, zmin, zmax = self.roi_bounds
        config["grid"]["selected_bounds_xyz_nm"] = {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "zmin": float(zmin),
            "zmax": float(zmax),
        }

        return config

    def export_config(self):
        if self.input_mode is None or self.mesh is None:
            QMessageBox.warning(
                self,
                "No Mesh",
                "Please load a single mesh or labelled components first.",
            )
            return

        if self.roi_center_nm is None:
            QMessageBox.warning(
                self,
                "No ROI",
                "Please add or select a rendering ROI first.",
            )
            return

        gui_config_dir = os.path.join(
            self.get_repo_root(),
            "gui",
            "configs",
        )
        os.makedirs(gui_config_dir, exist_ok=True)

        default_path = os.path.join(
            gui_config_dir,
            "gui_render.yaml",
        )
        

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Render Config",
            default_path,
            "YAML Files (*.yaml *.yml)",
        )
        if not output_path:
            return

        try:
            self.save_yaml(self.build_config(), output_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Could not save configuration:\n\n{exc}",
            )
            return

        self.exported_config_path = os.path.abspath(output_path)
        self.run_button.setEnabled(True)
        self.add_queue_button.setEnabled(True)

        self.render_log.append(
            f"\nConfig exported:\n{self.exported_config_path}\n"
        )
        QMessageBox.information(
            self,
            "Config Exported",
            "Render configuration exported successfully.",
        )

    @staticmethod
    def save_yaml(config, output_path):
        with open(output_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def run_render(self):
        if not self.exported_config_path:
            QMessageBox.warning(
                self,
                "No Config",
                "Please export a render config first.",
            )
            return

        if not self._renderer_is_idle():
            return

        render_script = self.get_render_script()
        if not os.path.exists(render_script):
            QMessageBox.critical(
                self,
                "Renderer Not Found",
                f"Could not find:\n{render_script}",
            )
            return

        self.render_log.clear()
        self.render_log.append("Starting render...\n")
        self.render_log.append(f"Python: {sys.executable}\n")
        self.render_log.append(
            f"Config: {self.exported_config_path}\n"
        )

        self.run_button.setEnabled(False)
        self.start_render_process(self.exported_config_path)

    def start_render_process(self, config_path):
        repo_root = self.get_repo_root()
        render_script = self.get_render_script()

        environment = self.render_process.processEnvironment()
        existing_pythonpath = environment.value("PYTHONPATH")
        pythonpath = (
            repo_root + os.pathsep + existing_pythonpath
            if existing_pythonpath
            else repo_root
        )

        environment.insert("PYTHONPATH", pythonpath)
        self.render_process.setProcessEnvironment(environment)
        self.render_process.setWorkingDirectory(repo_root)

        self.render_process.start(
            sys.executable,
            [render_script, "--config", config_path],
        )

    def _renderer_is_idle(self):
        if self.render_process.state() == QProcess.NotRunning:
            return True

        QMessageBox.warning(
            self,
            "Renderer Busy",
            "A rendering process is already running.",
        )
        return False

    # ------------------------------------------------------------------
    # Render queue
    # ------------------------------------------------------------------

    def add_to_queue(self):
        if not self.exported_config_path:
            QMessageBox.warning(
                self,
                "No Config",
                "Please export a render config first.",
            )
            return

        config_path = os.path.abspath(self.exported_config_path)
        queue_dir = os.path.join(
            self.get_repo_root(),
            "gui",
            "configs",
            "render_queue",
        )
        os.makedirs(queue_dir, exist_ok=True)

        job_number = len(self.render_queue) + 1
        source_name = os.path.splitext(
            os.path.basename(config_path)
        )[0]

        queue_config_path = os.path.join(
            queue_dir,
            f"{source_name}_job_{job_number:03d}.yaml",
        )

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

            config_data["output"]["output_name"] = (
                f"{source_name}_job_{job_number:03d}"
            )
            self.save_yaml(config_data, queue_config_path)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Queue Error",
                f"Could not create queued configuration:\n\n{exc}",
            )
            return

        self.render_queue.append(
            {
                "config": os.path.abspath(queue_config_path),
                "status": "Waiting",
            }
        )

        self.run_queue_button.setEnabled(True)
        self.update_queue_display()
        self.render_log.append(
            f"\nAdded render to queue:\n{queue_config_path}\n"
        )

    def update_queue_display(self):
        if not self.render_queue:
            self.queue_display.setPlainText("No renders queued.")
            return

        lines = [
            f"{index}. {os.path.basename(job['config'])}    "
            f"[{job['status']}]"
            for index, job in enumerate(self.render_queue, start=1)
        ]
        self.queue_display.setPlainText("\n".join(lines))

    def run_queue(self):
        if self.queue_running:
            return

        if not self._renderer_is_idle():
            return

        if not any(
            job["status"] == "Waiting"
            for job in self.render_queue
        ):
            QMessageBox.information(
                self,
                "Render Queue",
                "There are no waiting renders.",
            )
            return

        self.queue_running = True
        self.run_queue_button.setEnabled(False)
        self.run_button.setEnabled(False)

        self.render_log.append(
            "\n==============================\n"
            "Starting render queue\n"
            "==============================\n"
        )
        self.run_next_queue_job()

    def run_next_queue_job(self):
        next_job = next(
            (
                job
                for job in self.render_queue
                if job["status"] == "Waiting"
            ),
            None,
        )

        if next_job is None:
            self.finish_queue()
            return

        self.current_queue_job = next_job
        next_job["status"] = "Running"
        self.update_queue_display()

        config_path = next_job["config"]
        self.render_log.append(
            "\n--------------------------------\n"
            f"Starting queued render:\n{config_path}\n"
            "--------------------------------\n"
        )
        self.start_render_process(config_path)

    def finish_queue(self):
        self.queue_running = False
        self.current_queue_job = None
        self.run_queue_button.setEnabled(True)

        if self.exported_config_path:
            self.run_button.setEnabled(True)

        self.render_log.append(
            "\n==============================\n"
            "Render queue finished.\n"
            "==============================\n"
        )
        QMessageBox.information(
            self,
            "Render Queue",
            "All queued renders have finished.",
        )

    # ------------------------------------------------------------------
    # Process output and completion
    # ------------------------------------------------------------------

    def read_render_output(self):
        self._append_process_output(
            self.render_process.readAllStandardOutput()
        )

    def read_render_error(self):
        self._append_process_output(
            self.render_process.readAllStandardError()
        )

    def _append_process_output(self, byte_array):
        data = byte_array.data().decode(errors="replace")
        if not data:
            return

        cursor = self.render_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(data)
        self.render_log.setTextCursor(cursor)
        self.render_log.ensureCursorVisible()

    def render_finished(self, exit_code, exit_status):
        if self.queue_running and self.current_queue_job is not None:
            self.current_queue_job["status"] = (
                "Completed" if exit_code == 0 else "Failed"
            )

            message = (
                "\nQueued render completed successfully.\n"
                if exit_code == 0
                else f"\nQueued render failed with exit code {exit_code}.\n"
            )
            self.render_log.append(message)

            self.update_queue_display()
            self.current_queue_job = None
            self.run_next_queue_job()
            return

        if self.exported_config_path:
            self.run_button.setEnabled(True)

        self.render_log.append(
            f"\nRender finished with exit code {exit_code}\n"
        )

        if exit_code == 0:
            QMessageBox.information(
                self,
                "Render Complete",
                "Rendering completed successfully.",
            )
        else:
            QMessageBox.warning(
                self,
                "Render Failed",
                "Rendering failed.\nCheck the renderer log for details.",
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.render_process.state() != QProcess.NotRunning:
            self.render_process.kill()
            self.render_process.waitForFinished(2000)

        self.plotter.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
