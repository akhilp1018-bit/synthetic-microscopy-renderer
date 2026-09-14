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
    4. Add, position, and orient the rendering ROI.
    5. Export the YAML configuration.
    6. Run immediately or add the render to the queue.

See gui/README.md for detailed usage and configuration information.
"""


import copy
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile
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
from vtkmodules.vtkCommonTransforms import vtkTransform


MESH_UNITS_PER_UM = 1000.0

# The GUI uses configs/default.yaml as its base configuration. Rendering, PSF,
# noise, masks, and renderer-specific defaults stay owned by render.py/config.
_GUI_OUTPUT_DIR = "outputs/gui_results"
_GUI_OUTPUT_NAME = "gui_render"



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
        self.automatic_anchor_nm = None
        self.mesh_actors = []
        self.render_volume_actor = None
        self.render_volume_grid = None
        self.last_rendered_image_path = None

        # ROI interaction state used to reproduce the selected rendering volume.
        # The transform maps the mouse-positioned ROI into renderer-local coordinates.
        self.roi_actor = None
        self.roi_widget = None
        self.roi_center_nm = None
        self.roi_initial_center_nm = None
        self.roi_bounds = None
        self.roi_rotation_deg_xyz = (0.0, 0.0, 0.0)
        self.roi_world_to_local_4x4 = np.eye(4, dtype=np.float64)

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
        self._build_volume_view_controls()
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

        view_layout = QHBoxLayout()

        self.view_xy_button = QPushButton("XY")
        self.view_xy_button.clicked.connect(self.view_roi_xy)

        self.view_xz_button = QPushButton("XZ")
        self.view_xz_button.clicked.connect(self.view_roi_xz)

        self.view_yz_button = QPushButton("YZ")
        self.view_yz_button.clicked.connect(self.view_roi_yz)

        self.view_3d_button = QPushButton("3D")
        self.view_3d_button.clicked.connect(self.view_roi_3d)

        view_layout.addWidget(self.view_xy_button)
        view_layout.addWidget(self.view_xz_button)
        view_layout.addWidget(self.view_yz_button)
        view_layout.addWidget(self.view_3d_button)
        self.left_layout.addLayout(view_layout)

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

    def _build_volume_view_controls(self):
        """Controls for viewing the most recently rendered TIFF volume."""
        layout = QHBoxLayout()

        self.view_render_button = QPushButton("View Rendered Volume")
        self.view_render_button.setEnabled(False)
        self.view_render_button.clicked.connect(self.view_rendered_volume)

        self.back_mesh_button = QPushButton("Back to Mesh")
        self.back_mesh_button.setEnabled(False)
        self.back_mesh_button.clicked.connect(self.back_to_mesh_view)

        layout.addWidget(self.view_render_button)
        layout.addWidget(self.back_mesh_button)
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

    def get_base_config_path(self):
        return os.path.join(
            self.get_repo_root(),
            "configs",
            "default.yaml",
        )

    def load_base_config(self):
        """Load the same baseline configuration used by render.py."""
        config_path = self.get_base_config_path()
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Base renderer config not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "configs/default.yaml must contain a YAML mapping."
            )

        return copy.deepcopy(config)

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
        self.automatic_anchor_nm = None
        self.mesh_actors = []
        self.render_volume_actor = None
        self.render_volume_grid = None
        self.last_rendered_image_path = None

        if hasattr(self, "view_render_button"):
            self.view_render_button.setEnabled(False)
        if hasattr(self, "back_mesh_button"):
            self.back_mesh_button.setEnabled(False)

        try:
            self.plotter.disable_picking()
        except Exception:
            pass

        try:
            self.plotter.clear_box_widgets()
        except Exception:
            pass

        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.add_axes()

        self.roi_actor = None
        self.roi_widget = None
        self.roi_center_nm = None
        self.roi_initial_center_nm = None
        self.roi_bounds = None
        self.roi_rotation_deg_xyz = (0.0, 0.0, 0.0)
        self.roi_world_to_local_4x4 = np.eye(4, dtype=np.float64)

        self.roi_label.setText("No ROI selected")
        self.invalidate_exported_config()

    @staticmethod
    def _preview_anchor_from_mesh(mesh):
        """Mirror the renderer anchor rule for ROI visualization only.

        The exported untouched ROI does not force this centre into the YAML;
        render.py still chooses the authoritative automatic anchor itself.
        """
        if mesh is None or mesh.n_points == 0:
            raise ValueError("Mesh contains no vertices.")

        vertices = np.asarray(mesh.points, dtype=np.float64)
        mean_xyz = vertices.mean(axis=0)
        distances_sq = np.sum((vertices - mean_xyz) ** 2, axis=1)
        anchor = vertices[int(np.argmin(distances_sq))]
        return tuple(float(v) for v in anchor)

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
        self.automatic_anchor_nm = self._preview_anchor_from_mesh(mesh)

        actor = self.plotter.add_mesh(
            self.mesh,
            show_edges=False,
        )
        self.mesh_actors = [actor]
        self.plotter.reset_camera()

        self.mesh_label.setText(
            "Input mode: single mesh\n"
            f"{self.make_repo_relative_path(self.mesh_path)}"
        )

    def load_labelled_components(self):
        """Load a folder containing dendrite and individual spine meshes."""
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
        # Manual rendering uses the dendrite mesh to choose the automatic
        # fixed-grid anchor for labelled-component input.
        self.automatic_anchor_nm = self._preview_anchor_from_mesh(
            dendrite_mesh
        )

        self.mesh_actors = []
        dendrite_actor = self.plotter.add_mesh(
            dendrite_mesh,
            show_edges=False,
            color="lightgray",
        )
        self.mesh_actors.append(dendrite_actor)

        for spine_mesh in spine_meshes:
            spine_actor = self.plotter.add_mesh(
                spine_mesh,
                show_edges=False,
                color="lightblue",
            )
            self.mesh_actors.append(spine_actor)

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

    def get_render_extent_um(self):
        """Return the renderer grid extent between first/last voxel centres."""
        return (
            max(self.x_spin.value() - 1, 0) * self.xy_spin.value(),
            max(self.y_spin.value() - 1, 0) * self.xy_spin.value(),
            max(self.z_spin.value() - 1, 0)
            * self.z_sampling_spin.value(),
        )

    def update_volume_label(self):
        size_x, size_y, size_z = self.get_render_extent_um()
        self.volume_label.setText(
            "Render grid extent:\n"
            f"{size_x:.3f} × {size_y:.3f} × {size_z:.3f} µm"
        )

    def settings_changed(self):
        self.update_volume_label()
        self.invalidate_exported_config()

        if self.roi_center_nm is not None:
            # Recreate the ROI at its current centre using the selected physical size.
            # Rotation is reset when sampling or output dimensions change so the
            # exported ROI transform remains unambiguous and reproducible.
            self.create_roi_at_center(tuple(self.roi_center_nm))

    def _require_mesh(self):
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

        if self.automatic_anchor_nm is None:
            QMessageBox.warning(
                self,
                "Anchor Error",
                "Could not determine the automatic mesh anchor.",
            )
            return

        # Start from exactly the same automatic anchor as manual fixed-grid
        # rendering. Mouse movement and rotation are optional adjustments.
        self.create_roi_at_center(self.automatic_anchor_nm)

    def create_roi_at_center(self, center):
        """Create a fixed-size mouse-movable and mouse-rotatable ROI."""
        size_x_um, size_y_um, size_z_um = self.get_render_extent_um()
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

        try:
            self.plotter.clear_box_widgets()
        except Exception:
            pass

        if self.roi_actor is not None:
            try:
                self.plotter.remove_actor(self.roi_actor)
            except Exception:
                pass

        self.roi_actor = None
        self.roi_widget = None
        self.roi_center_nm = (cx, cy, cz)
        self.roi_initial_center_nm = (cx, cy, cz)
        self.roi_bounds = bounds
        self.roi_rotation_deg_xyz = (0.0, 0.0, 0.0)

        # The renderer uses ROI-local coordinates for reproducible placement.
        # The initial transform translates the selected ROI centre to the
        # local origin; mouse rotation is incorporated by the widget callback.
        self.roi_world_to_local_4x4 = np.eye(4, dtype=np.float64)
        self.roi_world_to_local_4x4[:3, 3] = -np.asarray(
            self.roi_initial_center_nm,
            dtype=np.float64,
        )

        # A translucent filled box makes depth relationships easier to judge
        # than the wireframe widget alone. The actor follows the same widget
        # transform used for the exported ROI pose.
        roi_display_mesh = pv.Box(bounds=bounds)
        self.roi_actor = self.plotter.add_mesh(
            roi_display_mesh,
            color="orange",
            opacity=0.14,
            show_edges=False,
            pickable=False,
            reset_camera=False,
        )

        self.roi_widget = self.plotter.add_box_widget(
            callback=self.roi_widget_changed,
            bounds=bounds,
            rotation_enabled=True,
            use_planes=False,
            color="orange",
            pass_widget=True,
            interaction_event="always",
        )

        # Prevent resizing through the mouse widget. ROI dimensions are controlled
        # by the output shape and spatial sampling parameters.
        try:
            self.roi_widget.SetScalingEnabled(False)
        except Exception:
            pass

        self.roi_label.setText(
            "ROI starts at the renderer automatic mesh anchor.\n"
            "Use the mouse to move or rotate it if needed.\n"
            "Use XY/XZ/YZ to verify depth placement.\n"
            f"Center: [{cx:.1f}, {cy:.1f}, {cz:.1f}] nm\n"
            "Rotation XYZ: [0.0°, 0.0°, 0.0°]"
        )

        self.plotter.render()
        self.invalidate_exported_config()

    def roi_widget_changed(self, box, widget):
        """Update ROI centre/orientation and world-to-local transform."""
        if widget is None or self.roi_initial_center_nm is None:
            return

        transform = vtkTransform()
        widget.GetTransform(transform)
        vtk_matrix = transform.GetMatrix()

        widget_matrix = np.array(
            [
                [vtk_matrix.GetElement(r, c) for c in range(4)]
                for r in range(4)
            ],
            dtype=np.float64,
        )

        if self.roi_actor is not None:
            try:
                self.roi_actor.SetUserTransform(transform)
            except Exception:
                pass

        initial_center_h = np.array(
            [*self.roi_initial_center_nm, 1.0],
            dtype=np.float64,
        )
        current_center_h = widget_matrix @ initial_center_h
        current_center = current_center_h[:3]

        self.roi_center_nm = tuple(float(v) for v in current_center)

        orientation = transform.GetOrientation()
        self.roi_rotation_deg_xyz = tuple(
            float(v) for v in orientation
        )

        # GetTransform maps the initially placed box to its current pose.
        # Inverting it maps current world coordinates back to the original
        # axis-aligned box. We then shift the original box centre to zero.
        inverse_widget_matrix = np.linalg.inv(widget_matrix)
        translate_initial_center_to_origin = np.eye(4, dtype=np.float64)
        translate_initial_center_to_origin[:3, 3] = -np.asarray(
            self.roi_initial_center_nm,
            dtype=np.float64,
        )
        self.roi_world_to_local_4x4 = (
            translate_initial_center_to_origin @ inverse_widget_matrix
        )

        if box is not None:
            self.roi_bounds = tuple(float(v) for v in box.bounds)

        cx, cy, cz = self.roi_center_nm
        rx, ry, rz = self.roi_rotation_deg_xyz
        self.roi_label.setText(
            "Mouse-controlled ROI\n"
            f"Center: [{cx:.1f}, {cy:.1f}, {cz:.1f}] nm\n"
            f"Rotation XYZ: [{rx:.1f}°, {ry:.1f}°, {rz:.1f}°]"
        )

        self.invalidate_exported_config()

    def view_roi_xy(self):
        """Show an orthographic XY view along the Z axis."""
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()
        self.plotter.render()

    def view_roi_xz(self):
        """Show an orthographic XZ view along the Y axis."""
        self.plotter.enable_parallel_projection()
        self.plotter.view_xz()
        self.plotter.render()

    def view_roi_yz(self):
        """Show an orthographic YZ view along the X axis."""
        self.plotter.enable_parallel_projection()
        self.plotter.view_yz()
        self.plotter.render()

    def view_roi_3d(self):
        """Return to a perspective isometric 3D view."""
        self.plotter.disable_parallel_projection()
        self.plotter.view_isometric()
        self.plotter.render()

    def enable_roi_movement(self):
        """Enable surface-click placement of the rendering ROI."""
        if self.mesh is None:
            return

        self.roi_label.setText(
            "Click on the mesh to place the ROI centre. "
            "After placement, drag the ROI widget to rotate/move it."
        )

        self.plotter.enable_surface_point_picking(
            callback=self.move_roi_to_point,
            show_message=False,
            show_point=False,
            left_clicking=True,
            picker="cell",
        )

    def move_roi_to_point(self, point):
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
        """Build a small GUI override on top of configs/default.yaml.

        render.py remains the source of truth for renderer, PSF, noise, mask,
        and fixed-grid behaviour. The GUI only supplies input, sampling, shape,
        and the optional user-selected ROI pose.
        """
        config = self.load_base_config()

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

        config.setdefault("output", {})
        config["output"]["output_dir"] = _GUI_OUTPUT_DIR
        config["output"]["output_name"] = _GUI_OUTPUT_NAME

        grid = config.setdefault("grid", {})
        grid["xy_um_per_px"] = float(self.xy_spin.value())
        grid["z_step_um"] = float(self.z_sampling_spin.value())
        grid["use_roi"] = False
        grid["shape_mode"] = "fixed"
        grid["output_shape_zyx"] = [
            int(self.z_spin.value()),
            int(self.y_spin.value()),
            int(self.x_spin.value()),
        ]

        # Remove optional GUI fields that may already be present in a reused
        # config. They are added back only when required by the current ROI.
        for key in (
            "center_xyz_nm",
            "selected_bounds_xyz_nm",
            "roi_rotation_deg_xyz",
            "roi_world_to_local_4x4",
        ):
            grid.pop(key, None)

        rotation = np.asarray(
            self.roi_rotation_deg_xyz, dtype=np.float64
        )
        has_rotation = bool(np.any(np.abs(rotation) > 1e-6))

        if has_rotation:
            # Rotation requires the exact world-to-local transform. render.py
            # applies it during mesh preparation and renders around local zero.
            grid["roi_world_to_local_4x4"] = (
                self.roi_world_to_local_4x4.tolist()
            )
        else:
            # Pass the current ROI centre directly to render.py. This avoids
            # reloading a very large mesh only to recompute the automatic anchor.
            # When the ROI is untouched this is the same centre shown by the GUI.
            grid["center_xyz_nm"] = [
                float(v) for v in self.roi_center_nm
            ]

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
    # Rendered-volume visualization
    # ------------------------------------------------------------------

    def _find_rendered_image_path(self, config_path=None):
        """Locate the rendered image TIFF for a completed GUI render."""
        config_path = config_path or self.exported_config_path
        if not config_path or not os.path.exists(config_path):
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
        except Exception:
            return None

        output = config.get("output", {})
        output_dir = output.get("output_dir", _GUI_OUTPUT_DIR)
        output_name = output.get("output_name", _GUI_OUTPUT_NAME)

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.get_repo_root(), output_dir)

        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(output_dir):
            return None

        candidates = sorted(
            Path(output_dir).glob(f"zstack_{output_name}_*_image.tif"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None

        return str(candidates[0])

    def view_rendered_volume(self):
        """Display the most recent rendered TIFF as a 3D intensity volume."""
        image_path = self.last_rendered_image_path
        if image_path is None or not os.path.exists(image_path):
            image_path = self._find_rendered_image_path()

        if image_path is None:
            QMessageBox.warning(
                self,
                "Rendered Volume",
                "No rendered image TIFF was found. Run a render first.",
            )
            return

        try:
            volume_zyx = tifffile.imread(image_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Rendered Volume Error",
                f"Could not load rendered TIFF:\n\n{exc}",
            )
            return

        volume_zyx = np.asarray(volume_zyx)
        if volume_zyx.ndim != 3:
            QMessageBox.warning(
                self,
                "Rendered Volume",
                f"Expected a 3D TIFF stack, got shape {volume_zyx.shape}.",
            )
            return

        if not np.any(np.isfinite(volume_zyx)):
            QMessageBox.warning(
                self,
                "Rendered Volume",
                "The rendered volume contains no finite intensity values.",
            )
            return

        # Hide mesh/ROI while showing the rendered local volume.
        for actor in self.mesh_actors:
            try:
                actor.SetVisibility(False)
            except Exception:
                pass

        if self.roi_actor is not None:
            try:
                self.roi_actor.SetVisibility(False)
            except Exception:
                pass

        if self.roi_widget is not None:
            try:
                self.roi_widget.SetEnabled(False)
            except Exception:
                pass

        if self.render_volume_actor is not None:
            try:
                self.plotter.remove_actor(self.render_volume_actor)
            except Exception:
                pass
            self.render_volume_actor = None

        # TIFF data are stored ZYX. PyVista expects an XYZ scalar layout.
        values_xyz = np.transpose(volume_zyx, (2, 1, 0))

        grid = pv.ImageData()
        grid.dimensions = values_xyz.shape
        grid.spacing = (
            float(self.xy_spin.value()) * MESH_UNITS_PER_UM,
            float(self.xy_spin.value()) * MESH_UNITS_PER_UM,
            float(self.z_sampling_spin.value()) * MESH_UNITS_PER_UM,
        )
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data["intensity"] = values_xyz.ravel(order="F")

        self.render_volume_grid = grid
        self.render_volume_actor = self.plotter.add_volume(
            grid,
            scalars="intensity",
            cmap="gray",
            opacity="sigmoid",
            shade=False,
        )

        self.plotter.set_background("black")
        self.plotter.disable_parallel_projection()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

        self.last_rendered_image_path = image_path
        self.back_mesh_button.setEnabled(True)

        self.render_log.append(
            f"\nViewing rendered volume:\n{image_path}\n"
        )

    def back_to_mesh_view(self):
        """Hide the rendered TIFF volume and return to the mesh/ROI view."""
        if self.render_volume_actor is not None:
            try:
                self.plotter.remove_actor(self.render_volume_actor)
            except Exception:
                pass
            self.render_volume_actor = None
            self.render_volume_grid = None

        for actor in self.mesh_actors:
            try:
                actor.SetVisibility(True)
            except Exception:
                pass

        if self.roi_actor is not None:
            try:
                self.roi_actor.SetVisibility(True)
            except Exception:
                pass

        if self.roi_widget is not None:
            try:
                self.roi_widget.SetEnabled(True)
            except Exception:
                pass

        self.plotter.set_background("white")
        self.plotter.disable_parallel_projection()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

        self.back_mesh_button.setEnabled(False)

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
            self.last_rendered_image_path = self._find_rendered_image_path(
                self.exported_config_path
            )
            self.view_render_button.setEnabled(
                self.last_rendered_image_path is not None
            )

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
