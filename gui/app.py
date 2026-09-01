import sys
import os
import copy
import yaml

import pyvista as pv
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
)
from pyvistaqt import QtInteractor


NM_PER_UM = 1000.0


# ============================================================
# Default renderer configuration
# ============================================================
#
# These settings are used when the GUI exports a configuration.
# GUI controls override the relevant values such as mesh path,
# output shape, sampling and selected rendering position.
#
# Additional controls can later be added to the GUI for these
# parameters without changing the export architecture.
# ============================================================

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
        "path": (
            "psfs/"
            "psf_bornwolf_488nm_NA1_"
            "xy200nm_z500nm_65x65x13.tif"
        ),
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


class MeshViewer(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D Mesh Renderer")
        self.resize(1200, 800)

        # ----------------------------------------------------
        # Current mesh / ROI state
        # ----------------------------------------------------

        self.mesh = None
        self.mesh_path = None

        self.roi_actor = None
        self.roi_bounds = None
        self.roi_center_nm = None

        # ----------------------------------------------------
        # Main window
        # ----------------------------------------------------

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # ----------------------------------------------------
        # Load mesh
        # ----------------------------------------------------

        self.load_button = QPushButton("Load Mesh")
        self.load_button.clicked.connect(
            self.load_mesh
        )

        main_layout.addWidget(
            self.load_button
        )

        # ----------------------------------------------------
        # Output shape
        # ----------------------------------------------------

        shape_layout = QHBoxLayout()

        shape_layout.addWidget(
            QLabel("Output shape [Z,Y,X]:")
        )

        self.z_size = QSpinBox()
        self.z_size.setRange(1, 2048)
        self.z_size.setValue(64)

        self.y_size = QSpinBox()
        self.y_size.setRange(1, 4096)
        self.y_size.setValue(128)

        self.x_size = QSpinBox()
        self.x_size.setRange(1, 4096)
        self.x_size.setValue(128)

        shape_layout.addWidget(
            QLabel("Z")
        )
        shape_layout.addWidget(
            self.z_size
        )

        shape_layout.addWidget(
            QLabel("Y")
        )
        shape_layout.addWidget(
            self.y_size
        )

        shape_layout.addWidget(
            QLabel("X")
        )
        shape_layout.addWidget(
            self.x_size
        )

        main_layout.addLayout(
            shape_layout
        )

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        sampling_layout = QHBoxLayout()

        sampling_layout.addWidget(
            QLabel("Sampling:")
        )

        self.xy_sampling = QDoubleSpinBox()
        self.xy_sampling.setDecimals(3)
        self.xy_sampling.setRange(
            0.001,
            10.0,
        )
        self.xy_sampling.setSingleStep(
            0.001
        )
        self.xy_sampling.setValue(
            0.094
        )

        self.z_sampling = QDoubleSpinBox()
        self.z_sampling.setDecimals(3)
        self.z_sampling.setRange(
            0.001,
            20.0,
        )
        self.z_sampling.setSingleStep(
            0.1
        )
        self.z_sampling.setValue(
            0.5
        )

        sampling_layout.addWidget(
            QLabel("XY")
        )
        sampling_layout.addWidget(
            self.xy_sampling
        )
        sampling_layout.addWidget(
            QLabel("um/pixel")
        )

        sampling_layout.addWidget(
            QLabel("Z")
        )
        sampling_layout.addWidget(
            self.z_sampling
        )
        sampling_layout.addWidget(
            QLabel("um/slice")
        )

        main_layout.addLayout(
            sampling_layout
        )

        # ----------------------------------------------------
        # Physical rendering volume
        # ----------------------------------------------------

        self.volume_label = QLabel()

        main_layout.addWidget(
            self.volume_label
        )

        self.z_size.valueChanged.connect(
            self.settings_changed
        )

        self.y_size.valueChanged.connect(
            self.settings_changed
        )

        self.x_size.valueChanged.connect(
            self.settings_changed
        )

        self.xy_sampling.valueChanged.connect(
            self.settings_changed
        )

        self.z_sampling.valueChanged.connect(
            self.settings_changed
        )

        self.update_volume_label()

        # ----------------------------------------------------
        # ROI controls
        # ----------------------------------------------------

        roi_button_layout = QHBoxLayout()

        self.roi_button = QPushButton(
            "Add Rendering ROI"
        )

        self.roi_button.clicked.connect(
            self.add_roi_box
        )

        self.roi_button.setEnabled(
            False
        )

        roi_button_layout.addWidget(
            self.roi_button
        )

        self.move_roi_button = QPushButton(
            "Move ROI"
        )

        self.move_roi_button.clicked.connect(
            self.enable_roi_movement
        )

        self.move_roi_button.setEnabled(
            False
        )

        roi_button_layout.addWidget(
            self.move_roi_button
        )

        main_layout.addLayout(
            roi_button_layout
        )

        # ----------------------------------------------------
        # ROI information
        # ----------------------------------------------------

        self.roi_label = QLabel(
            "ROI center: not selected"
        )

        main_layout.addWidget(
            self.roi_label
        )

        # ----------------------------------------------------
        # Export configuration
        # ----------------------------------------------------

        self.export_button = QPushButton(
            "Export Config"
        )

        self.export_button.clicked.connect(
            self.export_config
        )

        self.export_button.setEnabled(
            False
        )

        main_layout.addWidget(
            self.export_button
        )

        # ----------------------------------------------------
        # 3D viewer
        # ----------------------------------------------------

        self.plotter = QtInteractor(
            self
        )

        main_layout.addWidget(
            self.plotter.interactor
        )

        self.plotter.set_background(
            "white"
        )

        self.plotter.add_axes()

    # ========================================================
    # Physical rendering size
    # ========================================================

    def get_physical_size_um(self):

        z = self.z_size.value()
        y = self.y_size.value()
        x = self.x_size.value()

        xy = self.xy_sampling.value()
        z_step = self.z_sampling.value()

        size_x_um = x * xy
        size_y_um = y * xy
        size_z_um = z * z_step

        return (
            size_x_um,
            size_y_um,
            size_z_um,
        )

    # ========================================================
    # Update physical volume label
    # ========================================================

    def update_volume_label(self):

        (
            size_x_um,
            size_y_um,
            size_z_um,
        ) = self.get_physical_size_um()

        self.volume_label.setText(
            "Calculated rendering volume: "
            f"X = {size_x_um:.3f} um | "
            f"Y = {size_y_um:.3f} um | "
            f"Z = {size_z_um:.3f} um"
        )

    # ========================================================
    # Shape / sampling changed
    # ========================================================

    def settings_changed(self):

        self.update_volume_label()

        # Keep the current ROI center while updating
        # the physical size of the rendering volume.
        if self.roi_center_nm is not None:

            self.create_roi_at_center(
                self.roi_center_nm
            )

    # ========================================================
    # Load mesh
    # ========================================================

    def load_mesh(self):

        file_path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Select Mesh",
                "",
                "Mesh files (*.ply *.stl *.obj);;"
                "All files (*)",
            )
        )

        if not file_path:
            return

        try:

            self.mesh = pv.read(
                file_path
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Mesh Error",
                "Could not load the mesh:\n"
                f"{exc}",
            )

            return

        self.mesh_path = file_path

        # Reset previous ROI.
        self.roi_actor = None
        self.roi_bounds = None
        self.roi_center_nm = None

        # Clear viewer.
        self.plotter.clear()

        # Display mesh.
        self.plotter.add_mesh(
            self.mesh,
            color="lightblue",
        )

        self.plotter.add_axes()
        self.plotter.reset_camera()

        self.roi_button.setEnabled(
            True
        )

        self.move_roi_button.setEnabled(
            False
        )

        self.export_button.setEnabled(
            False
        )

        self.roi_label.setText(
            "ROI center: not selected"
        )

    # ========================================================
    # Add initial rendering ROI
    # ========================================================

    def add_roi_box(self):

        if self.mesh is None:
            return

        (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) = self.mesh.bounds

        # Initial ROI is placed at the center
        # of the complete mesh bounding box.
        center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        )

        self.create_roi_at_center(
            center
        )

        self.move_roi_button.setEnabled(
            True
        )

    # ========================================================
    # Create fixed-size ROI
    # ========================================================

    def create_roi_at_center(
        self,
        center,
    ):

        (
            center_x,
            center_y,
            center_z,
        ) = center

        (
            size_x_um,
            size_y_um,
            size_z_um,
        ) = self.get_physical_size_um()

        # Mesh coordinates are in nanometers.
        size_x_nm = (
            size_x_um * NM_PER_UM
        )

        size_y_nm = (
            size_y_um * NM_PER_UM
        )

        size_z_nm = (
            size_z_um * NM_PER_UM
        )

        # ----------------------------------------------------
        # Calculate physical ROI bounds
        # ----------------------------------------------------

        bounds = (
            center_x - size_x_nm / 2.0,
            center_x + size_x_nm / 2.0,

            center_y - size_y_nm / 2.0,
            center_y + size_y_nm / 2.0,

            center_z - size_z_nm / 2.0,
            center_z + size_z_nm / 2.0,
        )

        # ----------------------------------------------------
        # Remove previous ROI
        # ----------------------------------------------------

        if self.roi_actor is not None:

            self.plotter.remove_actor(
                self.roi_actor
            )

        # ----------------------------------------------------
        # Create fixed wireframe ROI
        # ----------------------------------------------------

        roi_box = pv.Box(
            bounds=bounds
        )

        self.roi_actor = (
            self.plotter.add_mesh(
                roi_box,
                style="wireframe",
                line_width=3,
                color="black",
            )
        )

        # ----------------------------------------------------
        # Store GUI-selected ROI
        # ----------------------------------------------------

        self.roi_center_nm = (
            float(center_x),
            float(center_y),
            float(center_z),
        )

        self.roi_bounds = bounds

        # ----------------------------------------------------
        # Display ROI information
        # ----------------------------------------------------

        self.roi_label.setText(
            "ROI center XYZ (nm): "
            f"[{center_x:.1f}, "
            f"{center_y:.1f}, "
            f"{center_z:.1f}]"
            "    |    "
            "Size XYZ (um): "
            f"[{size_x_um:.3f}, "
            f"{size_y_um:.3f}, "
            f"{size_z_um:.3f}]"
        )

        self.export_button.setEnabled(
            True
        )

        self.plotter.render()

    # ========================================================
    # Enable ROI movement
    # ========================================================

    def enable_roi_movement(self):

        if self.mesh is None:
            return

        self.roi_label.setText(
            "Click on the mesh to place "
            "the ROI centre."
        )

        self.plotter.enable_surface_point_picking(
            callback=self.move_roi_to_point,
            show_message=False,
            show_point=False,
            left_clicking=True,
            picker="cell",
        )

    # ========================================================
    # Move ROI
    # ========================================================

    def move_roi_to_point(
        self,
        point,
    ):

        if point is None:
            return

        center = (
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

        self.create_roi_at_center(
            center
        )

    # ========================================================
    # Repository-relative mesh path
    # ========================================================

    def get_relative_mesh_path(self):

        if self.mesh_path is None:
            return None

        try:

            # app.py is located in <repo>/gui/.
            repo_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                )
            )

            relative_path = os.path.relpath(
                self.mesh_path,
                repo_root,
            )

            # Store portable paths in YAML.
            relative_path = (
                relative_path.replace(
                    "\\",
                    "/",
                )
            )

            return relative_path

        except ValueError:

            # Fallback for paths on different
            # Windows drives.
            return self.mesh_path.replace(
                "\\",
                "/",
            )

    # ========================================================
    # Build complete renderer configuration
    # ========================================================

    def build_render_config(self):

        # Deep copy prevents GUI exports from modifying
        # the global default dictionary.
        config = copy.deepcopy(
            DEFAULT_CONFIG
        )

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        config["input"]["mesh_path"] = (
            self.get_relative_mesh_path()
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        config["output"]["output_dir"] = (
            "outputs/gui_results"
        )

        config["output"]["output_name"] = (
            "gui_render"
        )

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        config["grid"][
            "xy_um_per_px"
        ] = float(
            self.xy_sampling.value()
        )

        config["grid"][
            "z_step_um"
        ] = float(
            self.z_sampling.value()
        )

        # ----------------------------------------------------
        # Fixed output shape
        # ----------------------------------------------------

        config["grid"][
            "shape_mode"
        ] = "fixed"

        config["grid"][
            "output_shape_zyx"
        ] = [
            int(self.z_size.value()),
            int(self.y_size.value()),
            int(self.x_size.value()),
        ]

        # ----------------------------------------------------
        # GUI-selected location
        # ----------------------------------------------------

        config["grid"][
            "use_roi"
        ] = False

        config["grid"][
            "center_xyz_nm"
        ] = [
            float(
                self.roi_center_nm[0]
            ),
            float(
                self.roi_center_nm[1]
            ),
            float(
                self.roi_center_nm[2]
            ),
        ]

        # ----------------------------------------------------
        # Store selected cube bounds
        # ----------------------------------------------------

        (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) = self.roi_bounds

        config["grid"][
            "selected_bounds_xyz_nm"
        ] = {
            "xmin": float(xmin),
            "xmax": float(xmax),

            "ymin": float(ymin),
            "ymax": float(ymax),

            "zmin": float(zmin),
            "zmax": float(zmax),
        }

        return config

    # ========================================================
    # Export renderer configuration
    # ========================================================

    def export_config(self):

        # ----------------------------------------------------
        # Validate GUI state
        # ----------------------------------------------------

        if self.mesh_path is None:

            QMessageBox.warning(
                self,
                "No Mesh",
                "Please load a mesh first.",
            )

            return

        if (
            self.roi_center_nm is None
            or self.roi_bounds is None
        ):

            QMessageBox.warning(
                self,
                "No ROI",
                "Please select a rendering "
                "ROI first.",
            )

            return

        # ----------------------------------------------------
        # Build complete config directly from GUI
        # ----------------------------------------------------

        config = self.build_render_config()

        # ----------------------------------------------------
        # Select destination
        # ----------------------------------------------------

        output_path, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Save Render Config",
                "configs/gui_render.yaml",
                "YAML files (*.yaml *.yml)",
            )
        )

        if not output_path:
            return

        if not output_path.lower().endswith(
            (
                ".yaml",
                ".yml",
            )
        ):

            output_path += ".yaml"

        # ----------------------------------------------------
        # Write YAML
        # ----------------------------------------------------

        try:

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as file:

                yaml.safe_dump(
                    config,
                    file,
                    sort_keys=False,
                    default_flow_style=False,
                )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Export Error",
                "Could not save the "
                "configuration:\n"
                f"{exc}",
            )

            return

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        QMessageBox.information(
            self,
            "Config Exported",
            "Render configuration saved "
            "successfully:\n"
            f"{output_path}"
            "\n\n"
            "Results will be written to:\n"
            "outputs/gui_results",
        )


# ============================================================
# Application
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = MeshViewer()

    window.show()

    sys.exit(
        app.exec()
    )