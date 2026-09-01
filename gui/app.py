import sys

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
)
from pyvistaqt import QtInteractor


NM_PER_UM = 1000.0


class MeshViewer(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D Mesh Renderer")
        self.resize(1200, 800)

        self.mesh = None

        self.roi_actor = None
        self.roi_bounds = None
        self.roi_center_nm = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # ----------------------------------------------------
        # Load mesh
        # ----------------------------------------------------

        self.load_button = QPushButton("Load Mesh")
        self.load_button.clicked.connect(self.load_mesh)
        main_layout.addWidget(self.load_button)

        # ----------------------------------------------------
        # Output shape
        # ----------------------------------------------------

        shape_layout = QHBoxLayout()

        shape_layout.addWidget(QLabel("Output shape [Z,Y,X]:"))

        self.z_size = QSpinBox()
        self.z_size.setRange(1, 2048)
        self.z_size.setValue(64)

        self.y_size = QSpinBox()
        self.y_size.setRange(1, 4096)
        self.y_size.setValue(128)

        self.x_size = QSpinBox()
        self.x_size.setRange(1, 4096)
        self.x_size.setValue(128)

        shape_layout.addWidget(QLabel("Z"))
        shape_layout.addWidget(self.z_size)

        shape_layout.addWidget(QLabel("Y"))
        shape_layout.addWidget(self.y_size)

        shape_layout.addWidget(QLabel("X"))
        shape_layout.addWidget(self.x_size)

        main_layout.addLayout(shape_layout)

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        sampling_layout = QHBoxLayout()

        sampling_layout.addWidget(QLabel("Sampling:"))

        self.xy_sampling = QDoubleSpinBox()
        self.xy_sampling.setDecimals(3)
        self.xy_sampling.setRange(0.001, 10.0)
        self.xy_sampling.setSingleStep(0.001)
        self.xy_sampling.setValue(0.094)

        self.z_sampling = QDoubleSpinBox()
        self.z_sampling.setDecimals(3)
        self.z_sampling.setRange(0.001, 20.0)
        self.z_sampling.setSingleStep(0.1)
        self.z_sampling.setValue(0.5)

        sampling_layout.addWidget(QLabel("XY"))
        sampling_layout.addWidget(self.xy_sampling)
        sampling_layout.addWidget(QLabel("um/pixel"))

        sampling_layout.addWidget(QLabel("Z"))
        sampling_layout.addWidget(self.z_sampling)
        sampling_layout.addWidget(QLabel("um/slice"))

        main_layout.addLayout(sampling_layout)

        # ----------------------------------------------------
        # Calculated physical volume
        # ----------------------------------------------------

        self.volume_label = QLabel()
        main_layout.addWidget(self.volume_label)

        self.z_size.valueChanged.connect(self.settings_changed)
        self.y_size.valueChanged.connect(self.settings_changed)
        self.x_size.valueChanged.connect(self.settings_changed)

        self.xy_sampling.valueChanged.connect(self.settings_changed)
        self.z_sampling.valueChanged.connect(self.settings_changed)

        self.update_volume_label()

        # ----------------------------------------------------
        # ROI buttons
        # ----------------------------------------------------

        roi_button_layout = QHBoxLayout()

        self.roi_button = QPushButton("Add Rendering ROI")
        self.roi_button.clicked.connect(self.add_roi_box)
        self.roi_button.setEnabled(False)

        roi_button_layout.addWidget(self.roi_button)

        self.move_roi_button = QPushButton("Move ROI")
        self.move_roi_button.clicked.connect(self.enable_roi_movement)
        self.move_roi_button.setEnabled(False)

        roi_button_layout.addWidget(self.move_roi_button)

        main_layout.addLayout(roi_button_layout)

        self.roi_label = QLabel("ROI center: not selected")
        main_layout.addWidget(self.roi_label)

        # ----------------------------------------------------
        # 3D viewer
        # ----------------------------------------------------

        self.plotter = QtInteractor(self)

        main_layout.addWidget(self.plotter.interactor)

        self.plotter.set_background("white")
        self.plotter.add_axes()

    # --------------------------------------------------------
    # Physical size
    # --------------------------------------------------------

    def get_physical_size_um(self):

        z = self.z_size.value()
        y = self.y_size.value()
        x = self.x_size.value()

        xy = self.xy_sampling.value()
        z_step = self.z_sampling.value()

        size_x_um = x * xy
        size_y_um = y * xy
        size_z_um = z * z_step

        return size_x_um, size_y_um, size_z_um

    # --------------------------------------------------------
    # Update volume label
    # --------------------------------------------------------

    def update_volume_label(self):

        size_x_um, size_y_um, size_z_um = self.get_physical_size_um()

        self.volume_label.setText(
            "Calculated rendering volume: "
            f"X = {size_x_um:.3f} um | "
            f"Y = {size_y_um:.3f} um | "
            f"Z = {size_z_um:.3f} um"
        )

    # --------------------------------------------------------
    # Settings changed
    # --------------------------------------------------------

    def settings_changed(self):

        self.update_volume_label()

        # If ROI already exists, recreate it using the
        # new size while keeping the same centre.
        if self.roi_center_nm is not None:
            self.create_roi_at_center(self.roi_center_nm)

    # --------------------------------------------------------
    # Load mesh
    # --------------------------------------------------------

    def load_mesh(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mesh",
            "",
            "Mesh files (*.ply *.stl *.obj);;All files (*)",
        )

        if not file_path:
            return

        self.mesh = pv.read(file_path)

        self.roi_actor = None
        self.roi_bounds = None
        self.roi_center_nm = None

        self.plotter.clear()

        self.plotter.add_mesh(
            self.mesh,
            color="lightblue",
        )

        self.plotter.add_axes()
        self.plotter.reset_camera()

        self.roi_button.setEnabled(True)
        self.move_roi_button.setEnabled(False)

        self.roi_label.setText("ROI center: not selected")

    # --------------------------------------------------------
    # Add initial ROI
    # --------------------------------------------------------

    def add_roi_box(self):

        if self.mesh is None:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds

        center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        )

        self.create_roi_at_center(center)

        self.move_roi_button.setEnabled(True)

    # --------------------------------------------------------
    # Create fixed-size ROI
    # --------------------------------------------------------

    def create_roi_at_center(self, center):

        center_x, center_y, center_z = center

        size_x_um, size_y_um, size_z_um = self.get_physical_size_um()

        size_x_nm = size_x_um * NM_PER_UM
        size_y_nm = size_y_um * NM_PER_UM
        size_z_nm = size_z_um * NM_PER_UM

        bounds = (
            center_x - size_x_nm / 2.0,
            center_x + size_x_nm / 2.0,
            center_y - size_y_nm / 2.0,
            center_y + size_y_nm / 2.0,
            center_z - size_z_nm / 2.0,
            center_z + size_z_nm / 2.0,
        )

        # Remove previous ROI
        if self.roi_actor is not None:
            self.plotter.remove_actor(self.roi_actor)

        roi_box = pv.Box(bounds=bounds)

        self.roi_actor = self.plotter.add_mesh(
            roi_box,
            style="wireframe",
            line_width=3,
            color="black",
        )

        self.roi_center_nm = (
            center_x,
            center_y,
            center_z,
        )

        self.roi_bounds = bounds

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

        self.plotter.render()

    # --------------------------------------------------------
    # Enable ROI movement
    # --------------------------------------------------------

    def enable_roi_movement(self):

        if self.mesh is None:
            return

        self.roi_label.setText(
            "Click on the mesh to place the ROI centre."
        )

        self.plotter.enable_surface_point_picking(
            callback=self.move_roi_to_point,
            show_message=False,
            show_point=False,
            left_clicking=True,
            picker="cell",
        )

    # --------------------------------------------------------
    # Move ROI
    # --------------------------------------------------------

    def move_roi_to_point(self, point):

        if point is None:
            return

        center = (
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

        self.create_roi_at_center(center)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MeshViewer()
    window.show()

    sys.exit(app.exec())