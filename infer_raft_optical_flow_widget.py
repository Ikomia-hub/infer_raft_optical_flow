from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion
from infer_raft_optical_flow.infer_raft_optical_flow_process import RaftOpticalFlowParam, RaftOpticalFlow
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class RaftOpticalFlowWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = RaftOpticalFlowParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # cuda parameter
        cuda_label = QLabel("CUDA")
        self.cuda_ckeck = QCheckBox()

        if self.parameters.cuda:
            self.cuda_ckeck.setChecked(True)

        if not self.parameters.cuda:
            self.cuda_ckeck.setChecked(False)
            self.cuda_ckeck.setEnabled(False)

        # model size parameter
        self.rbtn1 = QRadioButton("Small-sized Optical Flow")
        self.rbtn1.setChecked(True)

        self.rbtn2 = QRadioButton("Normal-sized Optical Flow")

        self.gridLayout.addWidget(self.cuda_ckeck, 0, 0)
        self.gridLayout.addWidget(cuda_label, 0, 1)
        self.gridLayout.addWidget(self.rbtn1, 1, 0)
        self.gridLayout.addWidget(self.rbtn2, 1, 1)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        if self.cuda_ckeck.isChecked():
            self.parameters.device = "cuda"
        else:
            self.parameters.device = "cpu"
        self.parameters.small=self.rbtn1.isChecked()

        self.parameters.model = RaftOpticalFlow.trained_model(self.parameters.small, self.parameters.device)
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class RaftOpticalFlowWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_raft_optical_flow"

    def create(self, param):
        # Create widget object
        return RaftOpticalFlowWidget(param, None)
