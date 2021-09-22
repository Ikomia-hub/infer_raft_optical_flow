from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_raft_optical_flow.infer_raft_optical_flow_process import RaftOpticalFlowFactory
        # Instantiate process object
        return RaftOpticalFlowFactory()

    def getWidgetFactory(self):
        from infer_raft_optical_flow.infer_raft_optical_flow_widget import RaftOpticalFlowWidgetFactory
        # Instantiate associated widget object
        return RaftOpticalFlowWidgetFactory()
