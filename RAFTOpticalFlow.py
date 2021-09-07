from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class RAFTOpticalFlow(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from RAFTOpticalFlow.RAFTOpticalFlow_process import RAFTOpticalFlowProcessFactory
        # Instantiate process object
        return RAFTOpticalFlowProcessFactory()

    def getWidgetFactory(self):
        from RAFTOpticalFlow.RAFTOpticalFlow_widget import RAFTOpticalFlowWidgetFactory
        # Instantiate associated widget object
        return RAFTOpticalFlowWidgetFactory()
