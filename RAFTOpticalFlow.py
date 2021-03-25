from ikomia import dataprocess
import RAFTOpticalFlow_process as processMod
import RAFTOpticalFlow_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class RAFTOpticalFlow(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.RAFTOpticalFlowProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.RAFTOpticalFlowWidgetFactory()
