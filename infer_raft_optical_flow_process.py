from ikomia import utils, core, dataprocess
import copy
import torch
import numpy as np
import cv2
from infer_raft_optical_flow.core.raft import RAFT
from infer_raft_optical_flow.core.utils import flow_viz
from collections import OrderedDict
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class RaftOpticalFlowParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.small = True
        self.cuda = True if torch.cuda.is_available() else False
        self.cuda = "cuda" if self.cuda else "cpu"
        self.model = None

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.small = utils.strtobool(param_map["small"])
        self.cuda = param_map["cuda"]
        self.model = None

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["small"] = str(self.small)
        param_map["cuda"] = str(self.cuda)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class RaftOpticalFlow(dataprocess.CVideoTask):

    def __init__(self, name, param):
        dataprocess.CVideoTask.__init__(self, name)
        # Add input/output of the process here
        # Set this variable to True if you want to work with the raw Optical Flow (vector field)
        self.rawOutput = False
        if self.rawOutput:
            self.add_output(dataprocess.CImageIO())

        self.frame_1 = None
        # Create parameters class
        if param is None:
            self.set_param_object(RaftOpticalFlowParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def notifyVideoStart(self, frame_count):
        # frame_1 is reset to avoid optical flow calculation between 2 images of different videos
        self.frame_1 = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def get_cpu_model(model):
        new_model = OrderedDict()
        # get all layer's names from model
        for name in model:
            # create new name and update new model
            new_name = name[7:]
            new_model[new_name] = model[name]
        return new_model

    def trained_model(small, device, dropout=0, mixed_precision=True, alternate_corr=False):
        print("Loading pre-trained model...")
        # get the RAFT model
        model = RAFT(small, dropout, mixed_precision, alternate_corr)
        # load pretrained weights
        if small:
            pretrained_weights = torch.load(os.path.dirname(os.path.realpath(__file__)) + "/models/raft-small.pth")
        else:
            pretrained_weights = torch.load(os.path.dirname(os.path.realpath(__file__)) + "/models/raft-sintel.pth")

        if device == "cuda":
            # parallel between available GPUs
            model = torch.nn.DataParallel(model)
            # load the pretrained weights into model
            model.load_state_dict(pretrained_weights)
            model.to(device)
        else:
            if device == "cpu":
                # change key names for CPU runtime
                pretrained_weights = RaftOpticalFlow.get_cpu_model(pretrained_weights)
                # load the pretrained weights into model
                model.load_state_dict(pretrained_weights)

        # change model's mode to evaluation
        model.eval()
        return model

    def frame_preprocess(frame, device):
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame = frame.unsqueeze(0)
        frame = frame.to(device)
        return frame

    def vizualize_flow(flo):
        # permute the channels and change device is necessary
        flo = flo[0].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
        return flo / 255.0

    def flow_from_images(model, device, frame_1, frame_2):
        # frame preprocessing
        frame_1 = RaftOpticalFlow.frame_preprocess(frame_1, device)
        frame_2 = RaftOpticalFlow.frame_preprocess(frame_2, device)
        with torch.no_grad():
            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=20, test_mode=True)
            # transpose the flow output and convert it into numpy array
        return flow_up

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization

        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        if not param.model:
            param.model = RaftOpticalFlow.trained_model(param.small, param.cuda)

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        srcImage = input.get_image()

        # Test for correct input shape
        w, h, c = np.shape(srcImage)
        badDimensions = w % 8 != 0 or h % 8 != 0
        if badDimensions:
            srcImage = cv2.resize(srcImage, dsize=(w//8*8, h//8*8))

        # Get output :
        output = self.get_output(0)
        outputFlow = self.get_output(1)

        if self.frame_1 is not None:
            frame_2 = self.frame_1
            frame_1 = srcImage
            flow = RaftOpticalFlow.flow_from_images(param.model, param.cuda, frame_1, frame_2)
            img_flo = RaftOpticalFlow.vizualize_flow(flow)

            # Set image of input/output (numpy array):
            output.set_image(img_flo)

            if self.rawOutput:
                flow = flow.cpu().numpy()
                outputFlow.set_image(flow[0])
        else:
            self.frame_1 = srcImage

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class RaftOpticalFlowFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_raft_optical_flow"
        self.info.short_description = "Estimate the optical flow from a video using a RAFT model."
        self.info.description = "Estimate per-pixel motion between two consecutive frames " \
                                "with a RAFT model which is a composition of CNN and RNN." \
                                "Models are trained with the Sintel dataset"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Optical Flow"
        self.info.version = "1.1.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Zachary Teed and Jia Deng"
        self.info.article = "RAFT: Recurrent All Pairs Field Transforms for Optical Flow"
        self.info.journal = "ECCV"
        self.info.year = 2020
        self.info.license = "BSD 3-Clause License"
        # URL of documentation
        self.info.documentation_link = "https://learnopencv.com/optical-flow-using-deep-learning-raft/"
        # Code source repository
        self.info.repository = "https://github.com/princeton-vl/RAFT"
        # Keywords used for search
        self.info.keywords = "optical,flow,RAFT,CNN,RNN"
        self.info.icon_path = "icon/RAFT.png"

    def create(self, param=None):
        # Create process object
        return RaftOpticalFlow(self.info.name, param)
