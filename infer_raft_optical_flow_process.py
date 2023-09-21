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
        self.small = True
        self.cuda = True
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.small = utils.strtobool(param_map["small"])
        new_cuda = utils.strtobool(param_map["cuda"]) and torch.cuda.is_available()
        if new_cuda != utils.strtobool(param_map["cuda"]):
            self.update = True
        self.cuda = utils.strtobool(param_map["cuda"]) and torch.cuda.is_available()

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "small": str(self.small),
            "cuda": str(self.cuda)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class RaftOpticalFlow(dataprocess.CVideoTask):

    def __init__(self, name, param):
        dataprocess.CVideoTask.__init__(self, name)
        self.model = None
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

    def notify_video_start(self, frame_count):
        # frame_1 is reset to avoid optical flow calculation between 2 images of different videos
        self.frame_1 = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def get_cpu_model(self, model):
        new_model = OrderedDict()
        # get all layer's names from model
        for name in model:
            # create new name and update new model
            new_name = name[7:]
            new_model[new_name] = model[name]

        return new_model

    def load_model(self, small, device, dropout=0, mixed_precision=True, alternate_corr=False):
        param = self.get_param_object()
        if self.model is None or param.update is True:
            print("Loading pre-trained model...")
            # get the RAFT model
            self.model = RAFT(small, dropout, mixed_precision, alternate_corr)
            # load pretrained weights
            if small:
                model_name = "raft-small.pth"
            else:
                model_name = "raft-sintel.pth"

            model_weight_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", model_name)
            if not os.path.exists(model_weight_file):
                print(f"Downloading model {model_name}, please wait...")
                self.download(f"{utils.get_model_hub_url()}/{self.name}/{model_name}", model_weight_file)

            pretrained_weights = torch.load(model_weight_file)

            if device == "cuda":
                # parallel between available GPUs
                self.model = torch.nn.DataParallel(self.model)
                # load the pretrained weights into model
                self.model.load_state_dict(pretrained_weights)
                self.model.to(device)
            elif device == "cpu":
                # change key names for CPU runtime
                pretrained_weights = self.get_cpu_model(pretrained_weights)
                # load the pretrained weights into model
                self.model.load_state_dict(pretrained_weights)

            # change model's mode to evaluation
            self.model.eval()
            param.update = False

    def frame_preprocess(self, frame, device):
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame = frame.unsqueeze(0)
        frame = frame.to(device)
        return frame

    def visualize_flow(self, flow):
        # permute the channels and change device is necessary
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        # map flow to rgb image
        flow = flow_viz.flow_to_image(flow)
        flow = cv2.cvtColor(flow, cv2.COLOR_RGB2BGR)
        return flow / 255.0

    def flow_from_images(self, device, frame_1, frame_2):
        # frame preprocessing
        frame_1 = self.frame_preprocess(frame_1, device)
        frame_2 = self.frame_preprocess(frame_2, device)

        with torch.no_grad():
            # predict the flow
            flow_low, flow_up = self.model(frame_1, frame_2, iters=20, test_mode=True)

        return flow_up

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        device = "cuda" if param.cuda else "cpu"
        self.load_model(param.small, device)

        # Get input :
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # Test for correct input shape
        w, h, c = np.shape(src_image)
        bad_dimensions = w % 8 != 0 or h % 8 != 0

        if bad_dimensions:
            src_image = cv2.resize(src_image, dsize=(w//8*8, h//8*8))

        # Get output :
        img_output = self.get_output(0)
        flow_output = self.get_output(1)

        if self.frame_1 is not None:
            frame_2 = self.frame_1
            frame_1 = src_image
            flow = self.flow_from_images(device, frame_1, frame_2)
            img_flow = self.visualize_flow(flow)

            # Set image of input/output (numpy array):
            img_output.set_image(img_flow)

            if self.rawOutput:
                flow = flow.cpu().numpy()
                flow_output.set_image(flow[0])
        else:
            self.frame_1 = src_image
            flow = self.flow_from_images(device, self.frame_1, self.frame_1)
            img_flow = self.visualize_flow(flow)

            # Set image of input/output (numpy array):
            img_output.set_image(img_flow)

            if self.rawOutput:
                flow = flow.cpu().numpy()
                flow_output.set_image(flow[0])

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
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Optical Flow"
        self.info.version = "1.2.0"
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
