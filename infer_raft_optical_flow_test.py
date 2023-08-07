import logging
from ikomia.utils.tests import run_for_test
import cv2

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer raft optical flow =====")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[:, :, ::-1]
    input_img_0 = t.get_input(0)
    input_img_0.set_image(img)
    params = t.get_parameters()
    # run once to set frame 1
    run_for_test(t)
    for small in [True, False]:
        params["small"] = small
        t.set_parameters(params)
        yield run_for_test(t)
