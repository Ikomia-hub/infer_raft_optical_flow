<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_raft_optical_flow/main/icon/RAFT.png" alt="Algorithm icon">
  <h1 align="center">infer_raft_optical_flow</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_raft_optical_flow">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_raft_optical_flow">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_raft_optical_flow/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_raft_optical_flow.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run RAFT optical flow algorithm. 

Estimate per-pixel motion between two consecutive frames with a RAFT model which is a composition of CNN and RNN. Models are trained with the Sintel dataset

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_raft_optical_flow/main/images/basket-result.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.core import IODataType
from ikomia.dataprocess import CImageIO
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2

# Init your workflow
wf = Workflow()

# Add RAFT optical flow algorithm
optical_flow = wf.add_task(name="infer_raft_optical_flow", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run algorithm on current frame
    # RAFT algorithm need at least 2 frames to give results
    optical_flow.set_input(CImageIO(IODataType.IMAGE, frame), 0)
    optical_flow.run()

    # Get and display results
    image_out = optical_flow.get_output(0)
    if image_out.is_data_available():
        img_res = (image_out.get_image()*255).astype('uint8')
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        display(img_res, title="RAFT", viewer="opencv")

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
# Init your workflow
wf = Workflow()

# Add RAFT optical flow algorithm
optical_flow = wf.add_task(name="infer_raft_optical_flow", auto_connect=True)

optical_flow.set_parameters({
    "small": "True",
    "cuda": "True",
})
```

- **small** (bool, default=True): True to use small model (faster), False to use large model (slower, better quality). 
- **cuda** (bool, default=True): CUDA acceleration if True, run on CPU otherwise.

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.core import IODataType
from ikomia.dataprocess import CImageIO
from ikomia.dataprocess.workflow import Workflow
import cv2

# Init your workflow
wf = Workflow()

# Add RAFT optical flow algorithm
optical_flow = wf.add_task(name="infer_raft_optical_flow", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run algorithm on current frame
    # RAFT algorithm need at least 2 frames to give results
    optical_flow.set_input(CImageIO(IODataType.IMAGE, frame), 0)
    optical_flow.run()

    # Iterate over outputs
    for output in optical_flow.get_outputs():
        # Print information
        print(output)
        # Export it to JSON
        output.to_json()

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()
```

RAFT algorithm generates 1 output:
- Optical flow image (CImageIO)