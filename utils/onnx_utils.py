import numpy as np
import onnxruntime as ort
import PIL.Image as Image
import torchvision.transforms as T


class SemSegONNX():
    """Wrapper for semantic segmentation ONNX model.

    How to use:

        1. Initialize
        seg_model = SemSegONNX(path-to-onnx-file)

        2. Inference
        seg = seg_model(rgb)

    """

    def __init__(self, sem_onnx_path: str):
        """
        """
        self.ort_session_semseg = ort.InferenceSession(
            sem_onnx_path, providers=['CUDAExecutionProvider'])

        # Transformation for input image
        self.input_preproc = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def pred(self, rgb: Image) -> np.array:
        """Computes a semantic segmentation for input RGB.
        """
        rgb = self.input_preproc(rgb)
        rgb = rgb.unsqueeze(0)
        # ONNX Runtime output prediction
        ort_inputs = {
            self.ort_session_semseg.get_inputs()[0].name: self.to_numpy_(rgb)
        }
        ort_outs = self.ort_session_semseg.run(None, ort_inputs)
        seg = ort_outs[0]

        return seg

    @staticmethod
    def to_numpy_(tensor):
        return tensor.detach().cpu().numpy(
        ) if tensor.requires_grad else tensor.cpu().numpy()
