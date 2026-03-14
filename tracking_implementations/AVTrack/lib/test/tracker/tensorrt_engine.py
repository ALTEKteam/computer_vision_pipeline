import tensorrt as trt
import numpy as np
from cuda import cudart
import os

class AVTrackEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        if not self.engine:
            raise RuntimeError("TensorRT Engine yüklenemedi!")
            
        self.context = self.engine.create_execution_context()
        
        # O performans uyarısını çözen kısım: Özel CUDA Stream oluşturuyoruz
        _, self.stream = cudart.cudaStreamCreate()
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        self._allocate_buffers()

    def _load_engine(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            if shape[0] == -1:
                shape = (1,) + shape[1:]
                self.context.set_input_shape(tensor_name, shape)
                
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            host_mem = np.empty(shape, dtype)
            err, device_mem = cudart.cudaMalloc(size)
            
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA malloc hatası: {err}")
                
            self.bindings.append(device_mem)
            
            tensor_info = {'host': host_mem, 'device': device_mem, 'shape': shape, 'name': tensor_name, 'size': size}
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
            else:
                self.outputs.append(tensor_info)

    def infer(self, template_img, search_img):
        """
        template_img: (1, 3, 128, 128) boyutunda normalize edilmiş numpy array
        search_img:   (1, 3, 256, 256) boyutunda normalize edilmiş numpy array
        """
        # 1. Girdileri Host (CPU) belleğine kopyala
        np.copyto(self.inputs[0]['host'], template_img.ravel())
        np.copyto(self.inputs[1]['host'], search_img.ravel())
        
        # 2. Host'tan Device'a (GPU) ASENKRON kopyala (Özel stream kullanarak)
        for inp in self.inputs:
            cudart.cudaMemcpyAsync(
                inp['device'], inp['host'].ctypes.data, inp['size'], 
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
            )
            self.context.set_tensor_address(inp['name'], inp['device'])
            
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], out['device'])
            
        # 3. Model hesaplamasını ASENKRON çalıştır
        self.context.execute_async_v3(self.stream)
        
        # 4. Sonuçları GPU'dan CPU'ya geri çek
        for out in self.outputs:
            cudart.cudaMemcpyAsync(
                out['host'].ctypes.data, out['device'], out['size'], 
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream
            )
            
        # 5. İşlemlerin bitmesini bekle (Senkronizasyon)
        cudart.cudaStreamSynchronize(self.stream)
        
        # Bounding box genelde ilk çıktı olur
        bbox_output = self.outputs[0]['host']
        return bbox_output

    def __del__(self):
        # Program kapanırken GPU belleğinde sızıntı (memory leak) olmaması için temizlik
        if hasattr(self, 'stream'):
            cudart.cudaStreamDestroy(self.stream)
        if hasattr(self, 'bindings'):
            for binding in self.bindings:
                cudart.cudaFree(binding)