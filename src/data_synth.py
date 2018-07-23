import numpy as np
import ctypes as c
import cv2
import time

def get_lib():
    lib = c.cdll.LoadLibrary('./maptextsynth_deps/libmtsi.so')

    # get_sample takes no args, returns void*
    lib.get_sample.argtypes = []
    lib.get_sample.restype = c.c_void_p 
    
    # free_sample takes void*, returns nothing
    lib.free_sample.argtypes = [c.c_void_p]
    lib.free_sample.restype = None
    
    # get_caption takes void*, returns char* (null terminated)
    lib.get_caption.argtypes = [c.c_void_p]
    lib.get_caption.restype = c.c_char_p 

    # get_height takes void*, returns size_t (null terminated)
    lib.get_height.argtypes = [c.c_void_p]
    lib.get_height.restype = c.c_ulonglong 

    # get_width takes void*, returns size_t (null terminated)
    lib.get_width.argtypes = [c.c_void_p]
    lib.get_width.restype = c.c_ulonglong 
    
    # get_img_data takes void*, returns void* (really an unsigned char* array of size width*height) 
    lib.get_img_data.argtypes = [c.c_void_p]
    lib.get_img_data.restype = c.c_void_p

    # in: void, out: void
    lib.mts_init.argtypes = []
    lib.mts_init.restype = None

    # in: void, out: void
    lib.mts_cleanup.argtypes = []
    lib.mts_cleanup.restype = None

    return lib


def data_generator():
    # For c array -> numpy conversion
    buffer_from_memory = c.pythonapi.PyBuffer_FromMemory
    buffer_from_memory.restype = c.py_object
    
    lib = get_lib()
    lib.mts_init()
    while True:
        ptr = lib.get_sample()
        if not ptr:
            print "No sample produced."
            exit()
        height = lib.get_height(ptr)
        raw_data = lib.get_img_data(ptr)
        caption = lib.get_caption(ptr)
        width = lib.get_width(ptr)
        raw_data_ptr = c.cast(raw_data, c.POINTER(c.c_ubyte))
        # https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
        # Above link used as reference for c array -> numpy conversion
        buffer = buffer_from_memory(raw_data_ptr, width*height)
        img_flat = np.frombuffer(buffer, np.uint8)

        img_shaped = np.reshape(img_flat, (height, width, 1))
        
        yield caption, img_shaped
        lib.free_sample(ptr)
    # Note: I don't know how to get this called (mts_cleanup)
    lib.mts_cleanup()

def gather_data(num_values):
    # Note: this is like this because data_generator used to take a lib as arg
    lib = get_lib()
    lib.mts_init()
    iter = data_generator() 
    for _ in range(num_values):
        caption, image = next(iter)
        cv2.imshow("windows", image)
        cv2.waitKey(0)
    lib.mts_cleanup()
