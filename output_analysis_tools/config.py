from  typing import Dict, Tuple, TypeVar
B:TypeVar = TypeVar("B", Tuple,str)
config:Dict[str, B] = {
   # size of the input images
    'input_shape_scene': (10, 120, 160, 1),
    'disparity_maps_s': (10,120,160,7),
    'disparity_maps_56':  (10,120,160,56), #(5,56,120,160),
    'disparity_maps': (10,7,8,120,160),
    'input_shape_binocular': (10, 2, 120,160, 1),

    # Paths to the datasets
    'path_labels': "/scratch/gucr/tEDRAM2/training_data/label_data.txt",

    'scene_image': "/scratch/gucr/tEDRAM2/training_data/training_data.h5",
    'disparity_maps_sum': "/scratch/gucr/tEDRAM2/training_data/disparity_maps_s.h5",
    'disparity_map_arrays_56': "/scratch/gucr/tEDRAM2/training_data/disparity_maps_56imgs.h5",
    'disparity_map_arrays': "/scratch/gucr/tEDRAM2/training_data/disparity_maps.h5",
    'input_binocular_arrays': "/scratch/gucr/tEDRAM2/training_data/binocular_image_data.h5",

}
datasets = [config['path_labels'],
            config['scene_image'],
            config['disparity_maps_sum'],
            config['disparity_map_arrays_56'],
            config['disparity_map_arrays'],
            config['input_binocular_arrays']
        ]