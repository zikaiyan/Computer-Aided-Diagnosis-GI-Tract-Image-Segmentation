import os
import glob
import numpy as np
import itertools

def get_scan_file_path(base_dir, scan_id):
    """
    Helper function to derive the 
    full file path for a given scan ID
    """
    id_parts = scan_id.split("_")
    case_part = id_parts[0]
    day_part = "_".join(id_parts[:2])
    scan_prefix = "_".join(id_parts[2:])
    scan_directory = os.path.join(base_dir, case_part, day_part, "scans")
    matching_files = glob.glob(f"{scan_directory}/{scan_prefix}*")  # Expecting a single match
    return matching_files[0]



def decode_rle(mask_rle, shape):
    '''
    Decode an RLE-encoded mask into a 2D numpy array.

    Parameters:
        mask_rle (str): Run-length encoding of the mask ('start length' pairs)
        shape (tuple): The (height, width) dimensions of the output array

    Returns:
        numpy.ndarray: A 2D array where 1s represent the mask and 0s represent the background
    '''
    # Split the RLE string into a list of strings
    encoded_pixels = mask_rle.split()
    
    # Extract start positions and lengths for the mask
    starts = np.asarray(encoded_pixels[0::2], dtype=int) - 1  # Convert to 0-based indexing
    lengths = np.asarray(encoded_pixels[1::2], dtype=int)
    
    # Calculate the end positions for each segment of the mask
    ends = starts + lengths
    
    # Initialize a flat array for the image
    flat_image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    # Apply the mask to the flat image array
    for start, end in zip(starts, ends):
        flat_image[start:end] = 1
    
    # Reshape the flat image array to 2D
    return flat_image.reshape(shape)



def convert_binary_mask_to_rle(binary_mask):
    '''
    Convert a binary mask to run-length encoding.

    Parameters:
        binary_mask (numpy.ndarray): A 2D array where 1s represent the mask and 0s the background.

    Returns:
        dict: A dictionary with two keys: 'counts' and 'size', where 'counts' holds the RLE and 'size' the dimensions of the mask.
    '''
    # Initialize the RLE dictionary with the size of the mask
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    
    # Prepare to fill the counts list in the RLE dictionary
    counts = rle['counts']
    
    # Flatten the array column-wise and group by value
    flattened = binary_mask.ravel(order='F')
    grouped = itertools.groupby(flattened)
    
    # Iterate through grouped data to form the RLE
    for i, (pixel_value, group_iter) in enumerate(grouped):
        # Convert group iterator to list to count elements
        group_length = len(list(group_iter))
        
        # If first group is mask, prepend 0
        if i == 0 and pixel_value == 1:
            counts.append(0)
        
        # Append the length of each group to the RLE counts
        counts.append(group_length)
    
    return rle
