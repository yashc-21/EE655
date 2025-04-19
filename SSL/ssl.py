
import tensorflow as tf

class SelfSimilarityLoss:
    def __init__(self, patch_size=4, threshold=0.2):
        self.patch_size = patch_size
        self.threshold = threshold

    def compute_patches(self, images):
        """
        Extract patches from images.
        """
        patches = tf.image.extract_patches(images, 
                                            sizes=[1, self.patch_size, self.patch_size, 1],
                                            strides=[1, self.patch_size, self.patch_size, 1],
                                            rates=[1, 1, 1, 1],
                                            padding='VALID')
        return patches

    def similarity_map(self, images):
        """
        Compute the self-similarity map for a batch of images.
        """
        patches = self.compute_patches(images)
        patches = tf.reshape(patches, [tf.shape(images)[0], -1, self.patch_size * self.patch_size, 3])  # Flatten patches

        # Compute pairwise similarity (inner product)
        similarity_matrix = tf.linalg.matmul(patches, patches, transpose_b=True)  # Shape: (batch_size, num_patches, num_patches)
        return similarity_matrix

    def get_loss(self, output_images, gt_images):
        """
        Compute the self-similarity loss between output and ground truth images.
        """
        # Compute similarity maps for output and ground truth images
        similarity_output = self.similarity_map(output_images)
        similarity_gt = self.similarity_map(gt_images)
        
        # Create masks based on the threshold
        mask_output = tf.greater(similarity_output, self.threshold)
        mask_gt = tf.greater(similarity_gt, self.threshold)
        
        # Apply masks to get masked similarity maps
        masked_similarity_output = tf.where(mask_output, similarity_output, tf.zeros_like(similarity_output))
        masked_similarity_gt = tf.where(mask_gt, similarity_gt, tf.zeros_like(similarity_gt))

        # Calculate the difference between the masked similarity maps of output and ground truth
        ssl_loss = tf.reduce_mean(tf.abs(masked_similarity_output - masked_similarity_gt))
        return ssl_loss

# Example usage
def ssl_loss_cal(output, gt):
    ssl = SelfSimilarityLoss()
    return ssl.get_loss(output, gt)

# import numpy as np
# import cv2  # OpenCV for Laplacian operation and image processing

# def compute_edge_map(image, threshold=20):
#     # Convert to grayscale
#     gray_image = tf.image.rgb_to_grayscale(image)
    
#     # Compute edges using Sobel filter (alternative to Laplacian in OpenCV)
#     sobel_edges = tf.image.sobel_edges(gray_image)
#     edge_magnitude = tf.sqrt(tf.reduce_sum(tf.square(sobel_edges), axis=-1))
    
#     # Threshold the edge map to create a binary mask
#     mask = tf.cast(edge_magnitude > threshold, tf.float32)
#     return mask

# def compute_similarity_map(gt_image, isr_image, patch_radius, search_radius, scale_factor, mask):
#     """
#     Computes the similarity map between ground truth (GT) and ISR image patches only for edge areas.
#     """
#     # H, W, C = gt_image.shape
#     # Unpack based on expected dimensions
#     if len(gt_image.shape) == 4:
#         _, H, W, C = gt_image.shape  # Ignore batch dimension
#     elif len(gt_image.shape) == 3:
#         H, W, C = gt_image.shape     # No batch dimension
#     else:
#         raise ValueError("gt_image has an unexpected shape: {}".format(gt_image.shape))

#     similarity_map = np.zeros((H, W))
#     epsilon = 1e-8  # to prevent division by zero
    
#     # Define patch and search window dimensions
#     patch_diameter = 2 * patch_radius + 1
#     search_diameter = 2 * search_radius + 1
    
#     for i in range(patch_radius, H - patch_radius):
#         for j in range(patch_radius, W - patch_radius):
#             if i < mask.shape[0] and j < mask.shape[1]: 
#                 if mask[i, j] == 1:  # Only compute for edge pixels
#                     # Extract GT and ISR patches centered at (i, j)
#                     gt_patch = gt_image[i-patch_radius:i+patch_radius+1, j-patch_radius:j+patch_radius+1, :]
#                     isr_patch = isr_image[i-patch_radius:i+patch_radius+1, j-patch_radius:j+patch_radius+1, :]

#                     # Calculate self-similarity for each pixel in search area
#                     for m in range(-search_radius, search_radius + 1):
#                         for n in range(-search_radius, search_radius + 1):
#                             if 0 <= i + m < H and 0 <= j + n < W:
#                                 # Extract search patches in GT and ISR
#                                 gt_search_patch = gt_image[i+m-patch_radius:i+m+patch_radius+1, j+n-patch_radius:j+n+patch_radius+1, :]
#                                 isr_search_patch = isr_image[i+m-patch_radius:i+m+patch_radius+1, j+n-patch_radius:j+n+patch_radius+1, :]
                                
#                                 # Compute Euclidean distance (squared)
#                                 dist_squared = np.sum((gt_patch - gt_search_patch)**2) / (C * patch_diameter**2)
                                
#                                 # Similarity calculation
#                                 similarity = np.exp(-dist_squared / scale_factor)
#                                 similarity_map[i, j] += similarity

#                     # Normalize the similarity map
#                     similarity_map[i, j] /= (search_diameter**2 + epsilon)
    
#     return similarity_map

# def get_loss(gt_image, isr_image, patch_radius=2, search_radius=3, scale_factor=0.5, threshold=20, alpha=1):
#     """
#     Computes Self-Similarity Loss (SSL) between GT and ISR images.
#     """
#     # Generate edge mask

#     isr_image= tf.cast(isr_image, tf.float32)
#     gt_image = tf.cast(gt_image, tf.float32)
#     mask = compute_edge_map(gt_image, threshold)

#     # Compute masked similarity maps for GT and ISR images
#     similarity_map_gt = compute_similarity_map(gt_image, gt_image, patch_radius, search_radius, scale_factor, mask)
#     similarity_map_isr = compute_similarity_map(gt_image, isr_image, patch_radius, search_radius, scale_factor, mask)

#     # Compute KL divergence and L1 distance as SSL loss
#     kl_div = np.sum(similarity_map_gt * np.log((similarity_map_gt + 1e-8) / (similarity_map_isr + 1e-8)))
#     l1_distance = np.sum(np.abs(similarity_map_gt - similarity_map_isr))

#     ssl_loss_value = kl_div + alpha * l1_distance

#     return ssl_loss_value



# Example usage
# def ssl_loss_cal(output, gt):

#     return SelfSimilarityLoss.get_loss(output, gt)
