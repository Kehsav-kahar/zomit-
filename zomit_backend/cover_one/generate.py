import os
import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from datetime import datetime
from .models import cover_one
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Helper function to adjust brightness of the user image
def adjust_brightness(user_image_cv, factor=1.2):
    hsv = cv2.cvtColor(user_image_cv, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)  # Enhance the brightness
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image





# def replace_green_area(template_image_path, user_image_cv):
#     try:
#         # Read the cover template using OpenCV
#         template_image_cv = cv2.imread(template_image_path)
#         if template_image_cv is None:
#             raise Exception('Error reading template image')

#         # Convert the template to HSV to detect green areas
#         hsv_template = cv2.cvtColor(template_image_cv, cv2.COLOR_BGR2HSV)

#         # Define the green color range (you can fine-tune these values)
#         lower_green = np.array([35, 100, 100])  # Lower boundary of green in HSV
#         upper_green = np.array([85, 255, 255])  # Upper boundary of green in HSV
#         mask = cv2.inRange(hsv_template, lower_green, upper_green)

#         # Get the bounding box of the green area
#         green_coords = np.column_stack(np.where(mask > 0))
#         top_left = green_coords.min(axis=0)
#         bottom_right = green_coords.max(axis=0)
#         green_box_h = bottom_right[0] - top_left[0]
#         green_box_w = bottom_right[1] - top_left[1]

#         # Invert the mask to get the non-green areas of the template
#         mask_inv = cv2.bitwise_not(mask)

#         # Black-out the green area in the template
#         template_bg = cv2.bitwise_and(template_image_cv, template_image_cv, mask=mask_inv)

#         # Get the size of the user's image
#         user_h, user_w = user_image_cv.shape[:2]

#         # Calculate the scale factor to fit the user's image into the green bounding box without cropping
#         scale_factor = min(green_box_w / user_w, green_box_h / user_h)

#         # Calculate the new size of the user's image
#         new_user_w = int(user_w * scale_factor)
#         new_user_h = int(user_h * scale_factor)

#         # Resize the user's image
#         user_image_resized = cv2.resize(user_image_cv, (new_user_w, new_user_h))

#         # Adjust the brightness of the resized user image
#         user_image_resized = adjust_brightness(user_image_resized, factor=1.2)

#         # Create a blank image (same size as template) to center the resized user image
#         user_image_centered = np.zeros_like(template_image_cv)

#         # Calculate the offsets to place the resized user's image within the green box
#         x_offset = top_left[1] + (green_box_w - new_user_w) // 2
#         y_offset = top_left[0] + (green_box_h - new_user_h) // 2

#         # Place the resized user's image in the center of the green box
#         user_image_centered[y_offset:y_offset + new_user_h, x_offset:x_offset + new_user_w] = user_image_resized

#         # Take only the region of the user's image where the green was in the template
#         user_fg = cv2.bitwise_and(user_image_centered, user_image_centered, mask=mask)

#         # Combine the two images using cv2.add to preserve the colors better
#         final_image = cv2.add(template_bg, user_fg)

#         return final_image

#     except Exception as e:
#         logger.error(f"Error in replace_green_area: {str(e)}")
#         raise







# Helper function to replace the green part of the template with the user's image without cropping
def replace_green_area(template_image_path, user_image_cv):
    try:
        # Read the cover template using OpenCV
        template_image_cv = cv2.imread(template_image_path)

        if template_image_cv is None:
            raise Exception('Error reading template image')

        # Convert the template to HSV to detect green areas
        hsv_template = cv2.cvtColor(template_image_cv, cv2.COLOR_BGR2HSV)

        # Define the green color range
        lower_green = np.array([35, 100, 100])  # Lower boundary of green in HSV
        upper_green = np.array([85, 255, 255])  # Upper boundary of green in HSV
        mask = cv2.inRange(hsv_template, lower_green, upper_green)

        # Invert the mask to get the non-green areas of the template
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the green area in the template
        template_bg = cv2.bitwise_and(template_image_cv, template_image_cv, mask=mask_inv)

        # Get the size of the template
        template_h, template_w = template_image_cv.shape[:2]

        # Get the size of the user's image
        user_h, user_w = user_image_cv.shape[:2]

        # Calculate the scale factors separately for width and height
        scale_factor_w = template_w / user_w
        scale_factor_h = template_h / user_h

        # Use the smaller scale factor to fit the image within the template without cropping
        scale_factor = min(scale_factor_w, scale_factor_h)

        # Calculate the new size of the user's image
        new_user_w = int(user_w * scale_factor)
        new_user_h = int(user_h * scale_factor)

        # Resize the user's image
        user_image_resized = cv2.resize(user_image_cv, (new_user_w, new_user_h))

        # Adjust the brightness of the resized user image (optional)
        user_image_resized = adjust_brightness(user_image_resized, factor=1.2)

        # Create a blank image (same size as the template) to center the resized user image
        user_image_centered = np.zeros_like(template_image_cv)

        # Calculate the offsets to center the user's image within the template
        x_offset = (template_w - new_user_w) // 2
        y_offset = (template_h - new_user_h) // 2

        # Place the resized user's image in the center of the blank image
        user_image_centered[y_offset:y_offset + new_user_h, x_offset:x_offset + new_user_w] = user_image_resized

        # Take only the region of the user's image where the green was in the template
        user_fg = cv2.bitwise_and(user_image_centered, user_image_centered, mask=mask)

        # Combine the two images using cv2.add to preserve the colors better
        final_image = cv2.add(template_bg, user_fg)

        return final_image

    except Exception as e:
        logger.error(f"Error in replace_green_area: {str(e)}")
        raise



# API for uploading a user image and generating the mobile cover
class GenerateCoverView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        model_name = request.data.get('cover_model')
        user_image = request.FILES.get('user_image')

        if not model_name or not user_image:
            return JsonResponse({'error': 'Model name and image are required'}, status=400)

        try:
            # Fetch the cover template from the database based on the model name
            cover_instance = get_object_or_404(cover_one, cover_model=model_name)

            # Path to the template
            cover_template_path = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'cover_templates', cover_instance.cover_template)

            if not os.path.exists(cover_template_path):
                return JsonResponse({'error': 'Cover template not found'}, status=404)

            # Save user image temporarily
            uploads_dir = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            user_image_path = os.path.join(uploads_dir, user_image.name)

            with open(user_image_path, 'wb+') as f:
                for chunk in user_image.chunks():
                    f.write(chunk)

            # Read user image
            user_image_cv = cv2.imread(user_image_path)
            if user_image_cv is None:
                return JsonResponse({'error': 'Error reading user image'}, status=500)

            # Replace green area
            final_image = replace_green_area(cover_template_path, user_image_cv)

            # Save the final image
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            generated_image_name = f"{model_name}_{timestamp}.png"

            # Save image with high quality (set compression to 0 for no loss)
            generated_covers_dir = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'generated_covers')
            os.makedirs(generated_covers_dir, exist_ok=True)
            generated_image_path = os.path.join(generated_covers_dir, generated_image_name)
            cv2.imwrite(generated_image_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Set to 0 for no compression

            # Clean up temporary uploaded image
            os.remove(user_image_path)

        except Exception as e:
            logger.error(f"Error in GenerateCoverView: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

        # Return the relative URL for React frontend
        generated_image_url = f'http://192.168.1.7:8000/cover_one/static/generated_covers/{generated_image_name}'

        return JsonResponse({
            'message': 'Cover generated successfully',
            'generated_image_url': generated_image_url,
            'alt': model_name
        }, status=200)

# API to get all generated mobile covers
class GetAllGeneratedCoversView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            # Directory where generated covers are stored
            generated_covers_dir = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'generated_covers')

            # Check if the directory exists
            if not os.path.exists(generated_covers_dir):
                return JsonResponse({'error': 'Generated covers directory not found'}, status=404)

            # List all images in the directory
            generated_images = []
            for image_file in os.listdir(generated_covers_dir):
                # Check for image file extensions
                if image_file.endswith((".png", ".jpg", ".jpeg")):
                    image_url = f'http://192.168.1.7:8000/cover_one/static/generated_covers/{image_file}'
                    generated_images.append(image_url)

            # Return the list of generated images without a message
            return JsonResponse(generated_images, safe=False, status=200)

        except Exception as e:
            logger.error(f"Error in GetAllGeneratedCoversView: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
        



# import os
# import cv2
# import numpy as np
# from django.conf import settings
# from django.http import JsonResponse
# from django.shortcuts import get_object_or_404
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from datetime import datetime
# from .models import cover_one
# import logging

# # Set up logger
# logger = logging.getLogger(__name__)

# # Helper function to adjust brightness of the user image
# def adjust_brightness(user_image_cv, factor=1.2):
#     hsv = cv2.cvtColor(user_image_cv, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)  # Enhance the brightness
#     brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return brightened_image

# # Helper function to replace the green part of the template with the user's image
# def replace_green_area(template_image_path, user_image_cv):
#     try:
#         # Read the cover template using OpenCV
#         template_image_cv = cv2.imread(template_image_path)

#         if template_image_cv is None:
#             raise Exception('Error reading template image')

#         # Convert the template to HSV to detect green areas
#         hsv_template = cv2.cvtColor(template_image_cv, cv2.COLOR_BGR2HSV)

#         # Define the green color range
#         lower_green = np.array([35, 100, 100])  # Lower boundary of green in HSV
#         upper_green = np.array([85, 255, 255])  # Upper boundary of green in HSV
#         mask = cv2.inRange(hsv_template, lower_green, upper_green)

#         # Invert the mask to get the non-green areas of the template
#         mask_inv = cv2.bitwise_not(mask)

#         # Black-out the green area in the template
#         template_bg = cv2.bitwise_and(template_image_cv, template_image_cv, mask=mask_inv)

#         # Resize the user's image to fit the template and adjust brightness
#         user_image_resized = cv2.resize(user_image_cv, (template_image_cv.shape[1], template_image_cv.shape[0]))
#         user_image_resized = adjust_brightness(user_image_resized, factor=1.2)  # Adjust brightness

#         # Take only the region of the user's image where the green was in the template
#         user_fg = cv2.bitwise_and(user_image_resized, user_image_resized, mask=mask)

#         # Combine the two images using cv2.add to preserve the colors better
#         final_image = cv2.add(template_bg, user_fg)

#         return final_image

#     except Exception as e:
#         logger.error(f"Error in replace_green_area: {str(e)}")
#         raise

# # API for uploading a user image and generating the mobile cover
# class GenerateCoverView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         model_name = request.data.get('cover_model')
#         user_image = request.FILES.get('user_image')

#         if not model_name or not user_image:
#             return JsonResponse({'error': 'Model name and image are required'}, status=400)

#         try:
#             # Fetch the cover template from the database based on the model name
#             cover_instance = get_object_or_404(cover_one, cover_model=model_name)

#             # Path to the template
#             cover_template_path = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'cover_templates', cover_instance.cover_template)

#             if not os.path.exists(cover_template_path):
#                 return JsonResponse({'error': 'Cover template not found'}, status=404)

#             # Save user image temporarily
#             uploads_dir = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'uploads')
#             os.makedirs(uploads_dir, exist_ok=True)
#             user_image_path = os.path.join(uploads_dir, user_image.name)

#             with open(user_image_path, 'wb+') as f:
#                 for chunk in user_image.chunks():
#                     f.write(chunk)

#             # Read user image
#             user_image_cv = cv2.imread(user_image_path)
#             if user_image_cv is None:
#                 return JsonResponse({'error': 'Error reading user image'}, status=500)

#             # Replace green area
#             final_image = replace_green_area(cover_template_path, user_image_cv)

#             # Save the final image
#             timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#             generated_image_name = f"{model_name}_{timestamp}.png"

#             # Save image with high quality (set compression to 0 for no loss)
#             generated_covers_dir = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'generated_covers')
#             os.makedirs(generated_covers_dir, exist_ok=True)
#             generated_image_path = os.path.join(generated_covers_dir, generated_image_name)
#             cv2.imwrite(generated_image_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Set to 0 for no compression

#             # Clean up temporary uploaded image
#             os.remove(user_image_path)

#         except Exception as e:
#             logger.error(f"Error in GenerateCoverView: {str(e)}")
#             return JsonResponse({'error': str(e)}, status=500)

#         # Return the relative URL for React frontend
#         generated_image_url = f'http://192.168.1.7:8000/cover_one/static/generated_covers/{generated_image_name}'

#         return JsonResponse({
#             'message': 'Cover generated successfully',
#             'generated_image_url': generated_image_url,
#             'alt': model_name
#         }, status=200)

# # API to get all generated mobile covers
# class GetAllGeneratedCoversView(APIView):
#     def get(self, request, *args, **kwargs):
#         try:
#             # Directory where generated covers are stored
#             generated_covers_dir = os.path.join(settings.BASE_DIR, 'cover_one', 'static', 'generated_covers')

#             # Check if the directory exists
#             if not os.path.exists(generated_covers_dir):
#                 return JsonResponse({'error': 'Generated covers directory not found'}, status=404)

#             # List all images in the directory
#             generated_images = []
#             for image_file in os.listdir(generated_covers_dir):
#                 # Check for image file extensions
#                 if image_file.endswith((".png", ".jpg", ".jpeg")):
#                     image_url = f'http://192.168.1.7:8000/cover_one/static/generated_covers/{image_file}'
#                     generated_images.append(image_url)

#             # Return the list of generated images without a message
#             return JsonResponse(generated_images, safe=False, status=200)

#         except Exception as e:
#             logger.error(f"Error in GetAllGeneratedCoversView: {str(e)}")
#             return JsonResponse({'error': str(e)}, status=500)

