from PIL import Image, ImageEnhance, ImageOps
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import os
from django.conf import settings

@method_decorator(csrf_exempt, name='dispatch')
class ImproveImageQualityView(View):
    def post(self, request, *args, **kwargs):
        try:
            # Check if image file is in the request
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({'error': 'No image uploaded'}, status=400)
            
            # Get alignment parameters (left, center, right) from the frontend (optional)
            alignment = request.POST.get('alignment', 'center')

            # Open the image using PIL
            image = Image.open(image_file)

            # Improve image quality (adjust contrast, sharpness, etc.)
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1.5)  # Increase contrast by 1.5 times

            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(2.0)  # Increase sharpness by 2 times

            # Ensure 'improved_images' directory exists inside the media root
            output_dir = os.path.join(settings.MEDIA_ROOT, 'improved_images')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the improved image
            output_path = os.path.join(output_dir, image_file.name)
            image.save(output_path)

            # Construct the image URL dynamically
            image_url = f'{request.scheme}://{request.get_host()}/media/improved_images/{image_file.name}'

            # Return the improved image URL in the response
            return JsonResponse({'image_url': image_url}, status=200)

        except Exception as e:
            return JsonResponse({'error': f'Error processing the image: {str(e)}'}, status=500)
