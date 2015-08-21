from django.http import JsonResponse
from django.shortcuts import render

from .models import Image


def random_img():
    """
    Returns a dictionary of info about a random non-annotated image
    """
    imgs = Image.objects.filter(annotation__isnull=True).order_by('?')
    if not imgs:
        return JsonResponse({
            'status': 'error',
            'error': 'No images remain to annotate'
        })

    i = imgs[0]
    return {
        'status': 'ok',
        'image_id': i.id,
        'image_url': i.url()
    }


def annotate(request):
    if request.method == 'POST':
        image_id = request.POST.get('image_id')
        bboxes = request.POST.getlist('new-bbox')
        img = Image.objects.get(id=image_id)
        img.annotation = bboxes
        img.save()

    return render(
        request,
        'train/annotate.html',
        {'img_data': random_img()}
    )


def getImage(request):
    """
    An API for getting random image data
    """
    return JsonResponse(random_img())
