from django.http import JsonResponse
from django.shortcuts import render

from .models import Image


def annotate(request):
    if request.method == 'POST':
        # TODO save form data
        pass
    return render(request, 'train/annotate.html')


def getImage(request):
    imgs = Image.objects.filter(annotation__isnull=True).order_by('?')
    if not imgs:
        return JsonResponse({
            'status': 'error',
            'error': 'No images remain to annotate'
        })

    i = imgs[0]
    return JsonResponse({
        'status': 'ok',
        'image_id': i.id,
        'image_url': i.url()
    })
