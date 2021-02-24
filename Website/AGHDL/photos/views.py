from django.shortcuts import render
from django.views.generic import ListView, DetailView
from .models import Photo
from .forms import PhotoForm
from django.http import HttpResponseRedirect
import torch
import torchvision
import torch.nn as nn
import os
import re
from PIL import Image
from torch.autograd import Variable


path_CNN = r'###' 


class ShowUserPhotos(ListView):
    model = Photo
    template_name = 'photos/user_photos_list.html'

class PhotoDetailView(DetailView):
    model = Photo
    template_name = 'photos/photo_detail.html'


def LikeView(request, pk):
    photo = Photo.objects.get(id=pk)
    photo.likes.add(request.user)
    return HttpResponseRedirect('http://127.0.0.1:8000/photos/photo/{}'.format(str(pk)))


def UnLikeView(request, pk):
    photo = Photo.objects.get(id=pk)
    photo.likes.remove(request.user)
    return HttpResponseRedirect('http://127.0.0.1:8000/photos/photo/{}'.format(str(pk)))


def image_loader(image_name, loader):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def model_form_upload(request):
    if request.method == 'POST':
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            form = form.save(commit=False)
            form.owner = request.user

            form.save()

            path = (os.path.join(r"###", str(form.img)))
            path = re.sub(r'/|\\', re.escape(os.sep), path)
            imsize = 256
            loader = torchvision.transforms.Compose([torchvision.transforms.Scale(imsize), torchvision.transforms.ToTensor()])
            frame = image_loader(path, loader)


            net = torchvision.models.resnet34(pretrained=False)
            for param in net.parameters():
                param.requires_grad = False

            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 8)
            net.load_state_dict(torch.load(path_CNN))

            classes = ['baby', 'cars', 'cats', 'dogs', 'product', 'food', 'gun', 'shoes']
            net.eval()
            with torch.no_grad():
                output = net(frame)
            _, indices = torch.sort(output, descending=True)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
            print(percentage)
            print('{} {} %'.format(classes[int(indices[0,0])], int(percentage[int(indices[0,0])])))
            if percentage[int(indices[0,0])] > 80:
                print('{} {} %'.format(classes[int(indices[0,0])], int(percentage[int(indices[0,0])])))
                form.tags = form.tags + ', ' + '#' + classes[int(indices[0,0])]
                if classes[int(indices[0,0])] == 'gun' or classes[int(indices[0,0])] == 'product':
                    form.tags_auto = 'violation'

            form.save()

            return HttpResponseRedirect('http://127.0.0.1:8000/photos/photo/{}'.format(form.pk))
    
    else:
        form = PhotoForm()
    return render(request, 'photos/add_photo.html', {
        'form': form
    })