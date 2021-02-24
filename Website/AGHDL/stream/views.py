from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import threading
from django.views.generic import TemplateView
from django.template import loader
from django.shortcuts import redirect
import torch
import torchvision
import torch.nn as nn
import numpy
import cv2
from PIL import Image
from torch.autograd import Variable


path_CNN = r'###' 
device = torch.device('cuda')
frame_hide = cv2.imread(r'###\static\icons\hide.png')
frame_stop = cv2.imread(r'###\static\icons\stop_512.png')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

    def get_frame_CNN(self, loader):
        image = numpy.transpose(self.frame, (2, 0, 1))
        image = Image.fromarray(self.frame)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        image = to_device(image, device)
        return image

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
            
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def gen(camera, net, classes, x):
    count = 0
    frame_count = 0
    hide_count = 0
    while x:
        frame = camera.get_frame()
        CNN_check, hide = check_CNN(camera, net, classes)

        if CNN_check:
            count += 1

        if count >= 200:
            x = False
        
        if frame_count == 1000:
            frame_count = 0
            count = 0
        frame_count += 1

        while hide:
            hide_count += 1
            ret, jpeg = cv2.imencode('.jpg', frame_hide)
            frame = jpeg.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            if hide_count == 1000:
                hide_count = 0
                CNN_check, hide = check_CNN(camera, net, classes)
            

            
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.__del__()
    while not x:
        ret, jpeg = cv2.imencode('.jpg', frame_stop)
        frame = jpeg.tobytes()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def check_CNN(camera, net, classes):
    imsize = 256
    loader = torchvision.transforms.Compose([torchvision.transforms.Scale(imsize), torchvision.transforms.ToTensor()])
    frame_CNN = camera.get_frame_CNN(loader)

    with torch.no_grad():
        output = net(frame_CNN)
    _, indices = torch.sort(output, descending=True)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100

    if percentage[int(indices[0,0])] > 80:
        print('{} {} %'.format(classes[int(indices[0,0])], int(percentage[int(indices[0,0])])))

        if classes[int(indices[0,0])] == 'gun' or classes[int(indices[0,0])] == 'product':
            CNN_check = True
            hide = False
            return CNN_check, hide

        elif classes[int(indices[0,0])] == 'cats' or classes[int(indices[0,0])] == 'dogs':
            CNN_check = False
            hide = True
            return CNN_check, hide

        else:
            CNN_check = False
            hide = False
            return CNN_check, hide

    else:
        CNN_check = False
        hide = False
        return CNN_check, hide


@gzip.gzip_page
def live_feed(request):
    try:
        net = torchvision.models.resnet34(pretrained=False)
        for param in net.parameters():
            param.requires_grad = False

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 7)

        net.load_state_dict(torch.load(path_CNN))
        to_device(net, device)
        net.eval()
        classes = ['cars', 'cats', 'dogs', 'product', 'food', 'gun', 'shoes']
        x=True
        return StreamingHttpResponse(gen(VideoCamera(), net, classes, x), content_type='multipart/x-mixed-replace; boundary=frame')
    except:
        pass


def stop_stream(request):

    return redirect('http://127.0.0.1:8000/photos/user_photos/')

class StreamView(TemplateView):
    template_name = 'stream/start_streaming.html'
    
    