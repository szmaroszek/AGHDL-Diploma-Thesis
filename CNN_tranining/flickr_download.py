import flickrapi
import os
import urllib

flickr = flickrapi.FlickrAPI('###', '###', cache=True)

keyword = '###'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_m',
                     per_page=100,
                     sort='relevance')

urls = []
max = 1000
path = r'###'
for i, photo in enumerate(photos):
    if i < max:
        url = photo.get('url_m')
        if url is not None:
            try:
                photo_path = os.path.join(path, '{}.jpg'.format(str(i)))
                print (photo_path)
                urllib.request.urlretrieve(str(url), str(photo_path))
            except:
                pass
    else:
        break

