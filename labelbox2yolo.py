import json
import os
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm import tqdm

from utils import make_dirs


def convert(file, zip=True):
    """Converts Labelbox JSON labels to YOLO format and saves them, with optional zipping."""
    names = []  # class names
    file = Path(file)
    save_dir = make_dirs(file.stem)
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    for img in tqdm(data, desc=f'Converting {file}'):
        im_path = img['data_row']['row_data']
        external_id = img['data_row']['external_id']
        im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)
        width, height = im.size

        labels_dir = save_dir / 'labels'
        os.makedirs(labels_dir, exist_ok=True)
        label_path = labels_dir / Path(external_id).with_suffix('.txt').name

        images_dir = save_dir / 'images'
        os.makedirs(images_dir, exist_ok=True)
        image_path = images_dir / external_id
        im.save(image_path, quality=95, subsampling=0)

        for project in img['projects'].values():
            for label in project['labels']:
                for obj in label['annotations']['objects']:
                    top, left, h, w = obj['bounding_box'].values()
                    xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]
                    cls = obj['name']
                    if cls not in names:
                        names.append(cls)
                    line = names.index(cls), *xywh
                    with open(label_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

    d = {'path': f"../datasets/{file.stem}", 'train': "images/train", 'val': "images/val", 'test': "", 'nc': len(names), 'names': names}
    with open(save_dir / file.with_suffix('.yaml').name, 'w') as f:
        yaml.dump(d, f, sort_keys=False)

    if zip:
        print(f'Zipping as {save_dir}.zip...')
        os.system(f'zip -qr {save_dir}.zip {save_dir}')
    print('Conversion completed successfully!')




if __name__ == "__main__":
    convert("labelbox.ndjson")