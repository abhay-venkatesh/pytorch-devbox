from pycocotools.coco import COCO

ANNOTATIONS_FILEPATH = "../../../data/coco/annotations/stuff_train2017.json"


class CocoBox:
    def __init__(self, anns_file_path):
        self.coco = COCO(anns_file_path)

    def count_occurrences(self, cat_id):
        img_ids = self.coco.getImgIds(catIds=cat_id)
        print(len(img_ids))
        return len(img_ids)

    def show_all_categories(self):
        cats = coco.loadCats(coco.getCatIds())
        names = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(names)))


if __name__ == "__main__":
    cb = CocoBox(ANNOTATIONS_FILEPATH)
    cb.count_occurrences(157)