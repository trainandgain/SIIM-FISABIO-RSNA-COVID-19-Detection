from model.gen_model import get_model
from dataset.gen_dataloader import get_dataloader
from transofrm.gen_transform import get_transform
import utils
import utils.config
import utils.device
import utils.input


def figure_boxes(config, j, i, image, gt_boxes, gt_label, pred_list, threshold=None, match_boxes=True):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    for i, box in enumerate(gt_boxes):
        xmin, ymin, xmax, ymax = box
        startX, endX, startY, endY = int(xmin), int(xmax), int(ymin), int(ymax)
        color = (0, 0, 120)
        thickness = 1
        image = cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    for i, (box, label, score) in enumerate(pred_list):
        if threshold:
            if score<threshold:
                continue
        if match_boxes:
            if i >= len(gt_boxes):
                continue

        xmin, ymin, xmax, ymax = box
        startX, endX, startY, endY = int(xmin), int(xmax), int(ymin), int(ymax)
        color = (1, 0, 0)
        thickness = 1
        image = cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        image = cv2.putText(image, f'C: {int(label)} S: {float(score):.3}', (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)

    ax.set_axis_off()
    ax.set_title('Image Classes: '+str(gt_label)+' Black: Ground-truth | White:
    Predicted, Image Precision: {}'.format(precision)
    ax.imshow(image, cmap=plt.cm.bone)
    save_dir = config['inference']['out']
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir+'{i}-{j}'.format(i=i, j=j, dpi=300, bbox_inches='tight', pad_inches=0)


def inference(config, model, dataloaders, DEVICE, metric):
    for j, (images, targets, idx) in enumerate(d):
        if j == 20:
            break
        pred = model(images)
        for i, image in enumerate(images):
            # zip up boxes, labels and scores
            pred_list = list(zip(pred[i]['boxes'].detach().numpy(),
                                  pred[i]['labels'].detach().numpy(),
                                  pred[i]['scores'].detach().numpy()))
            # reverse the order, ensure sort
            pred_list.sort(reverse=True, key=lambda x: x[2])
            # get ground truth boxes
            gt_boxes = targets[i]['boxes'].numpy().astype(np.int32)
            gt_label = targets[i]['labels'].numpy().astype(np.int32)
            # image
            image = image.squeeze().numpy()
            # calculate precision
            precision = metric(np.array([entry[0] for entry in pred_list]),
                                          gt_boxes,
                                          thresholds=Config.iou_threshold,
                                          form='pascal_voc')
            # display
            figure_boxes(config, j, i, image, gt_boxes, gt_label, pred_list)


def run(config):
    DEVICE = utils.device.get_device()
    model = get_model(config)
    model = utils.load(model, config).to(DEVICE)
    df = utils.input.get_dfs(config)
    dataloaders = {split:get_dataloader(config, df, split, get_transform(config, split)) for split in ['train', 'val']}
    # inference loop
    inference(config, model, dataloaders, DEVICE)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = utils.config.load(args.config_file)
    print(config)
    run(config)
