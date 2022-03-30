# Object Detection: A Survey
## MilleStone
<div align=center>
<img src="images/milestones.jpg"/>
</div>

# IOU

```python
# IOU计算
def iou(box1, box2):
    # box1:(N, 4), box2:(M, 4)
    N = box1.size(0)
    M = box2.size(0)
    # left top
    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0   # 两个box没有重叠区域
    inter = wh[:,:,0] * wh[:,:,1]   # [N,M]
    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # (N,)
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N,M)
    iou = inter / (area1+area2-inter)
    return iou
```

# NMS
NMS（非极大抑制）
NMS算法一般是为了去掉模型预测后的多余框，其一般设有一个nms_threshold=0.5，具体的实现思路如下：
1, 选取这类box中scores最大的哪一个，记为box_best，并保留它
2, 计算box_best与其余的box的IOU
3, 如果其IOU>0.5了，那么就舍弃这个box（由于可能这两个box表示同一目标，所以保留分数高的哪一个）
4, 从最后剩余的boxes中，再找出最大scores的哪一个，如此循环往复

```python
# NMS算法
def nms(bboxes, scores, threshold=0.5):
    # bboxes:(N, 4), scores:(N,)
    x1,y1, x2,y2 = bboxes[:]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列
    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor
```

# Two-stage detector

## [RCNN](https://arxiv.org/pdf/1311.2524.pdf)

## Method

<div align=center>
<img src="images/rcnn.jpg"/>
<center>R-CNN Pipeline</center>
</div>


2. Extract region proposals

objectness--[Measuring the objectness of image windows](http://calvin-vision.net/wp-content/uploads/Publications/alexe12pami.pdf)

selective search--[Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

category-independent object proposals--[Category Independent Object Proposals](http://dhoiem.cs.illinois.edu/publications/pami2013_proposals_endres.pdf)

4. Classify regions



## [SPPNet](https://arxiv.org/pdf/1406.4729.pdf)
<div align=center>
<img src="images/sppnet.jpg"/>
<center>对输入的特征图分别划分为16个、4个、1个不重叠区域池化</center>
</div>

## [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
<div align=center>
<img src="images/fast-rcnn.jpg"/>
<center></center>
</div>

## [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)

<div align=center>
<img src="images/fast-rcnn.jpg"/>
<center></center>
</div>


# One-stage detector

## [YoloV1](https://arxiv.org/pdf/1506.02640.pdf)

<div align=center>
<img src="images/yolov1.jpg"/>
<center>Architecture</center>
</div>

<div align=center>
<img src="images/yolov1det.jpg"/>
</div>
S为预先设定划分为SxS个不重叠的区域, 每个区域负责不同的预测, 对应网络输出为(S×S×(B∗5+C)), 其中B表示每个区域负责预测的BBox(y, x, h, w, c(置信度))个数, C代表类别数。

<div align=center>
<img src="images/yolov1loss.jpg"/>
<center>YoloV1 Loss, 空心1为指示函数(存在与否)</center>
</div>
首先初始化一个与输出对应的标签矩阵, 然后标签通过bbox定位到哪个区域, 存入相关信息, 具体代码如下: 

```python
# 生成训练GT
def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h
    if box_w < 1. or box_h < 1.:
        # print('A dirty data !!!')
        return False    
    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    weight = 2.0 - (box_w / w) * (box_h / h)
    return grid_x, grid_y, tx, ty, tw, th, weight

def gt_creator(input_size, stride, label_lists=[], name='VOC'):
    assert len(input_size) > 0 and len(label_lists) > 0
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    w = input_size[1]
    h = input_size[0]
    
    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])

    # generate gt whose style is yolo-v1
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result
                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0 # exist or not
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight
```

## [YoloV2](https://arxiv.org/pdf/1612.08242.pdf)