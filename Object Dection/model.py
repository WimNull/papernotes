# #以（8, 8）为中心, 长宽比分别为[0.5, 1, 2], 面积分别为16*16 *[8, 16, 32], 共9个anchor,
# return (N, 4), '(y_{min}, x_{min}, y_{max}, x_{max})' of a bounding box.
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    py = base_size / 2.
    px = base_size / 2.
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base

# anchor_base为generate_anchor_base所生成, feat_stride即图片下采样到featsmap的采样率--16, h,w为特征图大小
# 对应每个方格的anchor
def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # n_anchor*2, 作为每个anchor的前景背景的分类得分, 二分类所以*2
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 每个anchor*4, (y1, x1, y2, x2)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
    # img_size为输入原图的大小
    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)

        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=2)
        rpn_fg_scores = rpn_softmax_scores[..., 1].contiguous().view(n, -1)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(), rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


# roi = loc2bbox(anchor, loc), 通过相对位置进行转换，返回的依然是(y1, x1, y2, x2)
def loc2bbox(src_bbox, loc):
    return roi

class ProposalCreator:
    # loc: 预测位置(相对位置)，score: 前景(含有目标物体)得分, 图像中所有anchor, img_size: 图像原始大小
    # ROI通过loc相对位置和anchor可以算出, 经过边界条件和NMS初步筛选一部分ROI
    # 返回筛选后的ROI
    def __call__(self, loc, score, anchor, img_size, scale=1.):
        roi = loc2bbox(anchor, loc)
        return roi