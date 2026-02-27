import os


work_dir = os.environ.get("WORK_DIR", "./work_dirs/portable_sam_fusion/rsprompter_anchor_satS_v8_sam2")

num_classes = 1
prompt_shape = (32, 4)

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    os.path.abspath(os.path.join("checkpoints", "sam2.1_hiera_base_plus.pt")),
)

sam2_config_path = os.environ.get(
    "SAM2_CONFIG",
    os.path.abspath(os.path.join("checkpoints", "sam2.1_hiera_b+.yaml")),
)

model = dict(
    type="RSPrompterAnchorDroneGuidance",
    enable_drone_branch=False,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type="BatchFixedSizePad",
                size=(512, 512),
                img_pad_value=0,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=False,
            )
        ],
    ),
    decoder_freeze=False,
    shared_image_embedding=dict(
        type="RSSAM2PositionalEmbedding",
        image_size=512,
        patch_size=16,
        embed_dim=256,
    ),
    backbone=dict(
        type="RSSAM2VisionEncoder",
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        use_gradient_checkpointing=True,
        freeze_backbone=False,
        freeze_backbone_stages=-1,
    ),
    neck=dict(
        type="RSFeatureAggregatorSAM2",
        in_channels="sam2_hiera_base",
        out_channels=256,
        hidden_channels=32,
        select_layers=[0, 1, 2, 3],
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="RSPrompterAnchorRoIPromptHead",
        with_extra_pe=True,
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type="RSPrompterAnchorMaskHeadSAM2",
            sam2_mask_decoder=dict(
                checkpoint_path=sam2_ckpt_path,
                use_high_res_features=True,
            ),
            in_channels=256,
            roi_feat_size=14,
            per_pointset_point=prompt_shape[1],
            with_sincos=False,
            multimask_output=False,
            class_agnostic=True,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            mask_size=(512, 512),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.4),
    ),
)
