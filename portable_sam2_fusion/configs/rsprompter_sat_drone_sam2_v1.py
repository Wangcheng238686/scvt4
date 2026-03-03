import os


work_dir = os.environ.get("WORK_DIR", "./work_dirs/portable_sam2_fusion/rsprompter_sat_drone_sam2_v1")

num_classes = 1
prompt_shape = (32, 4)

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    "portable_sam2_fusion/checkpoints/sam2.1_hiera_base_plus.pt",
)

model = dict(
    type="RSPrompterAnchorDroneGuidance",
    enable_drone_branch=True,
    share_sam2_backbone=False,
    use_height_guided_fusion=True,
    
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
            num_classes=num_classes,
            in_channels=256,
            conv_out_channels=256,
            num_fcs=2,
            fc_out_channels=256,
            roi_feat_size=14,
            per_pointset_point=prompt_shape[1],
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            sam2_mask_decoder=dict(
                checkpoint_path=sam2_ckpt_path,
                use_high_res_features=True,
            ),
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
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
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
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=(512, 512),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        ),
    ),
    
    drone_branch=dict(
        use_sam2_backbone=True,
        freeze=False,
        dim=128,
        input_size=512,
        middle=[2, 2, 2, 2],
        scale=1.0,
        use_checkpoint=True,
        cross_view=dict(
            image_height=512,
            image_width=512,
            qkv_bias=True,
            heads=4,
            dim_head=32,
            no_image_features=False,
            skip=True,
        ),
        bev_embedding=dict(
            bev_height=80,
            bev_width=80,
            sigma=128,
        ),
        scene_alignment=dict(
            max_scenes=1000,
            embed_dim=32,
            init_scale=0.01,
        ),
        depth_encoder=dict(
            enabled=True,
            encoder="vits",
            freeze=True,
            use_depth_features=True,
            depth_dim=64,
            use_multiview_consistency=False,
            consistency_weight=0.1,
        ),
    ),
    
    guidance=dict(
        bev_dim=128,
        level_channels=[256, 256, 256, 256, 256],
    ),
    
    spatial_fusion_cfg=dict(
        num_heads=4,
        temperature=0.1,
        align_loss_weight=0.1,
        downsample_factor=1,
        max_attn_size=64,
        use_checkpoint=True,
        height_dim=64,
        use_height_gate=True,
        height_loss_weight=0.01,
        # Simplified height loss configuration - only keep essential variance loss
        use_variance_loss=True,      # Essential: encourage bimodal height distribution
        use_sparsity_loss=False,     # Disabled: not critical for convergence
        use_smoothness_loss=False,   # Disabled: not critical for convergence
        variance_weight=0.05,        # Reduced weight for stability
        sparsity_weight=0.0,         # Disabled
        smoothness_weight=0.0,       # Disabled
    ),
    
    # Cross-view losses disabled to reduce complexity and improve convergence
    # These losses are helpful but add too much optimization pressure during initial training
    cross_view_loss=None,
    
    max_scenes=1000,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        scale=(512, 512),
        keep_ratio=True,
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="SatelliteInstanceDataset",
        data_root="",
        ann_file="",
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="SatelliteInstanceDataset",
        data_root="",
        ann_file="",
        data_prefix=dict(img=""),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file="",
    metric=["bbox", "segm"],
    format_only=False,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.5, decay_mult=1.0),
            "neck": dict(lr_mult=1.0, decay_mult=1.0),
            "drone_encoder": dict(lr_mult=1.0, decay_mult=1.0),
            "guidance": dict(lr_mult=1.0, decay_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30, 40],
        gamma=0.1,
    ),
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

log_level = "INFO"
load_from = None
resume = False
