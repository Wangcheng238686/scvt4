import os


work_dir = os.environ.get("WORK_DIR", "./work_dirs/portable_sam_fusion/rsprompter_anchor_satS_drone_guidance_v1")

num_classes = 1
prompt_shape = (32, 4)

hf_sam_pretrain_name = os.environ.get("SAT_SAM_HF_PRETRAIN", "facebook/sam-vit-base")
hf_sam_pretrain_ckpt_path = os.environ.get(
    "SAT_SAM_CKPT",
    os.path.abspath(os.path.join("checkpoints", "sam_vit_b_01ec64.pth")),
)

model = dict(
    type="RSPrompterAnchorDroneGuidance",
    enable_drone_branch=True,
    use_spatial_fusion=True,
    spatial_fusion_cfg=dict(
        num_heads=4,
        temperature=0.1,
        align_loss_weight=0.1,
        downsample_factor=4,
        max_attn_size=128,
        use_checkpoint=True,
    ),
    max_scenes=1000,
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
                size=(1024, 1024),
                img_pad_value=0,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=False,
            )
        ],
    ),
    decoder_freeze=False,
    shared_image_embedding=dict(
        type="RSSamPositionalEmbedding",
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(image_size=1024),
        init_cfg=dict(type="Pretrained", checkpoint=hf_sam_pretrain_ckpt_path),
        use_offline_mode=True,
    ),
    backbone=dict(
        type="RSSamVisionEncoder",
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=True, image_size=1024),
        init_cfg=dict(type="Pretrained", checkpoint=hf_sam_pretrain_ckpt_path),
        use_offline_mode=True,
        use_gradient_checkpointing=True,
    ),
    neck=dict(
        type="RSFPN",
        feature_aggregator=dict(
            type="RSFeatureAggregator",
            in_channels=hf_sam_pretrain_name,
            out_channels=256,
            hidden_channels=32,
            select_layers=range(1, 13, 2),
        ),
        feature_spliter=dict(
            type="RSSimpleFPN",
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type="LN2d", requires_grad=True),
        ),
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
            type="RSPrompterAnchorMaskHead",
            mask_decoder=dict(
                type="RSSamMaskDecoder",
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type="Pretrained", checkpoint=hf_sam_pretrain_ckpt_path),
                use_offline_mode=True,
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
            mask_size=(1024, 1024),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.5),
    ),
    drone_branch=dict(
        pretrained=False,
        freeze=False,
        dim=128,
        middle=[2, 2, 2, 2],
        scale=1.0,
        use_checkpoint=True,
        cross_view=dict(
            image_height=512,
            image_width=512,
            qkv_bias=False,
            heads=4,
            dim_head=32,
            skip=True,
        ),
        bev_embedding=dict(
            sigma=1.0,
            bev_height=80,
            bev_width=80,
            h_meters=100.0,
            w_meters=100.0,
            offset=0.0,
            decoder_blocks=[2, 2],
            grid_refine_scale=0.5,
            init_scale=0.1,
        ),
        scene_alignment=dict(
            max_scenes=1000,
            embed_dim=32,
            init_scale=0.05,
        ),
    ),
    guidance=dict(
        bev_dim=128,
        level_channels=[256, 256, 256, 256, 256],
        gate_image_embeddings=False,
        gate_scale=0.1,
    ),
    cross_view_loss=dict(
        temperature=0.05,
        contrastive_weight=0.5,
        hidden_dim=256,
        consistency_weight=0.1,
        geometric_weight=0.0,
        geometric_trans_weight=1.0,
        geometric_rot_weight=1.0,
        smoothness_weight=0.0,
    ),
)
