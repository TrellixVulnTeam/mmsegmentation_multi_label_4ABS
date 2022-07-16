checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'

loss1 = [dict(type='FocalLoss',gamma=1.0,alpha=0.5,use_sigmoid=True,
loss_weight=2.0),dict(type='DiceLoss',multi_label=True, loss_weight=2.0)]
loss2 = [dict(type='FocalLoss',gamma=1.0,alpha=0.5,use_sigmoid=True,
loss_weight=0.8),dict(type='DiceLoss',multi_label=True, loss_weight=0.8)]

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=loss1),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=loss2),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', multi_label=True, train_test=True,
                  THRESHOD=[0.85, 0.5, 0.45], logits=True, nature=False))

# dataset settings
dataset_type = 'CustomDataset'
data_root = '/root/autodl-tmp/mmseg_train_dn_360/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0,0,0], [128,128,128], [255,255,255]]
img_norm_cfg = dict(mean=[0,0,0], std=[1,1,1], to_rgb=True)
size = (360,360)

albu = [dict(type='OneOf',
        transforms=[dict(type='RandomBrightnessContrast',
                        brightness_limit=[0.05, 0.20],
                        contrast_limit=[0.05, 0.20],
                        p=0.2),
                    dict(type='ElasticTransform',alpha=1, sigma=25, alpha_affine=25, interpolation=1, border_mode=2, value=None, mask_value=None, p=0.2),
                   ],p=0.5),
        dict(type='OneOf',
        transforms=[dict(type='MedianBlur', blur_limit=3, p=0.1),
                    dict(type='GridDistortion',num_steps=5, distort_limit=0.10, interpolation=1, border_mode=2, value=None, mask_value=None, p=0.2)
                   ],p=0.5),
       dict(
            type='ShiftScaleRotate',
            shift_limit=0.0625,
            scale_limit=0.0,
            rotate_limit=0,
            interpolation=1,
            p=0.2)]
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='max'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=size),
    dict(type='RandomRotate',prob=0.5, degree=45, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Albu', transforms=albu),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='max'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=size,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/train_all.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/train_all.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        test_mode=True,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/train_all.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
seed = 2022
total_iters = 120

optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1e-3,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
# runtime settings
find_unused_parameters=True
runner = dict(type='IterBasedRunner', max_iters=int(total_iters * 1000))
checkpoint_config = dict(interval=-1, save_optimizer=False)
evaluation = dict(by_epoch=False, interval=1000000, save_best='mDice',
                  save_optimizer=False , min_step=1000000, metric=['imDice','mDice','mIoU','PR'], pre_eval=False)
fp16 = dict()

work_dir = f'/root/Swin_Small_Uper'