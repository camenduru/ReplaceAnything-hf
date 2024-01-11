##!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023-06-01
# @Author  : ashui(Binghui Chen)
from sympy import im
from versions import RELEASE_NOTE, VERSION

import time
import cv2
import gradio as gr
import numpy as np
import random
import math
import uuid
import torch
from torch import autocast

from src.util import resize_image, HWC3, call_with_messages, upload_np_2_oss
from src.virtualmodel import call_virtualmodel
from src.person_detect import call_person_detect
from src.background_generation import call_bg_genration

import sys, os

from PIL import Image, ImageFilter, ImageOps, ImageDraw

from segment_anything import SamPredictor, sam_model_registry

mobile_sam = sam_model_registry['vit_h'](checkpoint='models/sam_vit_h_4b8939.pth').to("cuda")
mobile_sam.eval()
mobile_predictor = SamPredictor(mobile_sam)
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

# - - - - - examples  - - - - -  #
# 输入图地址, 文本, 背景图地址, index, []
image_examples = [
                            ["imgs/000.jpg", "一位年轻女性身穿短袖，展示一台手机", None, 0, []],
                            ["imgs/001.jpg", "一位年轻女性身穿短袖，手持杯子", None, 1, []],
                            ["imgs/003.png", "一名女子身穿黑色西服，背景蓝色", "imgs/003_bg.jpg", 2, []],
                            ["imgs/002.png", "一名年轻女性身穿裙子摆拍，背景是蓝色的", "imgs/002_bg.png", 3, []],
                            ["imgs/bg_gen/base_imgs/1cdb9b1e6daea6a1b85236595d3e43d6.png", "水滴飞溅", None, 4, []],
                            ["imgs/bg_gen/base_imgs/1cdb9b1e6daea6a1b85236595d3e43d6.png", "", "imgs/bg_gen/ref_imgs/df9a93ac2bca12696a9166182c4bf02ad9679aa5.jpg", 5, []],
                            ["imgs/bg_gen/base_imgs/IMG_2941.png", "在沙漠地面上", None, 6, []],
                            ["imgs/bg_gen/base_imgs/b2b1ed243364473e49d2e478e4f24413.png","白色地面，白色背景，光线射入，佳能",None,7,[]],
                        ]

img = "image_gallery/"
files = os.listdir(img)
files = sorted(files)
showcases = []
for idx, name in enumerate(files):
        temp = os.path.join(os.path.dirname(__file__), img, name)
        showcases.append(temp)

def process(input_image, original_image, original_mask, selected_points, source_background, prompt, face_prompt):
    if original_image is None or original_mask is None or len(selected_points)==0:
        raise gr.Error('请上传输入图片并通过点击鼠标选择需要保留的物体.')
    
    # load example image
    if isinstance(original_image, int):
            image_name = image_examples[original_image][0]
            original_image = cv2.imread(image_name)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    original_mask = np.clip(255 - original_mask, 0, 255).astype(np.uint8)

    request_id = str(uuid.uuid4())
    input_image_url = upload_np_2_oss(original_image, request_id+".png")
    input_mask_url = upload_np_2_oss(original_mask, request_id+"_mask.png")
    source_background_url = "" if source_background is None else upload_np_2_oss(source_background, request_id+"_bg.png")

    # person detect: [[x1,y1,x2,y2,score],]
    det_res = call_person_detect(input_image_url)

    res = []
    if len(det_res)>0:
        if len(prompt)==0:
            raise gr.Error('请输入prompt')
        res = call_virtualmodel(input_image_url, input_mask_url, source_background_url, prompt, face_prompt)
    else:
        ### 这里接入主图背景生成
        if len(prompt)==0:
            prompt=None
        ref_image_url=None if source_background_url =='' else source_background_url
        original_mask=original_mask[:,:,:1]
        base_image=np.concatenate([original_image, original_mask],axis=2)
        base_image_url=upload_np_2_oss(base_image, request_id+"_base.png")
        res=call_bg_genration(base_image_url,ref_image_url,prompt,ref_prompt_weight=0.5)

    return res, request_id, True

block = gr.Blocks(
        css="css/style.css",
        theme=gr.themes.Soft(
             radius_size=gr.themes.sizes.radius_none,
             text_size=gr.themes.sizes.text_md
         )
        ).queue(concurrency_count=3)
with block:
    with gr.Row():
        with gr.Column():
            
            gr.HTML(f"""
                    </br>
                    <div class="baselayout" style="text-shadow: white 0.01rem 0.01rem 0.4rem; position:fixed; z-index: 9999; top:0; left:0;right:0; background-size:100% 100%">
                        <h1 style="text-align:center; color:white; font-size:3rem; position: relative;"> ReplaceAnything (V{VERSION})</h1>
                    </div>
                    </br>
                    </br>
                    <div style="text-align: center;">
                        <h1 >ReplaceAnything as you want: Ultra-high quality content replacement</h1>
                        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                            <a href=""></a>
                            <a href='https://aigcdesigngroup.github.io/replace-anything/'><img src='https://img.shields.io/badge/Project_Page-ReplaceAnything-green' alt='Project Page'></a>
                            <a href='https://github.com/AIGCDesignGroup/ReplaceAnything'><img src='https://img.shields.io/badge/Github-Repo-blue'></a>
                        </div>
                        </br>
                        <h3>OffendingAIGC techniques have attracted lots of attention recently. They have demonstrated strong capabilities in the areas of image editing, image generation and so on. We find that generating new contents while strictly keeping the identity of use-specified object unchanged is of great demand, yet challenging. To this end, we propose ReplaceAnything framework. It can be used in many scenes, such as human replacement, clothing replacement, background replacement, and so on.</h3>
                        <h5 style="margin: 0; color: red">If you found the project helpful, you can click a Star on Github to get the latest updates on the project.</h5>
                        </br>
                    </div>
            """)

    with gr.Tabs(elem_classes=["Tab"]):
        with gr.TabItem("作品广场(Image Gallery)"):
            gr.Gallery(value=showcases,
                        height=800,
                        columns=4,
                        object_fit="scale-down"
                        )
        with gr.TabItem("创作图像(Image Create)"):  
            with gr.Accordion(label="🧭 操作指南(Instructions):", open=True, elem_id="accordion"):
                with gr.Row(equal_height=True):
                    gr.Markdown("""
                    - ⭐️ <b>step1：</b>在“输入图像”中上传or选择Example里面的一张图片(Upload or select one image from Example)
                    - ⭐️ <b>step2：</b>通过点击鼠标选择图像中希望保留的物体(Click on Input-image to select the object to be retained)
                    - ⭐️ <b>step3：</b>输入对应的参数，例如prompt等，点击Run进行生成(Input prompt or reference image for generating new contents)
                    - ⭐️ <b>step4 (optional)：</b>此外支持换背景操作，上传目标风格背景，执行完step3后点击Run进行生成(Click Run button)
                    """)                          
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_id="Input"):
                        with gr.Row():
                            with gr.Tabs(elem_classes=["feedback"]):
                                with gr.TabItem("输入图像(Input Image)"):
                                    input_image = gr.Image(type="numpy", label="输入图",scale=2)
                        original_image = gr.State(value=None,label="索引")
                        original_mask = gr.State(value=None)
                        selected_points = gr.State([],label="点选坐标")
                        with gr.Row(elem_id="Seg"):
                            radio = gr.Radio(['前景点选', '背景点选'], label='分割点选: ', value='前景点选',scale=2)
                            undo_button = gr.Button('撤销点选至上一步', elem_id="btnSEG",scale=1)
                    prompt = gr.Textbox(label="Prompt (支持中英文)", placeholder="请输入期望的文本描述",value='',lines=1)
                    run_button = gr.Button("生成图像（Run）",elem_id="btn")
                    
                    with gr.Accordion("更多输入参数 (推荐使用)", open=False, elem_id="accordion1"):
                        with gr.Row(elem_id="Image"):
                            with gr.Tabs(elem_classes=["feedback1"]):
                                with gr.TabItem("风格背景图输入(可选项)"):
                                    source_background = gr.Image(type="numpy", label="背景图")
                    
                        face_prompt = gr.Textbox(label="人脸 Prompt (支持中英文)", value='good face, beautiful face, best quality')
                with gr.Column():
                    with gr.Tabs(elem_classes=["feedback"]):
                        with gr.TabItem("输出结果"):
                            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True)
                            recommend=gr.Button("推荐至作品广场",elem_id="recBut")
                            request_id=gr.State(value="")
                            gallery_flag=gr.State(value=False)
            with gr.Row():
                with gr.Box():
                    def process_example(input_image, prompt, source_background, original_image, selected_points):
                        return input_image, prompt, source_background, original_image, []
                    example = gr.Examples(
                        label="输入图示例",
                        examples=image_examples,
                        inputs=[input_image, prompt, source_background, original_image, selected_points],
                        outputs=[input_image, prompt, source_background, original_image, selected_points],
                        fn=process_example,
                        run_on_click=True,
                        examples_per_page=10
                    )

     # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        # 图片太大传输太慢了
        if min(img.shape[0], img.shape[1]) > 1024:
            img = resize_image(img, 1024)
        return img, img, [], None  # when new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [input_image, original_image, selected_points, source_background]
    )

    # user click the image to get points, and show the points on the image
    def segmentation(img, sel_pix):
        # online show seg mask
        points = []
        labels = []
        for p, l in sel_pix:
            points.append(p)
            labels.append(l)
        mobile_predictor.set_image(img if isinstance(img, np.ndarray) else np.array(img))
        with torch.no_grad():
            with autocast("cuda"):
                masks, _, _ = mobile_predictor.predict(point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False)

        output_mask = np.ones((masks.shape[1], masks.shape[2], 3))*255
        for i in range(3):
                output_mask[masks[0] == True, i] = 0.0

        mask_all = np.ones((masks.shape[1], masks.shape[2], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
                mask_all[masks[0] == True, i] = color_mask[i]
        masked_img = img / 255 * 0.3 + mask_all * 0.7
        masked_img = masked_img*255
        ## draw points
        for point, label in sel_pix:
            cv2.drawMarker(masked_img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return masked_img, output_mask
    
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == '前景点选':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == '背景点选':
            sel_pix.append((evt.index, 0))    # append the background_point
        else:
            sel_pix.append((evt.index, 1))    # default foreground_point

        if isinstance(img, int):
            image_name = image_examples[img][0]
            img = cv2.imread(image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # online show seg mask
        masked_img, output_mask = segmentation(img, sel_pix)
        return masked_img.astype(np.uint8), output_mask
    
    input_image.select(
        get_point,
        [original_image, selected_points, radio],
        [input_image, original_mask],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        # draw points
        output_mask = None
        if len(sel_pix) != 0:
            if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
                temp = cv2.imread(image_examples[orig_img][0])
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                temp = orig_img.copy()
            sel_pix.pop()
            # online show seg mask
            if len(sel_pix) !=0:
                temp, output_mask = segmentation(temp, sel_pix)
            return temp.astype(np.uint8), output_mask
        else:
            gr.Error("暂无“上一步”可撤销")
    
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image, original_mask]
    )

    def upload_to_img_gallery(img, res, re_id, flag):
        if flag:
            if isinstance(img, int):
                image_name = image_examples[img][0]
                img = cv2.imread(image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _ = upload_np_2_oss(img, name=re_id+"_ori.jpg", gallery=True)
            for idx, r in enumerate(res):
                r = cv2.imread(r['name'])
                r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                _ = upload_np_2_oss(r, name=re_id+f"_res_{idx}.jpg", gallery=True)
            flag=False
            gr.Info("图片已经被上传完毕，待审核")
        else:
            gr.Info("暂无图片可推荐，或者已经推荐过一次了")
        return flag

    recommend.click(
        upload_to_img_gallery,
        [original_image, result_gallery, request_id, gallery_flag],
        [gallery_flag]
    )

    ips=[input_image, original_image, original_mask, selected_points, source_background, prompt, face_prompt]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, request_id, gallery_flag])


block.launch(share=True)
