"""Picture-book Images API integration. Credentials stay in environment variables."""
import base64
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import re
from functools import lru_cache
import ctypes
from ctypes import wintypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class BookRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1, max_length=50)
    character: str = Field(min_length=1, max_length=150)
    scene: str = Field(min_length=1, max_length=150)
    action: str = Field(min_length=1, max_length=150)
    style: str = Field(min_length=1, max_length=100)
    wishes: str = Field(default='', max_length=600)
    captions: list[str] = Field(min_length=1, max_length=4)


def sanitize_svg(source):
    if not isinstance(source, str) or len(source)>80000 or '<!DOCTYPE' in source.upper() or '<!ENTITY' in source.upper():
        raise ValueError('Invalid SVG')
    allowed={'svg','g','path','rect','circle','ellipse','line','polyline','polygon','text','tspan','defs','linearGradient','radialGradient','stop','clipPath'}
    attrs={'id','d','x','y','x1','x2','y1','y2','cx','cy','r','rx','ry','width','height','points','fill','stroke','stroke-width','stroke-linecap','stroke-linejoin','opacity','fill-opacity','stroke-opacity','transform','font-size','font-family','font-weight','text-anchor','offset','stop-color','stop-opacity','clip-path','gradientTransform','gradientUnits'}
    tree=ET.fromstring(source)
    if tree.tag.split('}')[-1]!='svg' or len(list(tree.iter()))>1200:
        raise ValueError('Invalid SVG structure')
    def copy(node, depth=0):
        tag=node.tag.split('}')[-1]
        if tag not in allowed or depth>32:
            return None
        result=ET.Element(tag)
        for key,value in node.attrib.items():
            if key not in attrs or len(value)>14000 or any(x in value.lower() for x in ('javascript:','data:','http:','https:','@import','expression')):
                continue
            if 'url(' in value.lower() and not re.fullmatch(r'url\(#[a-zA-Z0-9_-]+\)',value):
                continue
            result.set(key,value)
        result.text=node.text
        for child in node:
            clean=copy(child,depth+1)
            if clean is not None:
                result.append(clean)
        return result
    root=copy(tree)
    root.set('xmlns','http://www.w3.org/2000/svg')
    root.set('viewBox','0 0 768 768')
    root.set('width','768')
    root.set('height','768')
    return ET.tostring(root,encoding='utf-8')


@lru_cache(maxsize=1)
def api_key():
    if os.getenv('DEEPSEEK_API_KEY'):
        return os.environ['DEEPSEEK_API_KEY']
    if os.name=='nt' and os.getenv('DEEPSEEK_KEY_FILE'):
        class Blob(ctypes.Structure):
            _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]
        raw=Path(os.environ['DEEPSEEK_KEY_FILE']).read_bytes()
        buffer=ctypes.create_string_buffer(raw)
        source=Blob(len(raw),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
        result=Blob()
        if ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source),None,None,None,None,0,ctypes.byref(result)):
            try:
                return ctypes.string_at(result.data,result.size).decode()
            finally:
                ctypes.windll.kernel32.LocalFree(result.data)
    return ''


def generate_image(prompt):
    key = api_key()
    endpoint = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
    parsed = urlparse(endpoint)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Image API URL must be a plain HTTPS base URL')
    system=('You are a children\'s picture-book author and SVG illustrator. Return ONLY a json object '
        'with keys caption (Chinese, at most 160 characters) and svg (complete SVG string). '
        'Use viewBox="0 0 768 768", draw a rich friendly scene with identifiable characters, facial expressions, scenery and foreground details. '
        'Keep the requested characters and outfit consistent. Use 40-70 simple shapes; keep SVG under 12000 characters. Use only SVG shapes, paths, text, groups and gradients. '
        'No script, animation, foreignObject, external resources, image tags, CSS stylesheets or hyperlinks. '
        'Do not draw letters or captions inside the illustration. Keep everything child-appropriate. '
        'Example json: {"caption":"小兔子出发了。","svg":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 768 768\\"><rect width=\\"768\\" height=\\"768\\" fill=\\"#def\\"/></svg>"}')
    request = urllib.request.Request(endpoint + '/chat/completions',
        data=json.dumps({'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'),
            'messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
            'response_format':{'type':'json_object'}, 'thinking':{'type':'disabled'}, 'max_tokens':8192}).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read(18000000)
    response=json.loads(raw)
    choice=response['choices'][0]
    if choice.get('finish_reason')=='length':
        raise ValueError('Truncated response')
    data=json.loads(choice['message']['content'])
    image=sanitize_svg(data['svg'])
    caption=data.get('caption','')
    if not isinstance(caption,str) or not caption.strip() or len(caption)>200:
        raise ValueError('Invalid caption')
    return 'data:image/svg+xml;base64,' + base64.b64encode(image).decode(),caption


def create_router(authorized):
    router = APIRouter(prefix='/api/picturebook', dependencies=[Depends(authorized)])
    jobs, lock, pool = {}, threading.RLock(), ThreadPoolExecutor(max_workers=1)

    def public(job):
        return {key: value for key, value in job.items() if not key.startswith('_')}

    def worker(job_id, book):
        job = jobs[job_id]
        try:
            for index, caption in enumerate(book.captions):
                if job['_stop'].is_set():
                    job['status'] = 'cancelled'
                    return
                context = '\n'.join(f'{i+1}. {c}' for i, c in enumerate(book.captions))
                prompt = (f'Create a warm, child-appropriate picture-book illustration. No text, letters, watermarks or speech bubbles. '
                    f'Character identity must stay consistent across the story: {book.character}. Setting: {book.scene}. '
                    f'Overall adventure: {book.action}. Art style: {book.style}. Additional wishes: {book.wishes}. '
                    f'Whole story for context:\n{context}\nDraw ONLY scene {index+1}: {caption}. '
                    f'Single square illustration, clear composition, friendly expressions, imaginative and non-graphic.')
                image,caption = generate_image(prompt)
                with lock:
                    job['images'] = [*job['images'], image]
                    job['captions'] = [*job['captions'], caption]
            job['status'] = 'cancelled' if job['_stop'].is_set() else 'done'
        except urllib.error.HTTPError as exc:
            job['status'] = 'error'
            job['message'] = {401:'DeepSeek 密钥无效，请老师检查后端配置。', 402:'DeepSeek 账户余额不足，请老师在服务商账户中开通用量。', 403:'图像服务暂未允许这个请求，请检查模型权限或输入内容。',
                429:'图像服务额度不足或请求过于频繁，请老师检查账户用量后重试。'}.get(exc.code, '图像服务未完成请求，请稍后重试或调整故事要求。')
        except Exception:
            import logging
            logging.exception('DeepSeek picture-book generation failed')
            job.update(status='error', message='图像生成暂时没有完成，请检查网络、模型配置或稍后重试。')

    @router.get('/status')
    def status():
        return {'configured': bool(api_key()), 'provider':'DeepSeek SVG', 'version':1,
            'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-flash')}

    @router.post('/books', status_code=202)
    def start(book: BookRequest):
        if len(book.captions) not in (1, 2, 4) or any(not c.strip() or len(c) > 200 for c in book.captions):
            raise HTTPException(422, '请选择1、2或4幅，并为每幅填写不超过200字的故事。')
        if not api_key():
            raise HTTPException(503, '老师还没有配置图像服务密钥。请先保存故事草稿，再按接入说明配置后端。')
        with lock:
            for key in list(jobs):
                if jobs[key]['status'] != 'generating' and time.time()-jobs[key]['created_at']>1800:
                    del jobs[key]
            if any(j['status']=='generating' for j in jobs.values()):
                raise HTTPException(409, '画室正在生成一本画本，请等它完成。')
            if jobs and time.time()-max(j['created_at'] for j in jobs.values())<30:
                raise HTTPException(429, '画室需要休息一下，请30秒后再试。')
            while len(jobs)>=5:
                del jobs[next(iter(jobs))]
            job_id = secrets.token_urlsafe(24)
            jobs[job_id] = {'id':job_id,'status':'generating','images':[],'captions':[],'total':len(book.captions),
                'created_at':time.time(),'_stop':threading.Event()}
            pool.submit(worker,job_id,book)
            return public(jobs[job_id])

    @router.get('/books/{job_id}')
    def progress(job_id: str):
        with lock:
            if job_id not in jobs:
                raise HTTPException(404, '画本生成记录已过期，请查看已保存的作品。')
            return public(jobs[job_id])

    @router.post('/books/{job_id}/stop')
    def stop(job_id: str):
        with lock:
            if job_id not in jobs:
                raise HTTPException(404, '这次生成记录已过期。')
            jobs[job_id]['_stop'].set()
            return {'requested':True, 'message':'当前正在生成的一幅可能仍会完成并产生用量，之后停止。'}

    return router
