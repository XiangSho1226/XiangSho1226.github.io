"""DeepSeek converts children's instructions into a bounded block language, never JS."""
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from picturebook import api_key


class CodeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1, max_length=1200)
    actor: Literal['cat','robot','bunny','dragon'] = 'cat'
    scene: Literal['forest','space','ocean'] = 'forest'
    previous: list = Field(default_factory=list, max_length=60)


def validate_program(program):
    total = 0
    specs={'move':('value',-100,100),'turn':('value',-360,360),'jump':('value',0,60),
        'wait':('seconds',0,3),'repeat':('times',1,10)}
    allowed=set(specs)|{'say','if_edge','if_star','collect','pen','color'}
    def walk(items,depth=0):
        nonlocal total
        if not isinstance(items,list) or depth>4:
            raise ValueError('积木嵌套太深了，最多4层。')
        result=[];cost=0
        for node in items:
            total+=1
            if total>60 or not isinstance(node,dict) or node.get('kind') not in allowed:
                raise ValueError('最多60块积木，且只能使用工具箱中的指令。')
            kind=node['kind'];clean={'kind':kind};keys={'kind'}
            if kind in specs:
                field,low,high=specs[kind];value=node.get(field)
                if type(value) not in (int,float) or not low<=value<=high or (kind=='repeat' and type(value)!=int):
                    raise ValueError('积木的数值超出了课堂允许的范围。')
                clean[field]=value;keys.add(field)
            if kind=='say':
                if not isinstance(node.get('text'),str) or len(node['text'])>80:
                    raise ValueError('说话积木最多80个字。')
                clean['text']=node['text'];keys.add('text')
            if kind=='pen':
                if type(node.get('down')) is not bool:
                    raise ValueError('画笔需要选择落下或抬起。')
                clean['down']=node['down'];keys.add('down')
            if kind=='color':
                import re
                if not isinstance(node.get('color'),str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',node['color']):
                    raise ValueError('请使用六位颜色值。')
                clean['color']=node['color'];keys.add('color')
            child_cost=0
            if kind in ('repeat','if_edge','if_star'):
                clean['body'],child_cost=walk(node.get('body',[]),depth+1);keys.add('body')
            if set(node)-keys:
                raise ValueError('这块积木包含未支持的内容，请重新描述。')
            cost+=1+child_cost*(clean['times'] if kind=='repeat' else 1)
            if cost>400:
                raise ValueError('运行次数太多了，减少循环次数再试吧。')
            result.append(clean)
        return result,cost
    result,cost=walk(program)
    return result,cost


def generate(request):
    endpoint=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
    parsed=urlparse(endpoint)
    if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Invalid service URL')
    system='''You teach children programming through graphical blocks. Return only JSON:
{"title":"short Chinese project title","explanation":"Chinese explanation under 350 characters: what the blocks do and any unsupported request","program":[...]}
Never return executable code, HTML, URLs, markdown, variables or unlisted commands.
Allowed blocks with EXACT fields:
{"kind":"move","value":40} (-100..100 steps, 1 step=4 pixels in current heading)
{"kind":"turn","value":90} (-360..360 degrees; positive clockwise)
{"kind":"jump","value":10} (0..60 steps forward with jump animation)
{"kind":"say","text":"你好"} (at most 80 characters)
{"kind":"wait","seconds":0.5} (0..3 seconds)
{"kind":"repeat","times":4,"body":[...]} (1..10 integer)
{"kind":"if_edge","body":[...]} (execute if actor is at edge)
{"kind":"if_star","body":[...]} (execute if actor touches an uncollected star)
{"kind":"collect"} (collect a touching star only)
{"kind":"pen","down":true} (draw movement trail or lift pen with false)
{"kind":"color","color":"#8a6ab6"} (trail color)
Stage 640x400 pixels; actor starts (80,200), heading right (0 degrees). Stars at (240,200),(480,100),(150,80).
Actor stays 20px inside stage. Movement clamps at boundary. No keyboard events, random, physics or forever loops are supported.
Use at most 60 blocks, 4 nested levels, 400 expanded instructions, and keep waits short (runtime cap30s).
If modifying a previous program, preserve existing behavior unless requested. Keep content friendly and suitable for children.
Prefer simple readable blocks; explain limitations rather than pretending unsupported behavior works.'''
    payload={'model':os.getenv('DEEPSEEK_MODEL','deepseek-flash'),'thinking':{'type':'disabled'},
        'response_format':{'type':'json_object'},'max_tokens':4096,'messages':[
            {'role':'system','content':system},{'role':'user','content':json.dumps(request.model_dump(),ensure_ascii=False)}]}
    http=urllib.request.Request(endpoint+'/chat/completions',data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+api_key(),'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(http,timeout=90) as response:
        data=json.loads(response.read(2000000))
    choice=data['choices'][0]
    if choice.get('finish_reason')=='length':
        raise ValueError('AI回复太长了，请把想法说得短一些。')
    result=json.loads(choice['message']['content'])
    program,cost=validate_program(result['program'])
    title=result.get('title','我的小程序');explanation=result.get('explanation','')
    if not isinstance(title,str) or len(title)>60 or not isinstance(explanation,str) or len(explanation)>400:
        raise ValueError('AI返回格式不完整，请重新描述。')
    return {'title':title,'explanation':explanation,'program':program,'cost':cost,'actor':request.actor,'scene':request.scene}


def create_router(authorized):
    router=APIRouter(prefix='/api/aicode',dependencies=[Depends(authorized)])
    jobs={};lock=threading.RLock();pool=ThreadPoolExecutor(max_workers=1)
    def worker(job_id,request):
        try:
            result=generate(request)
            with lock:jobs[job_id].update(status='done',result=result)
        except urllib.error.HTTPError as exc:
            with lock:jobs[job_id].update(status='error',message={401:'DeepSeek密钥无效，请老师检查配置。',402:'DeepSeek余额不足，请老师检查账户用量。',429:'AI现在有些忙，请稍后再试。'}.get(exc.code,'AI暂时没完成请求，可以换个说法再试。'))
        except ValueError as exc:
            message=str(exc) if str(exc).startswith(('积木','最多','运行','AI回复','AI返回','这块','说话','画笔','请使用')) else 'AI返回的积木没有通过检查，请换一种描述再试。'
            with lock:jobs[job_id].update(status='error',message=message)
        except Exception:
            with lock:jobs[job_id].update(status='error',message='暂时连接不上AI，请检查网络后重试。')
    @router.get('/status')
    def status():return {'configured':bool(api_key()),'provider':'DeepSeek blocks','version':1}
    @router.post('/generate',status_code=202)
    def start(request:CodeRequest):
        if not request.prompt.strip():raise HTTPException(422,'先告诉AI你想让角色做什么。')
        try:validate_program(request.previous)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        if not api_key():raise HTTPException(503,'老师还没有配置DeepSeek，先用工具箱拼积木也可以。')
        with lock:
            if any(j['status']=='generating' for j in jobs.values()):raise HTTPException(409,'AI正在搭另一份积木，请稍后再试。')
            while len(jobs)>=12:del jobs[next(iter(jobs))]
            job_id=secrets.token_urlsafe(24);jobs[job_id]={'id':job_id,'status':'generating'}
            pool.submit(worker,job_id,request)
            return jobs[job_id].copy()
    @router.get('/jobs/{job_id}')
    def progress(job_id:str):
        with lock:
            if job_id not in jobs:raise HTTPException(404,'这次生成记录已过期，请重试。')
            return jobs[job_id].copy()
    return router
