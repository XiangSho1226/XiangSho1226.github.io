"""Small, single-worker CNN classroom lab. Run with: python main.py"""
from __future__ import annotations

import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent.parent
API_TOKEN = os.getenv('CNN_API_TOKEN', '')
app = FastAPI(title='小小 CNN 实验室', version='1.0.0')
app.add_middleware(CORSMiddleware,
    allow_origins=[s.strip() for s in os.getenv('CNN_ORIGINS',
        'https://xiangsho1226.github.io,http://127.0.0.1:8000,http://localhost:8000').split(',') if s.strip()],
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-Lab-Token'])


def authorized(x_lab_token: str = Header(default='')):
    if API_TOKEN and not secrets.compare_digest(x_lab_token, API_TOKEN):
        raise HTTPException(401, '老师设置了连接口令，请在连接设置中填写。')


class Layer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['conv', 'relu', 'pool', 'flatten', 'dense']
    channels: Literal[4, 8, 16] = 8
    kernel: Literal[3, 5] = 3
    padding: Literal['same', 'valid'] = 'same'
    units: Literal[10, 16, 32] = 16


class Design(BaseModel):
    model_config = ConfigDict(extra='forbid')
    layers: list[Layer] = Field(min_length=1, max_length=12)


class TrainRequest(Design):
    epochs: Literal[1, 3, 5, 8] = 5
    seed: int = Field(default=42, ge=0, le=9999)


class Prediction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pixels: list[float] = Field(min_length=64, max_length=64)


def build(design: Design):
    shape = [1, 8, 8]
    modules, trace, count = [], [], 0
    for index, layer in enumerate(design.layers):
        before = shape.copy()
        prefix = f'第 {index + 1} 块积木：'
        if layer.kind == 'conv':
            if len(shape) != 3:
                raise ValueError(prefix + '卷积需要一张图片，请把它放在展开积木之前。')
            pad = layer.kernel // 2 if layer.padding == 'same' else 0
            size = shape[1] + 2 * pad - layer.kernel + 1
            if size < 1:
                raise ValueError(prefix + '图片比卷积核还小，换成保留边缘，或减少缩小操作。')
            module = nn.Conv2d(shape[0], layer.channels, layer.kernel, padding=pad)
            shape = [layer.channels, size, size]
        elif layer.kind == 'pool':
            if len(shape) != 3 or shape[1] < 2:
                raise ValueError(prefix + '池化需要至少 2×2 的图片，不能继续缩小啦。')
            module = nn.MaxPool2d(2)
            shape = [shape[0], shape[1] // 2, shape[2] // 2]
        elif layer.kind == 'flatten':
            if len(shape) != 3:
                raise ValueError(prefix + '已经展开过了，保留一块展开积木就好。')
            module = nn.Flatten()
            shape = [int(np.prod(shape))]
        elif layer.kind == 'dense':
            if len(shape) != 1:
                raise ValueError(prefix + '做判断前，先用展开积木把图片排成一队。')
            module = nn.Linear(shape[0], layer.units)
            shape = [layer.units]
        else:
            module = nn.ReLU()
        count += sum(p.numel() for p in module.parameters())
        if count > 100000:
            raise ValueError(prefix + '需要学习的参数太多啦，试试加池化或减少通道数。')
        modules.append(module)
        trace.append({'kind': layer.kind, 'input': before, 'output': shape.copy()})
    if design.layers[-1].kind != 'dense' or shape != [10]:
        raise ValueError('最后需要一块“全连接”积木，并把输出数设为 10，分别判断数字 0–9。')
    if not any(layer.kind == 'conv' for layer in design.layers):
        raise ValueError('这节课要搭 CNN，请至少加入一块卷积积木。')
    return nn.Sequential(*modules), trace, count


digits = load_digits()
images = torch.tensor(digits.images[:, None] / 16.0, dtype=torch.float32)
labels = torch.tensor(digits.target, dtype=torch.long)
train_ids, test_ids = train_test_split(np.arange(len(labels)), test_size=0.2,
    stratify=digits.target, random_state=42)
sample_ids = [int(next(i for i in test_ids if int(labels[i]) == digit)) for digit in range(10)]
jobs: dict[str, dict] = {}
lock = threading.RLock()
executor = ThreadPoolExecutor(max_workers=1)


def public_job(job):
    return {key: value for key, value in job.items() if not key.startswith('_')}


def get_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, '这个实验已过期，请重新训练。')
    return job


def prune():
    now = time.time()
    for key in list(jobs):
        if jobs[key]['status'] not in ('queued', 'training') and now - jobs[key]['created_at'] > 1800:
            del jobs[key]
    completed = [key for key in jobs if jobs[key]['status'] not in ('queued', 'training')]
    while len(jobs) >= 12 and completed:
        del jobs[completed.pop(0)]


def run_training(job_id: str, request: TrainRequest):
    job = jobs[job_id]
    try:
        torch.manual_seed(request.seed)
        model, trace, parameters = build(request)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(request.seed)
        x_train, y_train = images[train_ids], labels[train_ids]
        x_test, y_test = images[test_ids], labels[test_ids]
        start = time.monotonic()
        with lock:
            job.update(status='training', trace=trace, parameters=parameters)
        for epoch in range(request.epochs):
            model.train()
            indices = torch.randperm(len(x_train), generator=generator)
            total_loss, seen = 0.0, 0
            for ids in indices.split(64):
                if job['_cancel'].is_set() or time.monotonic() - start > 90:
                    with lock:
                        job.update(status='cancelled', message='实验已停止，可以修改积木后再试。')
                    return
                optimizer.zero_grad()
                loss = loss_fn(model(x_train[ids]), y_train[ids])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(ids)
                seen += len(ids)
            model.eval()
            with torch.inference_mode():
                accuracy = float((model(x_test).argmax(1) == y_test).float().mean())
            metric = {'epoch': epoch + 1, 'loss': round(total_loss / seen, 4),
                'accuracy': round(accuracy, 4)}
            with lock:
                job['metrics'] = [*job['metrics'], metric]
        with lock:
            if job['_cancel'].is_set():
                job.update(status='cancelled', message='实验已停止。')
            else:
                job.update(status='done', _model=model, elapsed=round(time.monotonic() - start, 2))
    except Exception:
        with lock:
            job.update(status='error', message='实验没有完成，请检查积木组合后重试。')
        import logging
        logging.exception('CNN experiment failed')


@app.get('/api/health', dependencies=[Depends(authorized)])
def health():
    return {'ok': True, 'engine': 'PyTorch', 'version': '1', 'dataset': 'sklearn digits',
        'shape': [1, 8, 8], 'train_samples': len(train_ids), 'test_samples': len(test_ids)}


@app.get('/api/samples', dependencies=[Depends(authorized)])
def samples():
    return {'samples': [{'label': int(labels[i]), 'pixels': images[i].flatten().tolist()} for i in sample_ids]}


@app.post('/api/validate', dependencies=[Depends(authorized)])
def validate(design: Design):
    try:
        _, trace, parameters = build(design)
        return {'valid': True, 'trace': trace, 'parameters': parameters}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/train', status_code=202, dependencies=[Depends(authorized)])
def train(request: TrainRequest):
    try:
        build(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with lock:
        prune()
        if any(job['status'] in ('queued', 'training') for job in jobs.values()):
            raise HTTPException(409, '实验室正在训练另一个网络，请稍后再试。')
        job_id = secrets.token_urlsafe(24)
        jobs[job_id] = {'id': job_id, 'status': 'queued', 'metrics': [],
            'created_at': time.time(), 'epochs': request.epochs,
            'design': request.model_dump(), '_cancel': threading.Event()}
        executor.submit(run_training, job_id, request)
        return public_job(jobs[job_id])


@app.get('/api/jobs/{job_id}', dependencies=[Depends(authorized)])
def job_status(job_id: str):
    with lock:
        prune()
        return public_job(get_job(job_id))


@app.post('/api/jobs/{job_id}/cancel', dependencies=[Depends(authorized)])
def cancel(job_id: str):
    with lock:
        job = get_job(job_id)
        job['_cancel'].set()
        return {'requested': True}


@app.post('/api/jobs/{job_id}/predict', dependencies=[Depends(authorized)])
def predict(job_id: str, request: Prediction):
    if any(not np.isfinite(p) or p < 0 or p > 1 for p in request.pixels):
        raise HTTPException(422, '像素值必须是 0 到 1 之间的数。')
    with lock:
        job = get_job(job_id)
        if job['status'] != 'done':
            raise HTTPException(409, '请先完成训练，再测试图片。')
        model = job['_model']
    tensor = torch.tensor(request.pixels, dtype=torch.float32).reshape(1, 1, 8, 8)
    features = []
    with torch.inference_mode():
        output = tensor
        for index, module in enumerate(model):
            output = module(output)
            if output.ndim == 4:
                features.append({'layer': index + 1, 'kind': job['design']['layers'][index]['kind'],
                    'maps': output[0, :4].tolist()})
        probabilities = output.softmax(1)[0].tolist()
    return {'digit': int(np.argmax(probabilities)), 'probabilities': probabilities, 'features': features}


@app.get('/')
def index():
    return FileResponse(ROOT / 'cnn.html')


@app.get('/cnn-backend/{asset}')
def course_download(asset: str):
    if asset not in ('使用说明.txt', '体验课源码.zip'):
        raise HTTPException(404, 'Not found')
    return FileResponse(Path(__file__).resolve().parent / asset)


@app.get('/{asset}')
def static_asset(asset: str):
    if asset not in ('cnn.html', 'cnn.css', 'cnn.js', 'cnn-core.js', 'style.css', 'app.js', 'index.html'):
        raise HTTPException(404, 'Not found')
    return FileResponse(ROOT / asset)


if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', 'localhost') and not API_TOKEN:
        parser.error('共享到其他设备前，请设置 CNN_API_TOKEN 连接口令。')
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
