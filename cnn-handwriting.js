'use strict';
window.CNNHandwriting=(()=>{
  let canvas,ctx,pixels=null,active=false,drawing=false,lastPoint=null,source='手写画板',generation=0,ready=false,notify=()=>{},recognize=()=>{};
  function sample(){return active?{label:null,pixels:pixels||Array(64).fill(0),source}:null}
  function hasInk(){return !!pixels?.some(x=>x>0.05)}
  function controls(){const b=document.querySelector('#recognize-handwriting');b.disabled=!ready||!hasInk();b.textContent=ready?'识别我写的数字':'先训练网络，再识别';}
  function changed(){controls();notify();}
  function process(){const data=ctx.getImageData(0,0,canvas.width,canvas.height);pixels=CNNCore.normalizeInk(data.data,canvas.width,canvas.height);active=true;document.querySelector('#handwriting-status').textContent=pixels?'已经整理成 8×8 数字图片，可以观察或识别。':'画板是空的，写一个大大的数字 0–9 吧。';changed()}
  function reset(){generation++;drawing=false;lastPoint=null;ctx.fillStyle='#171b2d';ctx.fillRect(0,0,canvas.width,canvas.height);pixels=null;active=true;source='手写画板';document.querySelector('#handwriting-status').textContent='写一个大大的数字 0–9，只写一个哦。';changed()}
  function point(e){const rect=canvas.getBoundingClientRect();return {x:(e.clientX-rect.left)*canvas.width/rect.width,y:(e.clientY-rect.top)*canvas.height/rect.height}}
  function ink(p,q){ctx.strokeStyle='#fff';ctx.fillStyle='#fff';ctx.lineWidth=21;ctx.lineCap='round';ctx.lineJoin='round';if(!q){ctx.beginPath();ctx.arc(p.x,p.y,10.5,0,Math.PI*2);ctx.fill()}else{ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(p.x,p.y);ctx.stroke()}}
  function init(options){
    notify=options.onChange;recognize=options.onRecognize;canvas=document.querySelector('#handwriting-canvas');ctx=canvas.getContext('2d',{willReadFrequently:true});reset();active=false;
    canvas.onpointerdown=e=>{if(e.button!==0&&e.pointerType==='mouse')return;e.preventDefault();generation++;if(source!=='手写画板')reset();drawing=true;source='手写画板';active=true;pixels=null;canvas.setPointerCapture(e.pointerId);lastPoint=point(e);ink(lastPoint);changed();};
    canvas.onpointermove=e=>{if(!drawing)return;e.preventDefault();const p=point(e);ink(p,lastPoint);lastPoint=p};
    const finish=e=>{if(!drawing)return;drawing=false;lastPoint=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);process()};
    canvas.onpointerup=finish;canvas.onpointercancel=finish;canvas.onlostpointercapture=finish;
    document.querySelector('#clear-handwriting').onclick=reset;
    document.querySelector('#recognize-handwriting').onclick=()=>{if(!ready||!hasInk())return;active=true;changed();recognize()};
    document.querySelector('#upload-handwriting-button').onclick=()=>document.querySelector('#upload-handwriting').click();
    document.querySelector('#upload-handwriting').onchange=async e=>{
      const file=e.target.files[0];e.target.value='';if(!file)return;
      const status=document.querySelector('#handwriting-status');
      if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>5*1024*1024){status.textContent='请选择不超过 5 MB 的 PNG、JPG 或 WebP 图片。';return}
      const stamp=++generation,url=URL.createObjectURL(file);status.textContent='正在整理图片…';
      try{const image=new Image();image.src=url;await image.decode();if(stamp!==generation)return;if(image.naturalWidth*image.naturalHeight>16000000)throw Error('图片尺寸太大，请先缩小到 1600 万像素以内。');ctx.fillStyle='#fff';ctx.fillRect(0,0,256,256);const scale=Math.min(256/image.naturalWidth,256/image.naturalHeight),w=image.naturalWidth*scale,h=image.naturalHeight*scale;ctx.drawImage(image,(256-w)/2,(256-h)/2,w,h);source='上传的数字图片';process();if(!pixels)status.textContent='没有找到清楚的笔迹，请选择白底、深色笔迹的单个数字图片。'}catch(err){if(stamp===generation)status.textContent=err.message.startsWith('图片尺寸')?err.message:'图片没有读取成功，请换一张清晰的单个数字图片。'}finally{URL.revokeObjectURL(url)}
    };
    controls();
  }
  return {init,sample,hasInk,setReady(value){ready=value;controls()},useExamples(){active=false;controls()}};
})();
