(function(root){
  'use strict';
  const defaults=()=>[{kind:'conv',channels:8,kernel:3,padding:'same'},{kind:'relu'},{kind:'pool'},{kind:'flatten'},{kind:'dense',units:10}];
  const names={conv:'卷积 · 找特征',relu:'激活 · 留下亮点',pool:'池化 · 缩小图',flatten:'展开 · 排成队',dense:'全连接 · 做判断'};
  function normalize(raw){
    if(!Array.isArray(raw)||raw.length<1||raw.length>12)throw Error('方案需要 1–12 块积木。');
    return raw.map(l=>{
      if(!l||!Object.hasOwn(names,l.kind))throw Error('方案包含不认识的积木。');
      if(l.kind==='conv'){
        if(![4,8,16].includes(l.channels)||![3,5].includes(l.kernel)||!['same','valid'].includes(l.padding))throw Error('卷积参数不在课堂允许的范围内。');
        return {kind:l.kind,channels:l.channels,kernel:l.kernel,padding:l.padding};
      }
      if(l.kind==='dense'){
        if(![10,16,32].includes(l.units))throw Error('全连接输出数需要为 10、16 或 32。');
        return {kind:l.kind,units:l.units};
      }
      return {kind:l.kind};
    });
  }
  function analyze(layers){
    let shape=[1,8,8],parameters=0;const trace=[];
    try{
      for(let i=0;i<layers.length;i++){
        const l=layers[i],input=shape.slice();let error='';
        if(l.kind==='conv'){
          if(shape.length!==3)error='卷积需要一张图片，请把它放在展开积木之前。';
          else {const n=shape[1]+(l.padding==='same'?l.kernel-1:0)-l.kernel+1;if(n<1)error='图片比卷积核还小，换成保留边缘，或减少缩小操作。';else {parameters+=l.channels*(shape[0]*l.kernel*l.kernel+1);shape=[l.channels,n,n];}}
        }else if(l.kind==='pool'){
          if(shape.length!==3||shape[1]<2)error='池化需要至少 2×2 的图片，不能继续缩小啦。';
          else shape=[shape[0],Math.floor(shape[1]/2),Math.floor(shape[2]/2)];
        }else if(l.kind==='flatten'){
          if(shape.length!==3)error='已经展开过了，保留一块展开积木就好。';
          else shape=[shape.reduce((a,b)=>a*b,1)];
        }else if(l.kind==='dense'){
          if(shape.length!==1)error='做判断前，先用展开积木把图片排成一队。';
          else {parameters+=(shape[0]+1)*l.units;shape=[l.units];}
        }else if(l.kind!=='relu')error='不认识这块积木。';
        if(parameters>100000)error='需要学习的参数太多啦，试试加池化或减少通道数。';
        trace.push({input,output:shape.slice(),error});
        if(error)return {valid:false,error:`第 ${i+1} 块：${error}`,trace,parameters};
      }
      if(!layers.length||layers.at(-1).kind!=='dense'||shape.length!==1||shape[0]!==10)return {valid:false,error:'最后需要一块全连接积木，输出数设为 10，分别判断数字 0–9。',trace,parameters};
      if(!layers.some(l=>l.kind==='conv'))return {valid:false,error:'这节课要搭 CNN，请至少加入一块卷积积木。',trace,parameters};
      return {valid:true,trace,parameters,error:''};
    }catch{return {valid:false,trace,parameters,error:'参数不完整，请恢复示例或重新导入方案。'}}
  }
  function preview(pixels,layers,until){
    let maps=[Array.from({length:8},(_,y)=>pixels.slice(y*8,y*8+8))];
    for(let i=0;i<=until&&i<layers.length;i++){
      const l=layers[i];
      if(l.kind==='flatten')return {vector:maps.flat(2)};
      if(l.kind==='dense')return {untrained:true};
      if(l.kind==='relu')maps=maps.map(m=>m.map(row=>row.map(x=>Math.max(0,x))));
      if(l.kind==='pool'){
        const n=Math.floor(maps[0].length/2);if(n<1)return {untrained:true};
        maps=maps.map(m=>Array.from({length:n},(_,y)=>Array.from({length:n},(_,x)=>Math.max(m[y*2][x*2],m[y*2+1][x*2],m[y*2][x*2+1],m[y*2+1][x*2+1]))));
      }
      if(l.kind==='conv'){
        const input=maps,n=input[0].length,k=l.kernel,p=l.padding==='same'?Math.floor(k/2):0,size=n+2*p-k+1;
        if(size<1)return {untrained:true};
        maps=Array.from({length:l.channels},(_,channel)=>Array.from({length:size},(_,y)=>Array.from({length:size},(_,x)=>{
          let value=0;for(const m of input)for(let ky=0;ky<k;ky++)for(let kx=0;kx<k;kx++){
            const yy=y+ky-p,xx=x+kx-p;if(yy<0||xx<0||yy>=n||xx>=n)continue;
            const center=Math.floor(k/2);let weight;
            switch(channel%4){case 0:weight=kx===center?1:-(1/(k-1));break;case 1:weight=ky===center?1:-(1/(k-1));break;case 2:weight=1/(k*k);break;default:weight=(kx===center&&ky===center)?1:-(1/(k*k-1));}
            value+=m[yy][xx]*weight/input.length;
          }return value;
        })));
      }
    }
    return {maps};
  }
  function normalizeInk(rgba,width,height){
    if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||rgba.length!==width*height*4)throw Error('图片数据尺寸不正确。');
    const gray=new Float32Array(width*height);let border=0,borderCount=0,min=1,max=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*width+x,a=rgba[i*4+3]/255;const g=(rgba[i*4]*.299+rgba[i*4+1]*.587+rgba[i*4+2]*.114)/255*a+1-a;gray[i]=g;min=Math.min(min,g);max=Math.max(max,g);if(x===0||y===0||x===width-1||y===height-1){border+=g;borderCount++}}
    if(max-min<.12)return null;
    const invert=border/borderCount>.5;let left=width,right=-1,top=height,bottom=-1,count=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*width+x;gray[i]=Math.max(0,((invert?1-gray[i]:gray[i])-.12)/.88);if(gray[i]>.15){left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);count++}}
    if(right<left||count<8)return null;
    const bw=right-left+1,bh=bottom-top+1,scale=Math.min(6/bw,8/bh),dw=bw*scale,dh=bh*scale,ox=(8-dw)/2,oy=(8-dh)/2,pixels=Array(64).fill(0);
    for(let y=0;y<8;y++)for(let x=0;x<8;x++){
      const x0=Math.max(left,left+(x-ox)/scale),x1=Math.min(right+1,left+(x+1-ox)/scale),y0=Math.max(top,top+(y-oy)/scale),y1=Math.min(bottom+1,top+(y+1-oy)/scale);
      if(x1<=x0||y1<=y0)continue;let sum=0;
      for(let sy=Math.floor(y0);sy<Math.ceil(y1);sy++)for(let sx=Math.floor(x0);sx<Math.ceil(x1);sx++){const area=(Math.min(sx+1,x1)-Math.max(sx,x0))*(Math.min(sy+1,y1)-Math.max(sy,y0));sum+=gray[sy*width+sx]*area}
      pixels[y*8+x]=Math.min(1,Math.max(0,sum*scale*scale));
    }
    const peak=Math.max(...pixels);return peak>.05?pixels.map(x=>x/peak):null;
  }
  const api={defaults,names,normalize,analyze,preview,normalizeInk};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.CNNCore=api;
})(typeof globalThis!=='undefined'?globalThis:this);
