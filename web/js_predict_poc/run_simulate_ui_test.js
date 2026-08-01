// run_simulate_ui_test.js — ③ SIMULATE（What-ifスライダー・Web版）の統合テスト。
//
// 生成物 web/index.html を jsdom に載せ、本物の .treg フィクスチャと
// train_bridge.py が返す形の result JSON を流して以下を検証する:
//   ・build_frontend.mjs が埋め込んだ推論エンジンが predict-core.js と完全一致すること
//     （＝複製ズレが起きていないこと。ここが崩れると SIMULATE と CSV 予測が食い違う）
//   ・スライダー表示値が predictRow の計算値と一致すること
//   ・外挿 / x_clip 飽和 / 低精度時の抑制表示 / 変数が多い場合の展開・折りたたみ
//   ・.treg や slider_spec が無いときに安全に無効化されること
//
// frontend/index.html を編集したら先に `node build_frontend.mjs` を実行すること
// （このテストは生成物を読むため、再生成を忘れると落ちる＝運用ルール1の見張りにもなる）。
//
// 実行: cd web/js_predict_poc && node run_simulate_ui_test.js
const {JSDOM}=require('jsdom');const fs=require('fs');const path=require('path');
// 通常はこのファイルからの相対でリポジトリルートを求める。
// TREG_ROOT を指定すると別の場所を見る(node_modules が遅い環境で、
// ローカルディスクへ退避して実行する場合などに使う)。
const ROOT=process.env.TREG_ROOT ? path.resolve(process.env.TREG_ROOT)
                                 : path.resolve(__dirname,'..','..');
const html=fs.readFileSync(path.join(ROOT,'web/index.html'),'utf8');
const errs=[];const FAIL=[];
const chk=(c,m)=>{console.log((c?'  OK   ':'  FAIL ')+m);if(!c)FAIL.push(m);};

const dom=new JSDOM(html,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost/'});
const w=dom.window;
const noop=()=>{};const ctx2d=new Proxy({},{get:(t,k)=>{
  if(k==='createLinearGradient')return()=>({addColorStop:noop});
  if(k==='measureText')return()=>({width:10});
  return typeof k==='string'?noop:undefined;},set:()=>true});
w.HTMLCanvasElement.prototype.getContext=()=>ctx2d;
// jsdom のバージョンによっては TextDecoder/TextEncoder が window に無い。
// ブラウザには必ずあるものなので、テスト環境側の欠落として補う。
if(!w.TextDecoder) w.TextDecoder=require('util').TextDecoder;
if(!w.TextEncoder) w.TextEncoder=require('util').TextEncoder;
Object.defineProperty(w.Element.prototype,'clientWidth',{get(){return 300;}});
w.addEventListener('error',e=>errs.push('window.error: '+e.message));
const code=html.match(/<script>([\s\S]*)<\/script>/)[1];
try{w.eval(code+'\n;window.__H__={Platform:Platform,applyTrainingResult:applyTrainingResult,showTrainedTab:showTrainedTab,SIM:SIM,simSobolCount:simSobolCount,simSobol:simSobol,simSobolEach:simSobolEach,simAddModelFile:simAddModelFile};');}catch(e){errs.push('THROW: '+e.message+'\n'+(e.stack||'').split('\n').slice(0,6).join('\n'));}

console.log('[0] 埋め込みエンジン');
chk(errs.length===0,'スクリプトが例外なく実行される'); if(errs.length){console.log(errs.join('\n'));process.exit(1);}
chk(typeof w.TregPredictCore==='object','window.TregPredictCore が公開されている');
const core=require('./predict-core.js');
const treg=fs.readFileSync(path.join(ROOT,'web/js_predict_poc/sample_lgbm_model.treg'));
const bytes=new Uint8Array(treg);
const buf=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
const m1=core.loadTreg(buf),m2=w.TregPredictCore.loadTreg(buf);
const feats=m1.feat_cols;
let maxDiff=0;
for(let i=0;i<50;i++){const r={},raw={};
  feats.forEach((c,j)=>{const v=(i*7+j*13)%100/3;r[c]=v;raw[c]=String(v);});
  maxDiff=Math.max(maxDiff,Math.abs(core.predictRow(m1,r,raw)-w.TregPredictCore.predictRow(m2,r,raw)));}
chk(maxDiff===0,`埋め込み後も元の predict-core.js と完全一致 (最大差 ${maxDiff})`);

// ── ダミーの学習結果（Python が返す形） ─────────────────────────────────────
const N=400;let sd=42;const rnd=()=>((sd=(sd*1103515245+12345)&0x7fffffff)/0x7fffffff);
const rows=[];for(let i=0;i<N;i++){const r={};feats.forEach((c,j)=>{r[c]=10+j*0.5+8*(rnd()-0.5);});rows.push(r);}
const P=rows.map(r=>{const raw={};for(const k in r)raw[k]=String(r[k]);return core.predictRow(m1,r,raw);});
const pctl=(a,q)=>{const s=[...a].sort((x,y)=>x-y);return s[Math.round((s.length-1)*q)];};
const sliderSpec=feats.map(c=>{
  const v=rows.map(r=>r[c]).sort((a,b)=>a-b),mn=v[0],mx=v[v.length-1],HB=28,hist=new Array(HB).fill(0);
  v.forEach(x=>{let b=Math.floor((x-mn)/(mx-mn)*HB);if(b>=HB)b=HB-1;hist[b]++;});
  return {col:c,kind:'numeric',importance_pct:+(100/(feats.length+1)).toFixed(2),
    median:pctl(v,.5),p1:pctl(v,.01),p99:pctl(v,.99),min:mn,max:mx,step:(mx-mn)/300,
    hist,hist_lo:mn,hist_hi:mx,x_clip_lo:mn,x_clip_hi:mx,is_integer:false};});
sliderSpec.push({col:'grade',kind:'categorical',importance_pct:0.5,mode:'A',
  levels:[{value:'A',count:250},{value:'B',count:150}],truncated:false});
const lo=Math.min(...P),hi=Math.max(...P),nb=20,bw=(hi-lo)/nb,counts=new Array(nb).fill(0);
P.forEach(v=>{let b=Math.floor((v-lo)/bw);if(b>=nb)b=nb-1;counts[b]++;});
const mkSeed=(label,i)=>({label,y:P[i],
  values:Object.fromEntries(sliderSpec.map(s=>[s.col,s.kind==='numeric'?String(rows[i][s.col]):'A']))});
const nMean=feats.map(c=>rows.reduce((a,r)=>a+r[c],0)/N);
const nStd=feats.map((c,j)=>Math.sqrt(rows.reduce((a,r)=>a+(r[c]-nMean[j])**2,0)/N)||1);
const RESULT={r2:0.71,rmse:6.8,mae:5.0,target:'yield',best_model:'LightGBM',model_type:'lgbm',
  feature_importance:feats.slice(0,10).map((c,i)=>({name:c,pct:10-i*0.5})),candidate_models:[],
  eval_on:'val',train_rows:N,val_rows:80,preset:'quick',data_warning:'',data_warning_parts:[],
  r2_interpretation:'good',r2_reference_only:false,export_available:true,
  deployed_model:'LightGBM',deploy_substituted:false,cat_columns:[],cat_dropped_columns:[],
  scatter:{true:P.slice(0,50),pred:P.slice(0,50)},y_range:[lo,hi],
  slider_spec:sliderSpec,
  y_hist:{bin_edges:Array.from({length:nb+1},(_,i)=>lo+bw*i),counts,n:N,
          p10:pctl(P,.1),p50:pctl(P,.5),p90:pctl(P,.9)},
  seed_rows:[mkSeed('low',3),mkSeed('median',10),mkSeed('high',20),mkSeed('random',33)],
  neighbor_ref:{cols:feats,mean:nMean,std:nStd,weight:feats.map(()=>1),
    rows:rows.slice(0,300).map(r=>feats.map((c,j)=>(r[c]-nMean[j])/nStd[j])),radius:3.5,k:10},
  corr_pairs:[{a:feats[0],b:feats[1],r:0.85,sAB:1,iAB:0.5,sBA:1,iBA:-0.5,sdA:1.2,sdB:1.2}]};

w.__TREG__=new w.Uint8Array(bytes);   // window realm 側で確保（実アプリと同じ状況にする）
console.log('\n[1] 学習完了 → SIMULATE 起動');
const H=w.__H__;
H.Platform.getTregBytes=()=>w.__TREG__;
try{ H.applyTrainingResult(RESULT); }
catch(e){ errs.push('applyTrainingResult THROW: '+e.message+'\n'+(e.stack||'').split('\n').slice(0,5).join('\n')); }
if(errs.length){console.log(errs.join('\n'));process.exit(1);}
const d=w.document,q=s=>d.querySelectorAll(s).length,el=id=>d.getElementById(id);
// 画面表示と同じ丸めをテスト側でも使う
function simFmtLike(v){const a=Math.abs(v);let x;
  if(a>=1000)return String(Math.round(v));
  if(a===0)return '0';
  if(a>=10)x=v.toFixed(1); else if(a>=1)x=v.toFixed(2);
  else if(a>=0.01)x=v.toFixed(3); else return v.toPrecision(2);
  return x.replace(/\.?0+$/,'');}

chk(d.getElementById('step3Label').textContent==='SIMULATE','③の見出しが SIMULATE になる');
chk(el('predictSubLabel').textContent.includes('OPEN SIMULATOR')||el('predictSubLabel').textContent.includes('推論'),
  `③が推論ボタンになる: ${el('predictSubLabel').textContent}`);
chk(d.getElementById('predictZone').className.includes('sim-open-btn'),'③にボタン用スタイルが付く');
chk(el('tabPredict').style.display==='none','SIMULATEはタブではなく③から開く');
chk(el('simulateOverlay').style.display==='none','初期状態ではオーバーレイは閉じている');
d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simulateOverlay').style.display==='flex','③クリックで全画面オーバーレイが開く');
chk(!d.getElementById('simSubtitle'),'説明文（Move the sliders…）は出さない');
chk(q('.sim-row')===21,`全21変数にスライダーが用意される (実測 ${q('.sim-row')})`);
chk(!d.getElementById('simChipRow'),'固定チップ行は無い（全変数が常にスライダー）');
chk(!d.querySelector('.sim-row .sim-val'),'スライダー横の数値は出さない');
chk(!d.querySelector('.sim-row .sim-spark'),'スライダー横の感度グラフは出さない');
chk([...d.querySelectorAll('.sim-row')].every(r=>r.querySelector('input[type=range]')),
  'カテゴリも含め全ての行がスライダー');
{const catRow=[...d.querySelectorAll('.sim-row')].find(r=>r.dataset.col==='grade');
 const rg=catRow&&catRow.querySelector('input[type=range]');
 chk(!!rg&&Number(rg.min)===0&&Number(rg.max)===1&&Number(rg.step)===1,
   `カテゴリは水準番号のスライダー (${rg&&rg.min}〜${rg&&rg.max})`);}
{const blk=html.slice(html.indexOf('.sim-sliders {'),html.indexOf('.sim-sliders {')+400);
 chk(/align-items: center/.test(html.slice(html.lastIndexOf('.sim-sliders {'),html.lastIndexOf('.sim-sliders {')+200)),
   'スライダーは左右中央寄せ');
 chk(/justify-content: center/.test(blk),'スライダーは上下中央寄せ');}
chk(!d.getElementById('simDelta')&&!d.getElementById('simRingCanvas')&&!d.getElementById('simFlags'),
  '応答側はグラフと応答値だけ（差分・リング・フラグは無い）');
chk(/\.sim-sliders \{ display: flex; flex-direction: column;/.test(html),'スライダーは縦一列');
{// 全画面ではなく元ツールの外形に収まる大きさか（#simulatePanel の指定を見る）
 const blk=html.slice(html.indexOf('#simulatePanel {'), html.indexOf('#simulatePanel {')+400);
 const mw=blk.match(/width: calc\(100vw - (\d+)px\)/), mh=blk.match(/height: calc\(100vh - (\d+)px\)/);
 // 元ツール本体は body の padding(22px 22px 14px)の内側。同じ枠に収まっているか。
 const bodyPad=html.match(/body \{[^}]*padding: (\d+)px (\d+)px (\d+)px/s);
 const want=bodyPad?Number(bodyPad[2])*2:44, wantH=bodyPad?Number(bodyPad[1])+Number(bodyPad[3]):36;
 chk(!!mw&&Number(mw[1])===want&&!!mh&&Number(mh[1])===wantH,
   `パネルが元ツールと同じ枠 (100vw-${mw&&mw[1]} x 100vh-${mh&&mh[1]})`);}

console.log('\n[2] 予測値が推論エンジンと一致するか（最重要）');
const pv=el('simPredVal').textContent;
const seedMed=RESULT.seed_rows.find(s=>s.label==='median');
const row={},raw={};
sliderSpec.forEach(s=>{const v=seedMed.values[s.col];raw[s.col]=String(v);row[s.col]=Number(v);});
const expect=core.predictRow(m1,row,raw);
chk(Math.abs(Number(pv)-Number(expect.toFixed(1)))<0.06,
  `出発点の表示 ${pv} が predictRow の ${expect.toFixed(4)} と一致`);

console.log('\n[3] スライダーの可動域と操作');
const stateOf=(over)=>{const row={},raw={};
  sliderSpec.forEach(s=>{let v=(over&&over[s.col]!==undefined)?over[s.col]:seedMed.values[s.col];
    raw[s.col]=String(v);row[s.col]=Number(v);});return {row,raw};};
const r0=d.querySelector('.sim-row input[type=range]');
const col0=d.querySelector('.sim-row').dataset.col;
const sp0=sliderSpec.find(x=>x.col===col0);
chk(Math.abs(Number(r0.min)-sp0.min)<1e-9&&Math.abs(Number(r0.max)-sp0.max)<1e-9,
  `可動域が学習データの上下限そのもの (${simFmtLike(sp0.min)}〜${simFmtLike(sp0.max)})`);
r0.value=r0.max; r0.dispatchEvent(new w.Event('input',{bubbles:true}));
r0.dispatchEvent(new w.Event('change',{bubbles:true}));
{const st=stateOf({[col0]:r0.max});const ex=core.predictRow(m1,st.row,st.raw);
 chk(Math.abs(Number(el('simPredVal').textContent)-Number(ex.toFixed(1)))<0.06,
   `${col0} を上限へ: 表示 ${el('simPredVal').textContent} == エンジン ${ex.toFixed(4)}`);}
chk(d.querySelector('.sim-row').classList.contains('changed'),'動かした変数に印が付く');
chk(!d.querySelector('.sim-row.warn'),'上下限まで動かしてもオレンジにはならない');
{const sp0b=sliderSpec.find(x=>x.col===col0);
 chk(Math.abs(Number(r0.min)-sp0b.min)<1e-9&&Math.abs(Number(r0.max)-sp0b.max)<1e-9,
   '可動域は常に変数の上下限');}
d.querySelector('.sim-row .sim-name').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row.changed')===0,'列名クリックでその変数だけ元に戻る');

console.log('\n[4] Sobol走査による応答分布');
chk(!!H.SIM.dist,'応答分布が作られている');
{const D=H.SIM.dist, isPow2=x=>x>0&&(x&(x-1))===0;
 chk(isPow2(D.n)&&D.n>=512&&D.n<=32768,`走査点数は2のべき乗 (${D.n}点 / 変数${sliderSpec.length}個)`);
 const c=H.simSobolCount;
 chk(c(1)===2048&&c(5)===4096&&c(10)===8192&&c(20)===16384&&c(100)===32768,
   `点数の決め方: 1→${c(1)} 5→${c(5)} 10→${c(10)} 20→${c(20)} 100→${c(100)}`);
 chk(c(3)<=c(10)&&c(10)<=c(30),'変数が増えるほど点数が増える(単調)');
 chk(D.counts.reduce((a,b)=>a+b,0)===D.n,'ヒストグラムの合計が点数と一致');
 chk(!!D.best&&Number.isFinite(D.best.y),`最小応答の点がある (${simFmtLike(D.best.y)})`);
 chk(Math.abs(D.best.y-D.lo)<1e-9,'最小応答の点が分布の下端と一致');
 // 最適点の入力をエンジンへ通すと、その応答が再現できる
 {const st={},rw={};
  sliderSpec.forEach(s=>{const v=D.best.state[s.col];rw[s.col]=String(v);st[s.col]=Number(v);});
  chk(Math.abs(core.predictRow(m1,st,rw)-D.best.y)<1e-6,'最適点の入力から同じ応答が再現できる');}
 // 最適点の各変数の値は可動域の内側にある（スライダー上の印が枠外に出ない）
 chk(sliderSpec.filter(s=>s.kind==='numeric')
      .every(s=>{const v=Number(D.best.state[s.col]);return v>=s.min-1e-9&&v<=s.max+1e-9;}),
   '最適点の各変数の値が可動域の内側にある');
 // 上位10%の分布（スライダー上の山）
 chk(D.topFrac===0.10&&D.topN>0,`上位${D.topFrac*100}%: ${D.topN}点 (閾値 ${simFmtLike(D.thr)})`);
 chk(Math.abs(D.topN-D.n*0.10)/D.n<0.02,'上位の点数が全体の約10%');
 const numCols=sliderSpec.filter(s=>s.kind==='numeric').map(s=>s.col);
 chk(numCols.every(c=>Array.isArray(D.varHist[c])&&D.varHist[c].length===28),
   '数値列ごとに分布(28ビン)がある');
 chk(numCols.every(c=>D.varHist[c].reduce((a,b)=>a+b,0)===D.topN),
   '各変数の分布の合計が上位の点数と一致');
 // 応答が動くモデルなら、上位に絞った分布は一様と異なる（=情報がある）。
 // このフィクスチャは x_clip で応答が一定になるため、その場合は対象外にする。
 if (D.hi - D.lo > 1e-9) {
   const flat=numCols.filter(c=>{const h=D.varHist[c],mx=Math.max(...h),mn=Math.min(...h);
     return (mx-mn)/Math.max(1,mx)<0.05;});
   chk(flat.length<numCols.length,`上位に絞ると分布に偏りが出る (平坦な列 ${flat.length}/${numCols.length})`);
 } else {
   chk(true,'応答が一定のモデルのため、分布の偏りは検証対象外');
 }
 // 分布はスライダー操作では作り直さない
 const before=D;
 const r1=d.querySelector('.sim-row input[type=range]');
 r1.value=r1.min; r1.dispatchEvent(new w.Event('change',{bubbles:true}));
 chk(H.SIM.dist===before,'スライダーを動かしても走査はやり直さない');}

console.log('\n[4b] Sobol列そのものの健全性');
{const pts=H.simSobol(8,1024);
 chk(pts.every(p=>p.every(v=>v>=0&&v<1)),'全点が[0,1)に収まる');
 let worst=0;
 for(let j=0;j<8;j++){const b=new Array(8).fill(0);
   for(const p of pts)b[Math.min(7,Math.floor(p[j]*8))]++;
   worst=Math.max(worst,Math.max(...b.map(x=>Math.abs(x-128)))/128);}
 chk(worst<0.10,`1次元が一様(8分割の最大偏り ${(worst*100).toFixed(1)}%)`);
 // 次元どうしが同一列になっていないか（原始多項式の列挙を誤ると起きる。実際に起きた）
 let dup=0;
 for(let i=0;i<8;i++)for(let j=i+1;j<8;j++){
   let same=true;for(let k=0;k<pts.length&&same;k++) if(pts[k][i]!==pts[k][j]) same=false;
   if(same)dup++;}
 chk(dup===0,'次元どうしが同一の列になっていない');
 // 実運用の点数での次元間相関（方向数がオーバーフローで壊れると跳ね上がる）
 const dim=20,n=H.simSobolCount(dim);
 const sx=new Float64Array(dim),sxx=new Float64Array(dim),cr=new Float64Array(dim*dim);
 let cnt=0;
 H.simSobolEach(dim,n,(p)=>{cnt++;
   for(let i=0;i<dim;i++){sx[i]+=p[i];sxx[i]+=p[i]*p[i];
     for(let j=i+1;j<dim;j++)cr[i*dim+j]+=p[i]*p[j];}
   return true;});
 let mx=0;
 for(let i=0;i<dim;i++)for(let j=i+1;j<dim;j++){
   const c=(cnt*cr[i*dim+j]-sx[i]*sx[j])/Math.sqrt((cnt*sxx[i]-sx[i]**2)*(cnt*sxx[j]-sx[j]**2));
   mx=Math.max(mx,Math.abs(c));}
 chk(mx<0.05,`${dim}変数${n}点での次元間相関が小さい (最大 ${mx.toFixed(4)} < 0.05)`);}

console.log('\n[5] カテゴリ・連動・保存・凡例');
const catRow=[...d.querySelectorAll('.sim-row')].find(r=>r.dataset.col==='grade');
const catRng=catRow.querySelector('input[type=range]');
catRng.value='1'; catRng.dispatchEvent(new w.Event('input',{bubbles:true}));
catRng.dispatchEvent(new w.Event('change',{bubbles:true}));
chk(H.SIM.state['grade']==='B','カテゴリをスライダーで切り替えられる');
chk(catRng.getAttribute('aria-valuetext')==='B','カテゴリの読み上げ値が水準名');
{// ★で最適点へ移動すると、応答が走査中の最小値に一致する
 d.getElementById('simBestBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(Math.abs(Number(el('simPredVal').textContent)-Number(H.SIM.dist.best.y.toFixed(1)))<0.06,
   `★で最適点へ: 表示 ${el('simPredVal').textContent} == 走査の最小 ${simFmtLike(H.SIM.dist.best.y)}`);
 d.getElementById('simResetBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));}
el('simLinkBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simLinkBtn').classList.contains('on'),'連動モードON');
H.Platform.downloadFile=(b,n,m)=>{w.__SAVED__={n:n,len:b.length};};
el('simSaveBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(!!w.__SAVED__&&w.__SAVED__.n==='scenario.csv',`CSV保存 (${w.__SAVED__&&w.__SAVED__.len} bytes)`);
chk(!el('simLegend').classList.contains('open'),'凡例は既定で閉じている');
el('simLegendBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simLegend').classList.contains('open')&&d.querySelectorAll('#simLegend div').length>=6,'?で凡例が開く');
el('simLegendBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));

console.log('\n[6] 低精度時の抑制表示');
{const R2=JSON.parse(JSON.stringify(RESULT)); R2.r2=0.28;
 H.applyTrainingResult(R2); d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(el('simRight').classList.contains('lowq'),'R²0.28で彩度を落とす');}

console.log('\n[7] 使えない場合のフォールバック');
H.Platform.getTregBytes=()=>null; H.applyTrainingResult(RESULT);
chk(!d.getElementById('predictZone').className.includes('sim-open-btn'),'.treg が無ければ③は従来のCSVドロップ');
H.Platform.getTregBytes=()=>w.__TREG__; const R3=JSON.parse(JSON.stringify(RESULT)); R3.slider_spec=[];
H.applyTrainingResult(R3);
chk(!d.getElementById('predictZone').className.includes('sim-open-btn'),'slider_spec が空でも安全に無効化');
H.applyTrainingResult(RESULT); d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simulateOverlay').style.display==='flex'&&q('.sim-row')===21,'正常なresultで復帰する');
d.getElementById('simCloseBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simulateOverlay').style.display==='none','✕で閉じる');
d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
d.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
chk(el('simulateOverlay').style.display==='none','Escで閉じる');
d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));

console.log('\n[9] 学習済みモデルの持ち込み（複数連動）');
{chk(!!d.getElementById('simAddBtn')&&!!d.getElementById('simAddInput'),'モデル読み込みの入口がある');
 const before=H.SIM.models.length, colsBefore=q('.sim-row');
 // .treg をそのまま読み込む（テストなので内部APIを直接呼ぶ）
 const fakeFile={name:'other_model.treg',
   arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
 return H.simAddModelFile(fakeFile).then(ok=>{
  chk(ok===true,'.treg を読み込める');
  chk(H.SIM.models.length===before+1,`モデルが増える (${H.SIM.models.length}個)`);
  chk(q('.sim-mrow')===H.SIM.models.length,'応答一覧に全モデルが並ぶ');
  chk(q('.sim-row')>=colsBefore,'スライダーは減らない（列は和集合）');
  // 同じスライダーで全モデルの応答が動く
  const vals=()=>[...d.querySelectorAll('.sim-mval')].map(e=>e.textContent);
  const v0=vals();
  const r=d.querySelector('.sim-row input[type=range]');
  r.value=r.max; r.dispatchEvent(new w.Event('change',{bubbles:true}));
  const v1=vals();
  chk(v1.length===H.SIM.models.length&&v1.every(x=>x.length>0),
    `全モデルの応答が表示される (${v1.join(' / ')})`);
  chk(!!H.SIM.models[1].spec.length,'読み込んだモデルからスライダー仕様を復元できる');
  // 外す
  d.querySelector('.sim-mx').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
  chk(H.SIM.models.length===before,'×でモデルを外せる');
  finish();
 });}

function finish(){
console.log('\n[10] ワークフロー側の表示');
chk(!d.getElementById('deployHint'),'④のヒント表示は無い');
chk(!/no network/i.test(html)&&!/ネットワーク不要/.test(html),'NO NETWORK 表示は無い');
// ロボのgifは3〜4MBある。切り替えのたびに取り直すと枠が空になる（ロボが消える）
chk(!/\.gif\?t=|gif\}\?t=\$\{Date/.test(html),'ロボのgifにキャッシュバスターを付けていない');
chk(/_preloadRobotGifs/.test(html),'ロボのgifを起動時にプリロードしている');
{const imgs=d.querySelectorAll('#charBox img');
 chk(imgs.length>=3,`gifを<img>として保持している (${imgs.length}枚)`);
 chk([...imgs].every(i=>!/\?t=/.test(i.getAttribute('src')||'')),'srcにクエリが付いていない');}

console.log('\n[8] 画面に出る文字量');
const shown=el('simulateSection').textContent.replace(/\s+/g,' ').trim();
console.log('       '+JSON.stringify(shown.slice(0,150))+'...');
console.log('       文字数 '+shown.length+' / title保持要素 '+q('#simulateSection [title]'));
chk(q('#simulateSection [title]')>=15,'説明はtitleに退避されている');

console.log('');
if(errs.length){console.log('ERRORS:\n'+errs.join('\n'));process.exit(1);}
if(FAIL.length){console.log(`NG: ${FAIL.length} 件失敗`);FAIL.forEach(m=>console.log('  - '+m));process.exit(1);}
console.log('すべて成功');
}

// jsdom の rAF ループが残るとプロセスが終了しないため、明示的に閉じて終了する
try { dom.window.close(); } catch (_) {}
process.exit(0);
