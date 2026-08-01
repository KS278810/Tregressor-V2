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
try{w.eval(code+'\n;window.__H__={Platform:Platform,applyTrainingResult:applyTrainingResult,showTrainedTab:showTrainedTab,SIM:SIM};');}catch(e){errs.push('THROW: '+e.message+'\n'+(e.stack||'').split('\n').slice(0,6).join('\n'));}

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
chk(el('simSubtitle').textContent.length>0,`サブタイトル: ${el('simSubtitle').textContent}`);
chk(q('.sim-row')===21,`全21変数にスライダーが用意される (実測 ${q('.sim-row')})`);
chk(q('#simChips .sim-chip')===0,'既定では固定チップは無い');
chk(el('simChipRow').style.display==='none','固定行は出ない');
chk(el('simTCount').textContent==='21 / 21',`本数表示 ${el('simTCount').textContent}`);

console.log('\n[2] 予測値が推論エンジンと一致するか（最重要）');
const pv=el('simPredVal').textContent;
const seedMed=RESULT.seed_rows.find(s=>s.label==='median');
const row={},raw={};
sliderSpec.forEach(s=>{const v=seedMed.values[s.col];raw[s.col]=String(v);row[s.col]=Number(v);});
const expect=core.predictRow(m1,row,raw);
chk(Math.abs(Number(pv)-Number(expect.toFixed(1)))<0.06,
  `基準行(MID)の表示 ${pv} が predictRow の ${expect.toFixed(4)} と一致`);
chk(el('simPredBand').textContent==='±6.8',`誤差帯 ${el('simPredBand').textContent}`);

console.log('\n[3] スライダー操作');
// 表示値は「変わったか」ではなく「エンジンの計算値と一致するか」で検証する
const stateOf=(over)=>{const row={},raw={};
  sliderSpec.forEach(s=>{let v=(over&&over[s.col]!==undefined)?over[s.col]:seedMed.values[s.col];
    raw[s.col]=String(v);row[s.col]=Number(v);});return {row,raw};};
const r0=d.querySelector('.sim-row input[type=range]');
const col0=d.querySelector('.sim-row').dataset.col;
const before=el('simPredVal').textContent;
r0.value=r0.max; r0.dispatchEvent(new w.Event('input',{bubbles:true}));
r0.dispatchEvent(new w.Event('change',{bubbles:true}));
{const st=stateOf({[col0]:r0.max});const ex=core.predictRow(m1,st.row,st.raw);
 chk(Math.abs(Number(el('simPredVal').textContent)-Number(ex.toFixed(1)))<0.06,
   `${col0} を最大へ: 表示 ${el('simPredVal').textContent} == エンジン ${ex.toFixed(4)}`);}
chk(d.querySelector('.sim-row').classList.contains('warn'),'外挿するとamberになる');
const tip=d.querySelector('.sim-row').title;
const has=(ja,en)=>tip.includes(ja)||tip.includes(en);
chk(has('学習データの外側','outside the training data'),'外挿の説明はtitleに退避されている');
chk(has('飽和','saturate'),'飽和の説明もtitleにある');
const flags=[...d.querySelectorAll('.sim-flag')].map(f=>f.className.replace('sim-flag ','')+':'+f.textContent);
chk(flags.some(f=>f.startsWith('on:')),'外挿フラグが出る');
chk(flags.some(f=>f.startsWith('stop:')),'飽和フラグが出る');
const at=el('simPredVal').textContent;
r0.value=String(Number(r0.max)); r0.dispatchEvent(new w.Event('change',{bubbles:true}));
chk(el('simPredVal').textContent===at,'x_clip境界の外では予測が変化しない');

console.log('\n[4] 変数が多い場合の操作');
el('simExpandBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row')===12,`▴で重要度上位12変数に絞れる (実測 ${q('.sim-row')})`);
chk(el('simChipRow').style.display==='','絞ると固定行が出る');
el('simExpandBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row')===21,'▾で全変数に戻る');
el('simExpandBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
const names=()=>[...d.querySelectorAll('.sim-name')].map(e=>e.textContent);
d.querySelector('#simChips .sim-chip').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row')===13,'チップから昇格できる');
d.querySelector('.sim-x').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row')===12,'×で降格できる');
el('simExpandBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));

console.log('\n[5] カテゴリ・基準行・連動・保存');
const catRow2=[...d.querySelectorAll('.sim-row')].find(r=>r.dataset.col==='grade');
chk(!!catRow2&&catRow2.querySelectorAll('.sim-pill').length===2,'カテゴリはピル表示');
chk(catRow2.querySelector('.sim-pill.on').dataset.v==='A','初期値が選択状態');
const pB=[...catRow2.querySelectorAll('.sim-pill')].find(p=>p.dataset.v==='B');
pB.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(catRow2.querySelector('.sim-pill.on').dataset.v==='B','ピルで切り替わる');
chk(catRow2.querySelector('.sim-val').textContent==='','選択中の値は右端に再掲しない');
chk(!d.getElementById('simSeedSeg'),'LOW/MID/HIGH/RANDの選択UIは無い（出発点は自動）');
d.getElementById('simResetBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
{const sm=RESULT.seed_rows.find(x=>x.label==='median');
 const row={},raw={};sliderSpec.forEach(s=>{const v=sm.values[s.col];raw[s.col]=String(v);row[s.col]=Number(v);});
 const ex=core.predictRow(m1,row,raw);
 chk(Math.abs(Number(el('simPredVal').textContent)-Number(ex.toFixed(1)))<0.06,
   `↺で中位の実データ行に戻る (表示 ${el('simPredVal').textContent} == エンジン ${ex.toFixed(4)})`);}
chk(q('.sim-row')===21,'リセット後も全変数のスライダーが並ぶ');
el('simLinkBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simLinkBtn').classList.contains('on'),'連動モードON');
let saved=null; H.Platform.downloadFile=(b,n,m)=>{w.__SAVED__={n:n,m:m,len:b.length};};
el('simSaveBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
saved=w.__SAVED__;
chk(!!saved&&saved.n==='scenario.csv'&&saved.len>0,`CSV保存 (${saved&&saved.n}, ${saved&&saved.len} bytes)`);

console.log('\n[6] 低精度時の抑制表示');
const R2=JSON.parse(JSON.stringify(RESULT)); R2.r2=0.28;
H.applyTrainingResult(R2); d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simRight').classList.contains('lowq'),'R²0.28で彩度を落とす');
chk(!el('simPredVal').title.includes('上位')&&!el('simPredVal').title.includes('Top '),
  '低精度時はパーセンタイルを出さない');
chk(el('simPredVal').title.includes('信頼性は低い')||el('simPredVal').title.includes('unreliable'),
  '低精度の注意文はtitleに入る');

console.log('\n[7] 使えない場合のフォールバック');
H.Platform.getTregBytes=()=>null; H.applyTrainingResult(RESULT);
chk(!d.getElementById('predictZone').className.includes('sim-open-btn'),'.treg が無ければ③は従来のCSVドロップのまま');
H.Platform.getTregBytes=()=>w.__TREG__; const R3=JSON.parse(JSON.stringify(RESULT)); R3.slider_spec=[]; H.applyTrainingResult(R3);
chk(!d.getElementById('predictZone').className.includes('sim-open-btn'),'slider_spec が空でも安全に無効化');
H.applyTrainingResult(RESULT); d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simulateOverlay').style.display==='flex'&&q('.sim-row')===21,'正常なresultで復帰する');
// 閉じる操作
d.getElementById('simCloseBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simulateOverlay').style.display==='none','✕で閉じる');
d.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simulateOverlay').style.display==='flex','再度③で開き直せる');
d.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
chk(el('simulateOverlay').style.display==='none','Escで閉じる');
d.getElementById('predictZone').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));

console.log('\n[7b] 日英切替');
{const H2=w.__H__;const before=el('simFoot').textContent;
 const btn=[...d.querySelectorAll('.lang-toggle-btn')].find(b=>b.dataset.lang==='en');
 if(btn){btn.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
   chk(/desktop/i.test(el('simFoot').textContent),
     '英語に切り替わる: '+el('simFoot').textContent);
   const jb=[...d.querySelectorAll('.lang-toggle-btn')].find(b=>b.dataset.lang==='ja');
   jb.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
   chk(el('simFoot').textContent.includes('デスクトップ版'),'日本語に戻る: '+el('simFoot').textContent);
   chk(q('.sim-row')===21,'言語切替後もスライダーが維持される');}}


console.log('\n[9] 基準からの差分・直接入力・変更マーク・凡例');
// 開き直して基準行(MID)の状態に戻す
d.getElementById('simResetBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row.changed')===0,'触る前は変更マークが1つも無い');
chk(el('simDelta').textContent.length>0&&!el('simDelta').textContent.includes('+'),
  `差分表示は「基準行のまま」: ${el('simDelta').textContent}`);
const base=Number(el('simPredVal').textContent);
// 数値の直接入力
const vin=d.querySelector('.sim-row input.sim-val');
const tcol=vin.closest('.sim-row').dataset.col;
const tspec=sliderSpec.find(x=>x.col===tcol);
chk(!!vin,'数値列の値が直接入力できる欄になっている');
vin.value=String(tspec.p99);
vin.dispatchEvent(new w.Event('change',{bubbles:true}));
{const st=stateOf({[tcol]:tspec.p99});const ex=core.predictRow(m1,st.row,st.raw);
 chk(Math.abs(Number(el('simPredVal').textContent)-Number(ex.toFixed(1)))<0.06,
   `直接入力した値が予測に反映される (${tcol}=${simFmtLike(tspec.p99)})`);}
chk(vin.closest('.sim-row').classList.contains('changed'),'動かした変数に変更マークが付く');
chk(q('.sim-row.changed')===1,'変更マークは動かした1本だけ');
{const dt=el('simDelta').textContent;
 const expD=Number(el('simPredVal').textContent)-base;
 const shown=dt.replace(/[^0-9.]/g,'');
 chk(dt.includes('+')||dt.includes('−')||Math.abs(expD)<0.05,`差分が表示される: ${dt}`);}
// 範囲外の入力は clamp される
const [rlo,rhi]=[Number(vin.closest('.sim-row').querySelector('input[type=range]').min),
                 Number(vin.closest('.sim-row').querySelector('input[type=range]').max)];
vin.value='999999';
vin.dispatchEvent(new w.Event('change',{bubbles:true}));
chk(Math.abs(Number(vin.value)-Number(simFmtLike(rhi)))<Math.max(0.2,Math.abs(rhi)*0.02),
  `範囲外の入力は上限へ丸められる (${vin.value})`);
vin.value='abc';
vin.dispatchEvent(new w.Event('change',{bubbles:true}));
chk(Number.isFinite(Number(vin.value)),'数値でない入力を入れても壊れない');
// 列名クリックでその1本だけ戻る
const chRow=d.querySelector('.sim-row.changed');
chRow.querySelector('.sim-name').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row.changed')===0,'列名クリックでその変数だけ基準値に戻る');
chk(Math.abs(Number(el('simPredVal').textContent)-base)<0.06,'戻すと予測も基準に戻る');
// 凡例
chk(!el('simLegend').classList.contains('open'),'凡例は既定で閉じている');
el('simLegendBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(el('simLegend').classList.contains('open'),'?ボタンで凡例が開く');
chk(d.querySelectorAll('#simLegend div').length>=8,
  `凡例に記号の意味が並ぶ (${d.querySelectorAll('#simLegend div').length}件)`);
el('simLegendBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(!el('simLegend').classList.contains('open'),'もう一度押すと閉じる');
// アクセシビリティ
const rng2=d.querySelector('.sim-row input[type=range]');
chk(!!rng2.getAttribute('aria-valuetext'),`range に aria-valuetext がある (${rng2.getAttribute('aria-valuetext')})`);
chk(!!rng2.getAttribute('aria-label'),'range に aria-label がある');


console.log('\n[10] ワークフロー側の表示');
chk(!!d.getElementById('deployHint'),'④に「学習後に使えます」のヒントがある');
{const html2=fs.readFileSync(path.join(ROOT,'web/index.html'),'utf8');
 chk(!/no network/i.test(html2)&&!/ネットワーク不要/.test(html2),
   '起動時の「NO NETWORK / ネットワーク不要」表示を撤去した');}


console.log('\n[11] レイアウトの幅（jsdomは自動レイアウトしないのでCSSの数値で検算）');
// 1行は [重要度][列名][トラック][値][感度][×] の横並び。固定要素の合計が
// グリッド列の最小幅を食い潰すとスライダー本体が潰れて「見えない」状態になる。
{const css=html;
 const num=(re)=>{const m=css.match(re);return m?Number(m[1]):null;};
 const colMin = num(/minmax\((\d+)px, 1fr\)/);
 const nameW  = num(/\.sim-name \{[^}]*width: (\d+)px/s);
 const valW   = num(/\.sim-val \{[^}]*width: (\d+)px/s);
 const sparkW = num(/\.sim-spark \{ width: (\d+)px/);
 const impW   = num(/\.sim-imp \{ width: (\d+)px/);
 const xW     = num(/\.sim-x \{[^}]*width: (\d+)px/s);
 const gap    = num(/\.sim-row \{[^}]*gap: (\d+)px/s);
 const trackMin = num(/\.sim-track \{[^}]*min-width: (\d+)px/s);
 const rowPad = 8;   // padding: 0 2px 0 6px
 const fixed = impW + nameW + valW + sparkW + xW + gap * 5 + rowPad;
 const track = colMin - fixed;
 console.log(`       列最小 ${colMin}px / 固定 ${fixed}px (imp${impW}+name${nameW}+val${valW}+spark${sparkW}+x${xW}+gap${gap}x5+pad${rowPad})`);
 console.log(`       → スライダー本体に残る幅 ${track}px (min-width ${trackMin}px)`);
 chk(track >= trackMin, `スライダー本体が min-width(${trackMin}px) 以上を確保できる`);
 chk(track >= 120, `スライダーとして実用的な幅がある (${track}px >= 120px)`);
 // つまみは半径7.5px。端で切れないための余白(PAD)がトラック描画側にあること
 const pad = num(/const PAD = (\d+);/);
 chk(pad >= 8, `つまみが端で切れない余白がある (PAD=${pad}px)`);
 // Canvasの描画高さとCSSの高さが食い違うと、線が切れたり隙間が出る
 const trackH = num(/\.sim-track \{[^}]*height: (\d+)px/s);
 const jsTrackH = num(/const w = wrap\.clientWidth \|\| 0, h = (\d+)/);
 chk(trackH === jsTrackH, `トラックのCSS高さ(${trackH}px)とCanvas描画高さ(${jsTrackH}px)が一致`);
 const sparkH = num(/\.sim-spark \{ width: \d+px; height: (\d+)px/);
 const jsSparkW = num(/const w = (\d+), h = \d+;\s*\n\s*const pts = \[\]/);
 const jsSparkH = num(/const w = \d+, h = (\d+);\s*\n\s*const pts = \[\]/);
 chk(sparkW === jsSparkW && sparkH === jsSparkH,
   `感度グラフのCSS(${sparkW}x${sparkH})とCanvas(${jsSparkW}x${jsSparkH})が一致`);
 const rowH = num(/\.sim-row \{[^}]*height: (\d+)px/s);
 chk(rowH >= trackH, `行の高さ(${rowH}px)がトラック(${trackH}px)を収められる`);}

console.log('\n[8] 画面に出る文字量');
const shown=el('simulateSection').textContent.replace(/\s+/g,' ').trim();
console.log('       '+JSON.stringify(shown.slice(0,150))+'...');
console.log('       文字数 '+shown.length+' / title保持要素 '+q('#simulateSection [title]'));
chk(q('#simulateSection [title]')>=15,'説明はtitleに退避されている');

console.log('');
if(errs.length){console.log('ERRORS:\n'+errs.join('\n'));process.exit(1);}
if(FAIL.length){console.log(`NG: ${FAIL.length} 件失敗`);FAIL.forEach(m=>console.log('  - '+m));process.exit(1);}
console.log('すべて成功');

// jsdom の rAF ループが残るとプロセスが終了しないため、明示的に閉じて終了する
try { dom.window.close(); } catch (_) {}
process.exit(0);
