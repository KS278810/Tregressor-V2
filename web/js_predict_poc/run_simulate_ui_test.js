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
try{w.eval(code+'\n;window.__H__={Platform:Platform,applyTrainingResult:applyTrainingResult,showTrainedTab:showTrainedTab,SIM:SIM,simSobolCount:simSobolCount,simSobol:simSobol,simSobolEach:simSobolEach,simAddModelFile:simAddModelFile,simBuildCliScript:simBuildCliScript,setLang:setLang,t:t};');}catch(e){errs.push('THROW: '+e.message+'\n'+(e.stack||'').split('\n').slice(0,6).join('\n'));}

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
const simSpecCols=sliderSpec.map(s=>s.col);
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
{const v0=d.querySelector('.sim-row input.sim-val');
 chk(!!v0,'スライダー横に数値欄がある');
 chk(v0.value.length>0,`数値が表示される (${v0.value})`);
 // 本体のR²表示と同じ書体（デジタル感のある --font-display）を使う
 const blk=html.slice(html.indexOf('.sim-val {'), html.indexOf('.sim-val {')+400);
 chk(/font-family: var\(--font-display\)/.test(blk),'数値は本体と同じ書体(--font-display)');
 chk(/tabular-nums/.test(blk),'桁が揺れないよう tabular-nums');
 // 応答の数値も同じ書体
 const pblk=html.slice(html.indexOf('.sim-pnum {'), html.indexOf('.sim-pnum {')+300);
 chk(/font-family: var\(--font-display\)/.test(pblk),'応答の数値も本体と同じ書体');
 chk(/tabular-nums/.test(pblk),'応答の数値も tabular-nums');
 const r2blk=html.slice(html.indexOf('.r2-big {'), html.indexOf('.r2-big {')+200);
 chk(/font-family: var\(--font-display\)/.test(r2blk),'本体のR²表示も同じ書体（踏襲できている）');}
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
chk(!d.getElementById('simSaveBtn'),'CSV保存(↓)ボタンは無い');
chk(!d.getElementById('simFoot'),'下部の案内文は無い');
chk(!d.getElementById('simLegendBtn')&&!d.getElementById('simLegend'),'？の凡例機能は無い');
{// 単体HTML(DLしたもの)は、本体のSIMULATEと寸分たがわず同じ見た目にする。
 // sim-only 側でパネルの大きさ・位置を上書きすると本体とずれ、端が切れる原因になる。
 // 上書きしてよいのは背景(後ろに本体UIが無いため)と閉じるボタンの非表示だけ。
 for (const prop of ['position','inset','width','height','max-width','max-height',
                     'top','left','right','bottom','transform','margin','padding']) {
   chk(!new RegExp('body\\.sim-only #simulatePanel[^}]*[;{ ]'+prop+' *:').test(html),
     `単体HTMLはパネルの ${prop} を上書きしない（本体のSIMULATEと同じ寸法）`);
 }
 chk(!/body\.sim-only #simulateOverlay[^}]*padding *:/.test(html),
   '単体HTMLはオーバーレイの余白も上書きしない');
 chk(/\.sim-grid \{[\s\S]{0,200}overflow: auto/.test(html),'収まらない場合はスクロールする');
 chk(/grid-template-columns: minmax\(0, 1fr\) minmax\(\d+px, \d+px\)/.test(html),
   '右列は狭い画面でも縮められる');
 chk(!/grid\.style\.transform = 'scale/.test(html),
   '拡大縮小で無理に収めない（親のoverflow:hiddenで切れるため）');
 // スクロールバーは本体と同じ見た目
 const fb=html.slice(html.indexOf('.feature-bars-scroll::-webkit-scrollbar'),
                     html.indexOf('.feature-bars-scroll::-webkit-scrollbar')+300);
 const w1=(fb.match(/width: (\d+)px/)||[])[1];
 const sm=html.slice(html.indexOf('.sim-sliders::-webkit-scrollbar,'),
                     html.indexOf('.sim-sliders::-webkit-scrollbar,')+700);
 const w2=(sm.match(/width: (\d+)px/)||[])[1];
 chk(w1===w2,`スクロールバーの太さが本体と同じ (${w1}px / ${w2}px)`);
 chk(/rgba\(63,196,236,0\.14\)/.test(sm),'スクロールバーの色も本体と同じ');}
chk(!/btn-deploy-ready \{ color: var\(--gold\)/.test(html),'DEPLOYの準備完了はオレンジではない');
chk(!/ctaGlowGold/.test(html),'DEPLOY用の金色アニメーションは無い');
chk(/\.sim-sliders \{ display: flex; flex-direction: column;/.test(html),'スライダーは縦一列');
{// 全画面ではなく元ツールの外形に収まる大きさか（#simulatePanel の指定を見る）
 const blk=html.slice(html.indexOf('#simulatePanel {'), html.indexOf('#simulatePanel {')+400);
 const mw=blk.match(/width: min\((\d+)px, calc\(100vw - (\d+)px\)\)/);
 const mh=blk.match(/height: min\((\d+)px, calc\(100vh - (\d+)px\)\)/);
 chk(!!mw&&!!mh,'パネルは上限つきのポップアップ');
 chk(mw&&Number(mw[2])>=80&&mh&&Number(mh[2])>=80,
   `画面から見切れない余白がある (左右${mw&&mw[2]}px / 上下${mh&&mh[2]}px)`);
 chk(mw&&Number(mw[1])<=1300&&mh&&Number(mh[1])<=800,
   `ポップアップとして控えめな上限 (${mw&&mw[1]}x${mh&&mh[1]}px)`);}

chk(el('simTargetTag').textContent.trim()==='PREDICTED',
  `見出しに目的変数名を含めない (${el('simTargetTag').textContent})`);

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
{// 数値欄への直接入力が予測に反映される
 const vi=d.querySelector('.sim-row input.sim-val');
 const target=(sp0.min+sp0.max)/2;
 vi.value=String(target);
 vi.dispatchEvent(new w.Event('change',{bubbles:true}));
 const st=stateOf({[col0]:target});const ex=core.predictRow(m1,st.row,st.raw);
 chk(Math.abs(Number(el('simPredVal').textContent)-Number(ex.toFixed(1)))<0.06,
   `数値を直接入力すると予測に反映される (${col0}=${simFmtLike(target)})`);
 vi.value='999999'; vi.dispatchEvent(new w.Event('change',{bubbles:true}));
 chk(Math.abs(Number(vi.value)-Number(simFmtLike(sp0.max)))<Math.max(0.2,Math.abs(sp0.max)*0.02),
   `範囲外の入力は上限へ丸められる (${vi.value})`);
 vi.value='abc'; vi.dispatchEvent(new w.Event('change',{bubbles:true}));
 chk(Number.isFinite(Number(vi.value)),'数値でない入力を入れても壊れない');
 // カテゴリは水準名を表示する読み取り専用欄
 const cr=[...d.querySelectorAll('.sim-row')].find(r=>r.dataset.col==='grade');
 const cv2=cr&&cr.querySelector('input.sim-val');
 chk(!!cv2&&cv2.readOnly&&cv2.value==='A',`カテゴリは水準名を表示 (${cv2&&cv2.value})`);}
chk(!d.querySelector('.sim-row.warn'),'上下限まで動かしてもオレンジにはならない');
{const sp0b=sliderSpec.find(x=>x.col===col0);
 chk(Math.abs(Number(r0.min)-sp0b.min)<1e-9&&Math.abs(Number(r0.max)-sp0b.max)<1e-9,
   '可動域は常に変数の上下限');}
// 変数名は書き換えられる入力欄になったので、名前をクリックしても値は戻らない
// （その変数だけ戻す操作は左端の重要度バーへ移した。[15]で確認する）
d.querySelector('.sim-row .sim-name').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row.changed')>0,'名前をクリックしても値は戻らない（名前を編集できる欄のため）');
d.querySelector('.sim-row.changed .sim-imp').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
chk(q('.sim-row.changed')===0,'左端の棒をクリックするとその変数だけ元に戻る');

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

console.log('\n[5] カテゴリ・連動・保存');
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
chk(!d.getElementById('simLinkBtn'),'連動モードのボタンは無い（変数は常に独立）');
{// 1本動かしても他の変数の値は変わらない
 const before={}; simSpecCols.forEach(c=>before[c]=String(H.SIM.state[c]));
 const rr=d.querySelectorAll('.sim-row input[type=range]')[2];
 const cc=d.querySelectorAll('.sim-row')[2].dataset.col;
 rr.value=rr.max; rr.dispatchEvent(new w.Event('change',{bubbles:true}));
 const moved=simSpecCols.filter(c=>String(H.SIM.state[c])!==before[c]);
 chk(moved.length===1&&moved[0]===cc,
   `動かした1本だけが変わる (変化した変数: ${moved.join(',')||'なし'})`);
 d.getElementById('simResetBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));}

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
{// 親の clientWidth は padding を含むので、それで描くと右端がはみ出す
 chk(/cv\.style\.width = '100%';[\s\S]{0,120}cv\.clientWidth/.test(html),
   'ヒストグラムは自分自身の内容幅で描く（右端が切れない）');}

{// 詳細ポップアップの散布図は「軸の内側」が正方形であるべき
 const m=html.match(/const _pad = \{ l: (\d+), r: (\d+), t: (\d+), b: (\d+) \};/);
 const hm=html.match(/const cssH = cssW - \(_pad\.l \+ _pad\.r\) \+ \(_pad\.t \+ _pad\.b\);/);
 chk(!!m&&!!hm,'散布図はpadの差を足した高さにしている');
 if(m){const L=+m[1],R=+m[2],T=+m[3],B=+m[4];
  const cssW=400, cssH=cssW-(L+R)+(T+B);
  chk(cssW-L-R===cssH-T-B,`グラフ本体が正方形 (${cssW-L-R}x${cssH-T-B})`);}}
{// ライセンス表記は「できること」を先に出す
 const jaLic=html.match(/'terms\.licenseLine': '([^']+)'/);
 chk(!!jaLic&&/教育|研究/.test(jaLic[1]),`ライセンス表記が用途を先に述べる: ${jaLic&&jaLic[1].slice(0,40)}...`);
 chk(!/❌/.test(html),'「できない」を強調する記号(❌)を使っていない');}


console.log('\n[12] ④モデルのDL＝SIMULATE単体HTML');
{const tmplPath=path.join(ROOT,'web/simulate_template.html');
 chk(fs.existsSync(tmplPath),'書き出し用の雛形が生成されている');
 const tmpl=fs.readFileSync(tmplPath,'utf8');
 const MARK='null /* __MODEL'+'_SLOT__ */';
 chk(tmpl.split(MARK).length-1===1,`雛形の差し込み口はちょうど1箇所 (${tmpl.split(MARK).length-1})`);
 // 実際に書き出しを行い、その中身を別のjsdomで起動してSIMULATEが立ち上がるか確認する
 let saved=null;
 // jsdom には fetch が無いので、雛形の取得だけローカル読み込みに差し替える
 w.fetch=async(u)=>({ok:true,text:async()=>fs.readFileSync(path.join(ROOT,'web',String(u).replace('./','')),'utf8')});
 // 書き出しは Blob + <a download> で行われるので、そこを捕まえる
 let blob=null,name=null;
 // jsdom の Blob は .text() を持たないので、生成時のバイト列を横取りする
 const OrigBlob=w.Blob;
 w.Blob=function(parts,opts){ blob={parts:parts}; return new OrigBlob(parts,opts); };
 w.URL.createObjectURL=()=>'blob:test';
 w.URL.revokeObjectURL=()=>{};
 w.HTMLAnchorElement.prototype.click=function(){ name=this.download; };
 H.Platform.getTregBytes=()=>w.__TREG__;
 return H.Platform.exportModel({fileStem:'model_test'}).then(()=>{const p0=blob.parts[0];
  const text=(typeof p0==='string')?p0:Buffer.from(p0).toString('utf8');
  saved={n:name,text:text};
  chk(!!saved&&saved.n==='model_test.html',`単体HTMLを書き出す (${saved&&saved.n})`);
  chk(saved.text.length>tmpl.length*0.9,'雛形と同等の大きさ（UIを含んでいる）');
  chk(!saved.text.includes(MARK),'差し込み口が実データに置換されている');

  // 書き出したHTMLを起動する
  const d2=new JSDOM(saved.text,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost/'});
  const w2=d2.window;
  w2.HTMLCanvasElement.prototype.getContext=()=>ctx2d;
  if(!w2.TextDecoder) w2.TextDecoder=require('util').TextDecoder;
  if(!w2.TextEncoder) w2.TextEncoder=require('util').TextEncoder;
  Object.defineProperty(w2.Element.prototype,'clientWidth',{get(){return 300;}});
  const errs2=[];
  w2.addEventListener('error',e=>errs2.push(e.message));
  w2.addEventListener('unhandledrejection',e=>errs2.push('unhandled: '+(e.reason&&e.reason.message||e.reason)));
  process.on('unhandledRejection',r=>errs2.push('unhandled: '+(r&&r.message||r)));
  const code2=saved.text.match(/<script>([\s\S]*)<\/script>/)[1];
  try{ w2.eval(code2+'\n;window.__H2__={SIM:SIM,IS_EMBEDDED:IS_EMBEDDED,bootEmbedded:_bootEmbedded};'); }
  catch(e){ errs2.push('THROW: '+e.message); }
  chk(errs2.length===0,'書き出したHTMLが例外なく実行される'+(errs2.length?': '+errs2[0]:''));
  const H2=w2.__H2__||{};
  chk(H2.IS_EMBEDDED===true,'埋め込みモードとして認識される');
  // 起動処理は手で呼ばない。実際のブラウザと同じく、ページ自身が立ち上げられるか見る。
  // (以前ここを手で呼んでいたため、起動が TDZ で落ちて真っ黒になる不具合を見逃した)
  return new Promise(res=>setTimeout(res,300)).then(()=>{
  chk(errs2.length===0,'起動時に例外が出ない'+(errs2.length?': '+errs2[0]:''));
  chk(w2.document.body.classList.contains('sim-only'),'本体UIを隠すモードになる');
  chk(w2.document.getElementById('simulateOverlay').style.display==='flex','SIMULATEが自動で開く');
  chk(w2.document.querySelectorAll('.sim-row').length===21,
    `スライダーが並ぶ (${w2.document.querySelectorAll('.sim-row').length}本)`);
  // 予測値が本体と一致する
  const pv2=w2.document.getElementById('simPredVal').textContent;
  chk(Math.abs(Number(pv2)-Number(expect.toFixed(1)))<0.06,
    `書き出し先でも同じ予測になる (${pv2})`);
  try{ d2.window.close(); }catch(_){}
  finish2();
  });
 });}

function finish2(){

console.log('\n[10] ワークフロー側の表示');
chk(!d.getElementById('deployHint'),'④のヒント表示は無い');
chk(!/no network/i.test(html)&&!/ネットワーク不要/.test(html),'NO NETWORK 表示は無い');
// ロボのgifは3〜4MBある。切り替えのたびに取り直すと枠が空になる（ロボが消える）
chk(!/\.gif\?t=|gif\}\?t=\$\{Date/.test(html),'ロボのgifにキャッシュバスターを付けていない');
chk(/_preloadRobotGifs/.test(html),'ロボのgifを起動時にプリロードしている');
{const imgs=d.querySelectorAll('#charBox img');
 chk(imgs.length>=3,`gifを<img>として保持している (${imgs.length}枚)`);
 chk([...imgs].every(i=>!/\?t=/.test(i.getAttribute('src')||'')),'srcにクエリが付いていない');}


console.log('\n[11] レイアウトの幅（jsdomは自動レイアウトしないのでCSSの数値で検算）');
// 1行 = [重要度][列名][トラック][数値]。固定要素の合計が行幅を食い潰すと
// スライダー本体が潰れて「見えない」状態になる（実際に一度起きた）。
{const num=(re)=>{const m=html.match(re);return m?Number(m[1]):null;};
 const rowMax = num(/\.sim-row \{ width: 100%; max-width: (\d+)px; \}/);
 const nameW  = num(/\.sim-name \{ width: (\d+)px/);
 const valW   = num(/\.sim-val \{[\s\S]{0,200}?width: (\d+)px/);
 const impW   = num(/\.sim-imp \{ width: (\d+)px/);
 const gap    = num(/\.sim-row \{[^}]*gap: (\d+)px/s);
 const trackMin = num(/\.sim-track \{[^}]*min-width: (\d+)px/s);
 const rowPad = 8;
 const fixed = impW + nameW + valW + gap * 3 + rowPad;
 const track = rowMax - fixed;
 console.log(`       行の最大幅 ${rowMax}px / 固定 ${fixed}px (imp${impW}+name${nameW}+val${valW}+gap${gap}x3+pad${rowPad})`);
 console.log(`       → スライダー本体に残る幅 ${track}px (min-width ${trackMin}px)`);
 chk(track >= trackMin, `スライダー本体が min-width(${trackMin}px) 以上を確保できる`);
 chk(track >= 200, `スライダーとして十分な幅がある (${track}px >= 200px)`);
 const trackH = num(/\.sim-track \{[^}]*height: (\d+)px/s);
 const jsTrackH = num(/const w = wrap\.clientWidth \|\| 0, h = (\d+)/);
 chk(trackH === jsTrackH, `トラックのCSS高さ(${trackH}px)とCanvas描画高さ(${jsTrackH}px)が一致`);
 const rowH = num(/\.sim-row \{[^}]*height: (\d+)px/s);
 chk(rowH >= trackH, `行の高さ(${rowH}px)がトラック(${trackH}px)を収められる`);
 const pad = num(/const PAD = (\d+);/);
 chk(pad >= 8, `つまみが端で切れない余白がある (PAD=${pad}px)`);}


console.log('\n[13] コマンドラインで使うスクリプト');
try{ {chk(!!d.getElementById('simCliBtn'),'CLIスクリプトの書き出しボタンがある');
 chk(typeof w.TregPredictCoreSource==='string'&&w.TregPredictCoreSource.length>1000,
   'エンジンのソースが文字列としても埋め込まれている');
 // ④の書き出し(model_test)を済ませてあるので、CLIの名前もそれに揃うはず
 let cliName=null;
 const OrigBlob3=w.Blob;
 w.Blob=function(parts,opts){ return new OrigBlob3(parts,opts); };
 w.URL.createObjectURL=()=>'blob:cli';
 w.HTMLAnchorElement.prototype.click=function(){ cliName=this.download; };
 d.getElementById('simCliBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(cliName==='model_test.cli.js',`CLIの名前が④の書き出しと揃う (${cliName})`);
 const text=H.simBuildCliScript('model_test.cli.js');
 chk(typeof text==='string'&&text.length>10000,`CLIスクリプトを生成できる (${Math.round(text.length/1024)}KB)`);
 // 実際に node で動かし、ブラウザと同じ値になるか確かめる
 // 一時ディレクトリは実行場所の隣に作る（環境によっては /tmp に書けない）
 const dir=fs.mkdtempSync(path.join(__dirname,'.treg-cli-'));
 const js=path.join(dir,'m.cli.js'), inCsv=path.join(dir,'in.csv'), outCsv=path.join(dir,'out.csv');
 fs.writeFileSync(js,text,'utf8');
 // 出発点の行をそのままCSVにする（引用符やカンマを含む列名も試す）
 const cols=sliderSpec.map(s=>s.col);
 const vals=cols.map(c=>String(seedMed.values[c]));
 fs.writeFileSync(inCsv,cols.join(',')+'\n'+vals.join(',')+'\n','utf8');
 const {execFileSync}=require('child_process');
 let ran=true,msg='';
 try{ msg=execFileSync(process.execPath,[js,inCsv,outCsv],{encoding:'utf8'}); }
 catch(e){ ran=false; msg=String(e.stderr||e.message).slice(0,200); }
 chk(ran,`node で実行できる (${msg.trim().slice(0,60)})`);
 if(ran){
  const outRows=fs.readFileSync(outCsv,'utf8').trim().split('\n');
  chk(outRows.length===2,`入力1行に対して出力1行 (${outRows.length-1}行)`);
  const head=outRows[0].split(',');
  chk(head[head.length-1].endsWith('_pred'),`予測列が付く (${head[head.length-1]})`);
  const cells=outRows[1].split(',');
  const cliVal=Number(cells[cells.length-1]);
  chk(Math.abs(cliVal-expect)<1e-9,
    `ブラウザと同じ予測値になる (CLI ${cliVal.toFixed(6)} / ブラウザ ${expect.toFixed(6)})`);
 }
 fs.rmSync(dir,{recursive:true,force:true});} }catch(e){ chk(false,'CLI検証で例外: '+e.message); }

console.log('\n[14] メモ');
{chk(!!d.getElementById('simNoteBtn')&&!!d.getElementById('simNoteText'),'メモの入口と入力欄がある');
 chk(d.getElementById('simNote').classList.contains('open'),'メモは既定で開いている');
 d.getElementById('simNoteBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(!d.getElementById('simNote').classList.contains('open'),'✎で閉じられる');
 d.getElementById('simNoteBtn').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 const ta=d.getElementById('simNoteText');
 ta.value='温度を上げると収率が落ちる。次は触媒比を振る。';
 ta.dispatchEvent(new w.Event('input',{bubbles:true}));
 let stored=null; try{ stored=w.localStorage.getItem('treg_note_model_test'); }catch(_){}
 chk(stored===ta.value,'入力がこの端末に自動保存される');
 chk(d.getElementById('simNoteBtn').classList.contains('on'),'メモがあると入口に印が付く');
 chk(!d.getElementById('simNoteSaveBtn'),'メモ欄の中に保存ボタンは無い（右上に集約した）');
 // メモ機能の文言(ボタンのtitle・placeholder)は共有先の言語を選ばないよう常に英語。
 // 本体の言語トグル(data-i18n-*)には追従させない。
 chk(!d.getElementById('simNoteBtn').hasAttribute('data-i18n-title'),
   'メモボタンのtitleは言語トグルに追従しない(常に英語)');
 chk(d.getElementById('simNoteBtn').title==='Notes about this model',
   `メモボタンのtitleは英語固定 (${d.getElementById('simNoteBtn').title})`);
 chk(!d.getElementById('simNoteText').hasAttribute('data-i18n-placeholder'),
   'メモ入力欄のplaceholderも言語トグルに追従しない(常に英語)');
 chk(/^Notes about this model/.test(d.getElementById('simNoteText').placeholder),
   `placeholderは英語固定 (${d.getElementById('simNoteText').placeholder})`);
 H.setLang && H.setLang('ja'); // 本体の言語をJPに変えても、メモの文言は英語のまま
 chk(d.getElementById('simNoteBtn').title==='Notes about this model',
   '本体の言語をJPにしてもメモボタンのtitleは英語のまま');
 H.setLang && H.setLang('en');

console.log('\n[15] 変数名の書き換え');
{const r0=d.querySelectorAll('.sim-row')[0], col0=r0.dataset.col;
 const nm=r0.querySelector('.sim-name');
 chk(nm&&nm.tagName==='INPUT','変数名は書き換えられる入力欄');
 chk(nm.value===col0,`最初は列名がそのまま出る (${nm.value})`);
 nm.value='反応温度'; nm.dispatchEvent(new w.Event('input',{bubbles:true}));
 chk(H.SIM.labels[col0]==='反応温度','書き換えた名前を覚えている');
 chk(r0.dataset.col===col0,'列名(モデル側)は変わらない');
 chk(H.SIM.state[col0]!==undefined,'値も保持される');
 // 空にしたら列名に戻る
 nm.value=''; nm.dispatchEvent(new w.Event('input',{bubbles:true}));
 chk(H.SIM.labels[col0]===undefined,'空にすると元の列名に戻る');
 nm.value='反応温度'; nm.dispatchEvent(new w.Event('input',{bubbles:true}));}
{// 変数だけを元に戻す操作は、名前ではなく左端の重要度バーに移った
 const r1=d.querySelectorAll('.sim-row')[1], c1=r1.dataset.col;
 const rng=r1.querySelector('input[type=range]');
 const before=String(H.SIM.state[c1]);
 rng.value=rng.max; rng.dispatchEvent(new w.Event('change',{bubbles:true}));
 chk(r1.classList.contains('changed'),'動かすと印が付く');
 r1.querySelector('.sim-imp').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(String(H.SIM.state[c1])===before,'左端の棒をクリックでその変数だけ元に戻る');}

console.log('\n[16] 組み合わせの登録（SET / CLEAR、押すたびに増える、色は1件ごとに違う）');
{const setB=d.getElementById('simSetBtn'), clrB=d.getElementById('simClearBtn');
 chk(!!setB&&!!clrB,'SET と CLEAR のボタンがある');
 // 変数側(左)に置く。応答側(右、応答値・ヒストグラムのある sim-right)には置かない
 chk(!!d.querySelector('.sim-left #simMarks'),'登録一覧は変数側(左)にある');
 chk(!d.querySelector('.sim-right #simMarks'),'登録一覧は応答側(右)には無い');
 chk(d.getElementById('simMarks').children.length===0,'登録前は何も出ない');
 chk(clrB.disabled,'登録が無ければCLEARは無効');
 const r2=d.querySelectorAll('.sim-row')[2], c2=r2.dataset.col;
 const rng2=r2.querySelector('input[type=range]');
 rng2.value=rng2.max; rng2.dispatchEvent(new w.Event('change',{bubbles:true}));
 const at=String(H.SIM.state[c2]), pred=el('simPredVal').textContent;
 setB.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(H.SIM.marks.length===1&&String(H.SIM.marks[0].state[c2])===at,'SETで今の組み合わせを登録できる');
 chk(d.getElementById('simMarks').children.length===1,'登録した行が一覧に出る');
 const rowEl=(i)=>d.getElementById('simMarks').children[i];
 chk(Math.abs(Number(rowEl(0).querySelector('.sim-mark-val').textContent)-Number(pred))<0.2,
   `登録時の予測値が出る (${rowEl(0).querySelector('.sim-mark-val').textContent} / 画面 ${pred})`);
 chk(Object.keys(H.SIM.marks[0].state).length===simSpecCols.length,'全変数の値を覚える');
 chk(!clrB.disabled,'SET直後はそれがアクティブになりCLEARが有効になる');
 chk(rowEl(0).classList.contains('active'),'SETした行は自動的にアクティブ表示になる');
 const color1=rowEl(0).style.getPropertyValue('--mc');
 chk(!!color1,'各行に固有の色が設定される(CSS変数 --mc)');
 // 別の場所へ動かしてもう一度SET → ボタンが増え、色も変わる
 rng2.value=rng2.min; rng2.dispatchEvent(new w.Event('change',{bubbles:true}));
 const at2=String(H.SIM.state[c2]);
 chk(at2!==at,'動かすと現在値は離れる');
 chk(String(H.SIM.marks[0].state[c2])===at,'1つ目の登録は動かない');
 setB.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(H.SIM.marks.length===2,`SETするたびにボタンが増える (${H.SIM.marks.length}個)`);
 chk(d.getElementById('simMarks').children.length===2,'一覧の行も増える');
 const color2=rowEl(1).style.getPropertyValue('--mc');
 chk(color1!==color2,`SETごとに違う色が割り当てられる (${color1} / ${color2})`);
 chk(rowEl(1).classList.contains('active')&&!rowEl(0).classList.contains('active'),
   '新しく登録したほうがアクティブになる(前のは外れる)');
 // 1つ目の行をクリックすると、その状態に戻りつつアクティブになる
 rowEl(0).dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(String(H.SIM.state[c2])===at,'一覧の行をクリックするとその登録内容に戻る');
 chk(rowEl(0).classList.contains('active')&&!rowEl(1).classList.contains('active'),
   'クリックした行がアクティブになる(クリック=CLEAR対象を選ぶ操作)');
 // CLEARは「アクティブな1件だけ」を消す(全部は消えない)
 clrB.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(H.SIM.marks.length===1,`CLEARはアクティブな1件だけを消す(残り${H.SIM.marks.length}件)`);
 chk(H.SIM.marks[0].state[c2]!==undefined&&String(H.SIM.marks[0].state[c2])!==at,
   '消えたのは選んでいた1つ目で、2つ目は残る');
 chk(clrB.disabled,'消した後は選択が無くなりCLEARは再び無効になる');
 // ×ボタンでも個別に消せる(クリックしてアクティブにしなくても消せる)
 d.getElementById('simMarks').children[0].querySelector('.sim-mark-x')
   .dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 chk(H.SIM.marks.length===0,'×でも1件だけ消せる');
 chk(d.getElementById('simMarks').children.length===0,'消すと一覧も空になる');
 // もう一度登録しておく（保存に含まれるか見るため）
 setB.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
 setB.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));}

console.log('\n[17] 右上の保存（今の状態ごと書き出す）');
{const btn=d.getElementById('simSaveStateBtn');
 chk(!!btn,'右上に保存ボタンがある');
 chk(d.querySelector('.sim-header').contains(btn),'置き場所はヘッダー（右上）');
 let outHtml=null,outName=null;
 const OB=w.Blob;
 w.Blob=function(parts,opts){ const p0=parts[0];
   outHtml=(typeof p0==='string')?p0:Buffer.from(p0).toString('utf8'); return new OB(parts,opts); };
 w.URL.createObjectURL=()=>'blob:n';
 w.HTMLAnchorElement.prototype.click=function(){ outName=this.download; };
 return Promise.resolve(btn.dispatchEvent(new w.MouseEvent('click',{bubbles:true})))
  .then(()=>new Promise(r=>setTimeout(r,200))).then(()=>{
   chk(!!outHtml&&outHtml.includes('温度を上げると収率が落ちる'),'メモが埋め込まれる');
   chk(outName==='model_test.html',`名前も④と揃う (${outName})`);
   const m=outHtml.match(/const EMBEDDED_MODEL = (\{[\s\S]*?\});\r?\n/);
   chk(!!m,'埋め込み先が1箇所に決まる');
   const pl=JSON.parse(m[1]);
   chk(!!pl.ui,'画面の状態も一緒に入る');
   chk(pl.ui.labels&&Object.values(pl.ui.labels).includes('反応温度'),'書き換えた変数名が入る');
   chk(Array.isArray(pl.ui.marks)&&pl.ui.marks.length===2,`登録した組み合わせが全部入る (${pl.ui.marks&&pl.ui.marks.length}件)`);
   chk(Object.keys(pl.ui.state).length===simSpecCols.length,'スライダーの位置が入る');
   // 保存したHTMLを開くと、その状態から始まる
   const d3=new JSDOM(outHtml,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost/'});
   const w3=d3.window;
   w3.HTMLCanvasElement.prototype.getContext=()=>ctx2d;
   if(!w3.TextDecoder) w3.TextDecoder=require('util').TextDecoder;
   if(!w3.TextEncoder) w3.TextEncoder=require('util').TextEncoder;
   Object.defineProperty(w3.Element.prototype,'clientWidth',{get(){return 300;}});
   const code3=outHtml.match(/<script>([\s\S]*)<\/script>/)[1];
   let boot3=null;
   try{ w3.eval(code3+'\n;window.__H3__={SIM:SIM};'); }catch(e){ boot3=e.message; }
   chk(!boot3,'保存したHTMLが例外なく実行される'+(boot3?': '+boot3:''));
   return new Promise(r=>setTimeout(r,300)).then(()=>{
    const S3=w3.__H3__&&w3.__H3__.SIM;
    const nm3=w3.document.querySelector('.sim-name');
    chk(!!nm3&&nm3.value==='反応温度',`開くと変数名も再現される (${nm3&&nm3.value})`);
    chk(!!S3&&Array.isArray(S3.marks)&&S3.marks.length===2,`登録した組み合わせも全部再現される (${S3&&S3.marks&&S3.marks.length})`);
    const same=simSpecCols.every(c=>String(S3.state[c])===String(H.SIM.state[c]));
    chk(same,'スライダーの位置も再現される');
    // 背景クリックで消えてしまうと、単体HTMLでは戻す手段が無い
    const ov3=w3.document.getElementById('simulateOverlay');
    ov3.dispatchEvent(new w3.MouseEvent('click',{bubbles:true}));
    chk(ov3.style.display==='flex','背景をクリックしても画面が消えない');
    try{ d3.window.close(); }catch(_){}
    finish3();
   });
  });}}

function finish3(){
console.log('\n[8] 画面に出る文字量');
const shown=el('simulateSection').textContent.replace(/\s+/g,' ').trim();
console.log('       '+JSON.stringify(shown.slice(0,150))+'...');
console.log('       文字数 '+shown.length+' / title保持要素 '+q('#simulateSection [title]'));
chk(q('#simulateSection [title]')>=15,'説明はtitleに退避されている');

console.log('\n[18] 中国語(zh)対応');
{
 const btns=[...d.querySelectorAll('.lang-toggle-btn')].map(b=>b.dataset.lang);
 chk(btns.includes('ja')&&btns.includes('en')&&btns.includes('zh'),
   `言語トグルに日本語・英語・中国語がある (${btns.join(',')})`);
 // 3言語すべてで同じキー集合を持つ(訳し忘れがあると、その言語だけ他言語の文言が
 // 出てしまうため、キーの過不足がないことを機械的に確認する)。
 const i18nSrc=html.slice(html.indexOf('const I18N = {'), html.indexOf('\n    };', html.indexOf('const I18N = {'))+8);
 const dictKeys=(name)=>{
   const s=i18nSrc.indexOf(name+': {');
   const nextNames=['ja: {','en: {','zh: {'].map(n=>i18nSrc.indexOf(n)).filter(i=>i>s);
   const e=nextNames.length?Math.min(...nextNames):i18nSrc.length;
   return new Set([...i18nSrc.slice(s,e).matchAll(/^\s*'([a-zA-Z0-9_.]+)':/gm)].map(m=>m[1]));
 };
 const jaK=dictKeys('ja'), enK=dictKeys('en'), zhK=dictKeys('zh');
 chk(jaK.size>200&&enK.size===jaK.size&&zhK.size===jaK.size,
   `ja/en/zh のキー数が一致する (ja=${jaK.size} en=${enK.size} zh=${zhK.size})`);
 const missingInZh=[...jaK].filter(k=>!zhK.has(k));
 chk(missingInZh.length===0,`zhに無いキーが無い (${missingInZh.slice(0,5).join(',')})`);
 // 実際に切り替えて、中国語の訳文が表示に反映されることを確認する
 H.setLang('zh');
 const exportBtnText=d.getElementById('exportBtn').textContent;
 chk(exportBtnText==='下载模型',`中国語に切り替えると翻訳が反映される (${exportBtnText})`);
 // メモの文言は言語トグルに追従せず英語のまま(前述の[14]の方針と矛盾しない)
 chk(d.getElementById('simNoteBtn').title==='Notes about this model',
   '中国語に切り替えてもメモの文言は英語のまま');
 H.setLang('en');
 chk(d.getElementById('exportBtn').textContent==='Download model','英語に戻せる');
}

console.log('');
if(errs.length){console.log('ERRORS:\n'+errs.join('\n'));process.exit(1);}
if(FAIL.length){console.log(`NG: ${FAIL.length} 件失敗`);FAIL.forEach(m=>console.log('  - '+m));process.exit(1);}
console.log('すべて成功');
}
}
}

// jsdom の rAF ループが残るとプロセスが終了しないため、明示的に閉じて終了する
try { dom.window.close(); } catch (_) {}
process.exit(0);
