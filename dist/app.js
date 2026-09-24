const R = 8.31446261815324;
const $ = (id) => document.getElementById(id);
let lastResult = null;
let gammaUpload = null;
let sigmaA = null, sigmaB = null;
let modelServiceReady = false;
let propertyServiceReady = false;
const propertyResolution = {A:null,B:null};

function parseCSV(text){
  const rows=text.trim().split(/\r?\n/).filter(Boolean).map(r=>r.split(',').map(v=>v.trim()));
  const head=rows[0].map(x=>x.toLowerCase());
  return rows.slice(1).map(r=>Object.fromEntries(head.map((h,i)=>[h,r[i]])));
}
function num(id){ const raw=$(id).value.trim(); const v=Number(raw); if(!raw||!Number.isFinite(v)) throw new Error('Complete all melting temperatures and fusion enthalpies, or load the example.'); return v; }
function branch(x,gamma,tm,h){
  const den=1/tm-(R/(h*1000))*Math.log(Math.max(1e-12,x*gamma));
  const t=den>0?1/den:NaN;
  return Number.isFinite(t)&&t<tm&&t>=150?t:NaN;
}
function interp(rows,x,key){
  const s=[...rows].sort((a,b)=>a.x-b.x); if(!s.length) return 1;
  if(x<=s[0].x) return s[0][key]; if(x>=s.at(-1).x) return s.at(-1)[key];
  for(let i=1;i<s.length;i++) if(x<=s[i].x){const a=s[i-1],b=s[i],q=(x-a.x)/(b.x-a.x);return a[key]+q*(b[key]-a[key]);}
  return 1;
}
function solveSegment(p,sigma,T){
  const n=p.length, alpha=7.5e7; let g=Array(n).fill(1);
  for(let iter=0;iter<10000;iter++){
    const next=Array(n).fill(0).map((_,i)=>{
      let sum=0; for(let j=0;j<n;j++) sum+=Math.exp(-(0.5*alpha*(sigma[i]+sigma[j])**2)/(R*T))*p[j]*g[j];
      return .5*g[i]+.5/sum;
    });
    let delta=0; for(let i=0;i<n;i++) delta=Math.max(delta,Math.abs(next[i]-g[i])); g=next;if(delta<1e-10)break;
  } return g;
}
function combGamma(c1,c2,x,i){
  const z=10,V0=66.69,A0=79.53, comps=[c1,c2], xs=[x,1-x];
  const r=comps.map(c=>c.volume/V0),q=comps.map(c=>c.area/A0);
  const sr=xs[0]*r[0]+xs[1]*r[1],sq=xs[0]*q[0]+xs[1]*q[1];
  const phi=xs.map((v,k)=>v*r[k]/sr),theta=xs.map((v,k)=>v*q[k]/sq),l=r.map((v,k)=>z/2*(v-q[k])-(v-1));
  return Math.log(phi[i]/xs[i])+z/2*q[i]*Math.log(theta[i]/phi[i])+l[i]-phi[i]/xs[i]*(xs[0]*l[0]+xs[1]*l[1]);
}
function sigmaGamma(c1,c2,x,T=298.15){
  if(c1.sigma.length!==c2.sigma.length || c1.sigma.some((v,i)=>Math.abs(v-c2.sigma[i])>1e-8)) throw new Error('Sigma grids must match.');
  const pmix=c1.p.map((v,i)=>x*v*c1.area+(1-x)*c2.p[i]*c2.area); const total=pmix.reduce((a,b)=>a+b,0); const p=pmix.map(v=>v/total);
  const gm=solveSegment(p,c1.sigma,T),g1=solveSegment(c1.p,c1.sigma,T),g2=solveSegment(c2.p,c2.sigma,T);
  const residual=(c,pure,mix)=>c.area/7.5*c.p.reduce((s,v,i)=>s+v*(Math.log(mix[i])-Math.log(pure[i])),0);
  return [Math.exp(combGamma(c1,c2,x,0)+residual(c1,g1,gm)),Math.exp(combGamma(c1,c2,x,1)+residual(c2,g2,gm))];
}
async function fileProfile(file,area,volume){
  if(!file) throw new Error('Choose both sigma-profile CSV files.');
  const rows=parseCSV(await file.text()); const sigma=rows.map(r=>Number(r.sigma)); const p=rows.map(r=>Number(r.p_sigma));
  if(sigma.length!==51||sigma.some((v,i)=>!Number.isFinite(v)||Math.abs(v-(-.025+i*.001))>1e-8)||p.some(v=>!Number.isFinite(v)||v<0)||p.reduce((a,b)=>a+b,0)<=0||area<=0||volume<=0) throw new Error('Use a non-negative profile on the 51-point −0.025 to 0.025 grid, with positive area and volume.');
  const sum=p.reduce((a,b)=>a+b,0); return {sigma,p:p.map(v=>v/sum),area,volume};
}
function sourceSummary(){
  if(['A','B'].some(side=>$(`name${side}`).value.startsWith('Illustrative component')))return 'Illustrative inputs — not experimental or fitted-model evidence';
  const ids=['tmSourceA','hfSourceA','tmSourceB','hfSourceB']; const model=ids.filter(id=>$(id).value==='model').length;
  return model?`${4-model}/4 experimental; ${model}/4 estimated`:'4/4 experimental inputs';
}
function supportSummary(){
  return ['Not assessed','Property provenance is not an applicability score'];
}
function propertyLine(result){
  const tm=result.properties.tm,hf=result.properties.hfus;
  const fmt=(p,label)=>p.origin==='experimental'?`${label}: reviewed experiment`:`${label}: ${p.model} (reference MAE ${p.reference_mae.toFixed(2)} ${p.unit})`;
  return `${result.matched_database?'Database match':'New structure'} · ${fmt(tm,'Tm')} · ${fmt(hf,'ΔHfus')}`;
}
async function resolveComponent(side){
  const smiles=$(`smiles${side}`).value.trim();
  if(!smiles)throw new Error(`Enter the SMILES for component ${side}.`);
  const selected=$(`class${side}`).value;
  const response=await fetch('/api/properties',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles,componentClass:selected==='auto'?null:selected})});
  const result=await response.json();if(!response.ok)throw new Error(`Component ${side}: ${result.error||'property resolution failed'}`);
  propertyResolution[side]=result;
  $(`smiles${side}`).value=result.canonical_smiles;
  $(`class${side}`).value=result.component_class;
  if(result.name&&$(`name${side}`).value.match(/^Component [AB]$/))$(`name${side}`).value=result.name;
  $(`tm${side}`).value=result.properties.tm.value.toFixed(2);
  $(`hf${side}`).value=result.properties.hfus.value.toFixed(3);
  $(`tmSource${side}`).value=result.properties.tm.origin;
  $(`hfSource${side}`).value=result.properties.hfus.origin;
  const note=$(`resolveNote${side}`);note.textContent=propertyLine(result);note.className=`resolve-note ${result.warning?'warning':'resolved'}`;
  return result;
}
async function resolveProperties(){
  const button=$('resolveProperties');$('errorBox').classList.add('hidden');button.disabled=true;
  button.querySelector('span').textContent='Resolving reviewed data and missing properties…';
  $('propertyServiceLabel').textContent='First model inference can take a minute';
  try{
    const a=await resolveComponent('A'),b=await resolveComponent('B');
    $('systemType').value=a.component_class==='neutral'&&b.component_class==='neutral'?'neutral-neutral':a.component_class==='salt'&&b.component_class==='salt'?'salt-salt':'salt-neutral';
    button.querySelector('span').textContent='Pure properties completed';$('propertyServiceLabel').textContent='Experimental values used first · missing values modelled';
  }catch(e){$('errorBox').textContent=e.message;$('errorBox').classList.remove('hidden');button.querySelector('span').textContent='Complete pure properties from SMILES';$('propertyServiceLabel').textContent='Correct the structure and try again';}
  finally{button.disabled=!propertyServiceReady;}
}
async function calculate(){
  $('errorBox').classList.add('hidden');
  $('calculate').disabled=true;$('calculate').textContent='Calculating…';
  try{
    const tm1=num('tmA'),tm2=num('tmB'),h1=num('hfA'),h2=num('hfB'); if(tm1<=150||tm2<=150||h1<=0||h2<=0)throw new Error('Use positive fusion enthalpies and plausible Kelvin temperatures.');
    const mode=document.querySelector('input[name=activity]:checked').value;
    if(mode==='sigma'){sigmaA=await fileProfile($('sigmaFileA').files[0],num('areaA'),num('volA'));sigmaB=await fileProfile($('sigmaFileB').files[0],num('areaB'),num('volB'));}
    const points=[];
    for(let k=2;k<=98;k++){
      const x=k/100; let g=[1,1];
      if(mode==='gamma'){if(!gammaUpload)throw new Error('Upload a γ(x) CSV first.');g=[interp(gammaUpload,x,'gamma1'),interp(gammaUpload,x,'gamma2')];}
      if(mode==='sigma')g=sigmaGamma(sigmaA,sigmaB,x);
      const b1=branch(x,g[0],tm1,h1),b2=branch(1-x,g[1],tm2,h2); const t=Number.isFinite(b1)&&Number.isFinite(b2)?Math.max(b1,b2):Number.isFinite(b1)?b1:b2;
      points.push({x,t,b1,b2,gamma1:g[0],gamma2:g[1]});
    }
    const valid=points.filter(p=>Number.isFinite(p.t)); if(!valid.length)throw new Error('No finite liquidus curve was obtained for these inputs.');
    const physicalMin=valid.reduce((a,b)=>a.t<b.t?a:b); const support=supportSummary(); let min=physicalMin;
    if($('useCorrection').checked){
      if(!modelServiceReady)throw new Error('The frozen B4 model service is not connected.');
      if(mode==='ideal')throw new Error('Ideal SLE is a physical fallback, not a validated replacement for the non-ideal features used by B4. Uncheck model prediction, or supply non-ideal inputs.');
      if(!$('smilesA').value.trim()||!$('smilesB').value.trim())throw new Error('Both SMILES are required for model prediction.');
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({activityMode:mode,systemType:$('systemType').value,nameA:$('nameA').value,nameB:$('nameB').value,smilesA:$('smilesA').value,smilesB:$('smilesB').value,tm1,tm2,h1,h2,points})});
      const body=await response.json(); if(!response.ok)throw new Error(body.error||'B4 inference failed.'); body.predictions.forEach((v,i)=>points[i].corrected=v); min=points.filter(p=>Number.isFinite(p.corrected)).reduce((a,b)=>a.corrected<b.corrected?a:b);
    }
    const central=$('useCorrection').checked?min.corrected:min.t;
    lastResult={created:new Date().toISOString(),method:$('useCorrection').checked?'Exploratory B4 prediction':mode==='ideal'?'ideal SLE':mode==='gamma'?'uploaded activity-coefficient SLE':'simplified COSMO-based SLE',inputs:{componentA:$('nameA').value,componentB:$('nameB').value,smilesA:$('smilesA').value,smilesB:$('smilesB').value,systemType:$('systemType').value,tm1,tm2,h1,h2,propertyProvenance:sourceSummary()},minimum:{temperature_K:central,composition_xA:min.x},calibratedRange:null,scope:'User-input curve; production profile, applicability and optimized-target deployment checks are not completed by this interface.',points};
    $('minTemp').textContent=`${central.toFixed(1)} K`; $('minX').textContent=min.x.toFixed(2); $('methodLabel').textContent=lastResult.method;
    $('interval').textContent='Not assessed';$('intervalNote').textContent='User-input curves are not yet production-calibrated';$('xRangeNote').textContent='97-point search · mole fraction, not mass fraction';
    $('support').textContent=support[0];$('supportNote').textContent=support[1];$('propertyProv').textContent=sourceSummary();
    $('gammaProv').textContent=mode==='ideal'?'Ideal fallback (γ = 1)':mode==='gamma'?'User-supplied γ(x)':'Uploaded sigma profiles';
    $('interpretTitle').textContent=$('useCorrection').checked?'Exploratory model curve':'Physical reference calculated';$('statusDot').style.background='#13a6ad';
    $('interpretText').textContent=`The ${lastResult.method} minimum is ${central.toFixed(1)} K near xA = ${min.x.toFixed(2)}. Use the curve to explore composition. Experimental priority and DES formation are not established by this calculation.`;
    draw(points,min);
  }catch(e){$('errorBox').textContent=e.message;$('errorBox').classList.remove('hidden');}
  finally{$('calculate').disabled=false;$('calculate').textContent='Calculate composition curve';}
}
function draw(points,min){
  const svg=$('curveChart'),W=760,H=390,m={l:62,r:25,t:30,b:55}; const vals=points.filter(p=>Number.isFinite(p.t)); const yvals=vals.flatMap(p=>Number.isFinite(p.corrected)?[p.t,p.corrected]:[p.t]); let ymin=Math.floor(Math.min(...yvals,250)/20)*20-10,ymax=Math.ceil(Math.max(...yvals,350)/20)*20+10;if(ymax-ymin<80)ymax=ymin+80;
  const X=x=>m.l+x*(W-m.l-m.r),Y=y=>H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b);let html=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="#fbfdff"/>`;
  for(let i=0;i<=5;i++){const y=ymin+i*(ymax-ymin)/5;html+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#dfe7ef"/><text x="${m.l-10}" y="${Y(y)+5}" text-anchor="end" font-size="13" fill="#64758a">${y.toFixed(0)}</text>`;}
  for(let i=0;i<=5;i++){const x=i/5;html+=`<line x1="${X(x)}" x2="${X(x)}" y1="${m.t}" y2="${H-m.b}" stroke="#edf2f7"/><text x="${X(x)}" y="${H-m.b+24}" text-anchor="middle" font-size="13" fill="#64758a">${x.toFixed(1)}</text>`;}
  const d=vals.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)},${Y(p.t).toFixed(1)}`).join(' ');html+=`<path d="${d}" fill="none" stroke="#13a6ad" stroke-width="4" stroke-linejoin="round"/>`;
  if(vals.some(p=>Number.isFinite(p.corrected))){const dc=vals.filter(p=>Number.isFinite(p.corrected)).map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)},${Y(p.corrected).toFixed(1)}`).join(' ');html+=`<path d="${dc}" fill="none" stroke="#e7a531" stroke-width="4" stroke-linejoin="round"/>`;}
  const minY=Number.isFinite(min.corrected)?min.corrected:min.t;html+=`<line x1="${X(min.x)}" x2="${X(min.x)}" y1="${Y(minY)}" y2="${H-m.b}" stroke="#e7a531" stroke-dasharray="5 5"/><circle cx="${X(min.x)}" cy="${Y(minY)}" r="7" fill="#e7a531" stroke="white" stroke-width="3"/><text x="${X(min.x)+12}" y="${Y(minY)-10}" font-size="14" font-weight="700" fill="#9b6815">${minY.toFixed(1)} K</text>`;
  html+=`<text x="${(m.l+W-m.r)/2}" y="${H-10}" text-anchor="middle" font-size="15" font-weight="700" fill="#243d5b">Mole fraction of component A, xA</text><text x="18" y="${H/2}" transform="rotate(-90 18 ${H/2})" text-anchor="middle" font-size="15" font-weight="700" fill="#243d5b">Temperature (K)</text>`;svg.innerHTML=html;
}
function download(name,obj,type='application/json'){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([typeof obj==='string'?obj:JSON.stringify(obj,null,2)],{type}));a.download=name;a.click();URL.revokeObjectURL(a.href);}
function queue(){return JSON.parse(localStorage.getItem('des-update-queue')||'[]');}function updateCount(){$('queueCount').textContent=queue().length;}
document.querySelectorAll('input[name=activity]').forEach(r=>r.addEventListener('change',()=>{$('gammaPanel').classList.toggle('hidden',r.value!=='gamma'||!r.checked);$('sigmaPanel').classList.toggle('hidden',r.value!=='sigma'||!r.checked);}));
$('gammaFile').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;const rows=parseCSV(await f.text()).map(r=>({x:Number(r.x),gamma1:Number(r.gamma1),gamma2:Number(r.gamma2)})).sort((a,b)=>a.x-b.x);if(rows.length<2||rows.some((r,i)=>Object.values(r).some(v=>!Number.isFinite(v))||r.x<0||r.x>1||r.gamma1<=0||r.gamma2<=0||(i>0&&r.x===rows[i-1].x))||rows[0].x>.02||rows.at(-1).x<.98){gammaUpload=null;$('gammaFileName').textContent='Use positive γ values, unique x values, and coverage from 0.02 to 0.98.';return;}gammaUpload=rows;$('gammaFileName').textContent=f.name;});
$('calculate').addEventListener('click',calculate);$('exportResult').addEventListener('click',()=>lastResult?download('des-screening-result.json',lastResult):null);
$('resolveProperties').addEventListener('click',resolveProperties);
$('smilesA').addEventListener('input',()=>{propertyResolution.A=null;$('resolveNoteA').textContent='Structure changed; resolve properties again.';$('resolveNoteA').className='resolve-note';});
$('smilesB').addEventListener('input',()=>{propertyResolution.B=null;$('resolveNoteB').textContent='Structure changed; resolve properties again.';$('resolveNoteB').className='resolve-note';});
$('loadDemo').addEventListener('click',()=>{Object.entries({nameA:'Illustrative component A',nameB:'Illustrative component B',smilesA:'NC(=O)N',smilesB:'CC(=O)N',tmA:405,tmB:353,hfA:14.5,hfB:12}).forEach(([k,v])=>$(k).value=v);['tmSourceA','hfSourceA','tmSourceB','hfSourceB'].forEach(id=>$(id).value='model');['A','B'].forEach(side=>{propertyResolution[side]=null;$(`resolveNote${side}`).textContent='Demonstration values only — not reviewed measurements.';});$('useCorrection').checked=false;document.querySelector('input[name=activity][value=ideal]').click();calculate();});
$('saveMeasurement').addEventListener('click',()=>{const value=Number($('measurementValue').value);if(!Number.isFinite(value))return;const q=queue();q.push({created:new Date().toISOString(),type:$('measurementType').value,value,note:$('measurementNote').value,system:lastResult?.inputs||null});localStorage.setItem('des-update-queue',JSON.stringify(q));$('measurementValue').value='';$('measurementNote').value='';updateCount();});
$('exportQueue').addEventListener('click',()=>download('des-model-update-queue.json',{exported:new Date().toISOString(),policy:'candidate for reviewed periodic release; no online retraining',records:queue()}));
updateCount();$('curveChart').innerHTML='<text x="380" y="175" text-anchor="middle" font-size="20" fill="#64758a">Your composition curve will appear here</text><text x="380" y="212" text-anchor="middle" font-size="16" fill="#64758a">Load the example to try the calculator</text>';
fetch('/api/health').then(r=>r.ok?r.json():Promise.reject()).then(body=>{modelServiceReady=true;propertyServiceReady=body.properties?.status==='ready';$('connectionStatus').textContent='Local prediction service connected';$('serviceNotice').textContent='Enter two SMILES and complete pure properties, or enter your measured values. Advanced activity inputs are optional for physical curves; model predictions require non-ideal inputs.';$('useCorrection').disabled=false;$('resolveProperties').disabled=!propertyServiceReady;if(propertyServiceReady)$('propertyServiceLabel').textContent=`${body.properties.database_rows.toLocaleString()} reviewed components · four frozen models ready`;const row=$('useCorrection').closest('.switch-row');row.classList.add('ready');row.querySelector('em').textContent='ready';row.querySelector('small').textContent='Frozen B4 inference service connected';}).catch(()=>{});
