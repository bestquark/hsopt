// Deterministic 3Dmol.js ball-and-stick illustrations; not AI-generated artwork.
// Coordinates are schematic geometries for visual identification, not benchmark inputs.
const {chromium} = require('playwright');
const fs = require('fs');
(async()=>{
 const browser=await chromium.launch({...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {}),headless:true,args:['--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:640,height:360},deviceScaleFactor:4});
 await page.setContent('<html><body style="margin:0"><div id="mol" style="position:absolute;width:640px;height:360px"></div></body></html>');
 await page.addScriptTag({path:require.resolve('3dmol/build/3Dmol-min.js')});
 const molecules={
  CoH:[['Co',-0.8,0,0],['H',0.8,0.10,0]],
  O3:[['O',-1.08,-0.32,0],['O',0,0.34,0],['O',1.08,-0.32,0]],
  Na2:[['Na',-1.5,0,0],['Na',1.5,0.1,0]],
  NaLi:[['Na',-1.4,0,0],['Li',1.4,0.12,0]]
 };
 for(const [name,coords] of Object.entries(molecules)){
  const uri=await page.evaluate(({name,coords})=>{
   const el=document.getElementById('mol');el.innerHTML='';
   const v=$3Dmol.createViewer(el,{backgroundColor:'white',antialias:true});
   const a=coords.map(([elem,x,y,z],i)=>({elem,x,y,z,serial:i,bonds:[],bondOrder:[]}));
   const edges=name==='O3'?[[0,1],[1,2]]:[[0,1]];
   for(const [i,j] of edges){a[i].bonds.push(j);a[i].bondOrder.push(1);a[j].bonds.push(i);a[j].bondOrder.push(1);}
   const m=v.addModel();m.addAtoms(a);
   const colors={Co:'#647f98',H:'#dedede',O:'#bf5550',Na:'#9c8db8',Li:'#c59766'};
   const radii={Co:0.62,H:0.38,O:0.47,Na:0.69,Li:0.55};
   for(const elem of Object.keys(colors))v.setStyle({elem},{sphere:{radius:radii[elem],color:colors[elem]},stick:{radius:0.13,color:colors[elem]}});
   v.zoomTo();v.zoom(3.3);v.rotate(8,'x');v.render();
   return v.pngURI();
  },{name,coords});
  fs.writeFileSync(name+'.png',Buffer.from(uri.split(',')[1],'base64'));
 }
 await browser.close();
})();
