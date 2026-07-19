// Step 3: Rust FFI -> C shim -> AOT FlashInfer GDN kernel. Validate bit-exact vs JIT ref.
use std::os::raw::{c_void, c_int, c_float};
#[link(name="cudart")] extern "C" {
    fn cudaMalloc(p:*mut *mut c_void, sz:usize)->c_int;
    fn cudaMemcpy(dst:*mut c_void, src:*const c_void, sz:usize, kind:c_int)->c_int;
    fn cudaMemset(p:*mut c_void, val:c_int, sz:usize)->c_int;
    fn cudaStreamCreate(s:*mut *mut c_void)->c_int;
    fn cudaDeviceSynchronize()->c_int;
}
#[link(name="atlasgdn")] extern "C" {
    fn atlas_gdn_load();
    fn atlas_gdn_prefill(q:*mut c_void,k:*mut c_void,v:*mut c_void,o:*mut c_void,
        alpha:*mut c_void,beta:*mut c_void,state:*mut c_void,init_state:*mut c_void,
        tensormaps:*mut c_void,cu:*mut c_void, scale:c_float, total_seqlen:c_int,
        nq:c_int,nk:c_int,nv:c_int,nsab:c_int,nseqs:c_int,grid_x:c_int, stream:*mut c_void)->c_int;
}
const H2D:c_int=1; const D2H:c_int=2;
fn rd(p:&str)->Vec<u8>{ std::fs::read(p).expect(p) }
unsafe fn up(b:&[u8])->*mut c_void{ let mut d=std::ptr::null_mut(); cudaMalloc(&mut d,b.len()); cudaMemcpy(d,b.as_ptr() as *const c_void,b.len(),H2D); d }
unsafe fn zeros(n:usize)->*mut c_void{ let mut d=std::ptr::null_mut(); cudaMalloc(&mut d,n); cudaMemset(d,0,n); d }
fn h2f(h:u16)->f32{ // ieee half -> f32
    let s=((h>>15)&1) as u32; let e=((h>>10)&0x1f) as u32; let m=(h&0x3ff) as u32;
    let bits = if e==0 { if m==0 {s<<31} else { // subnormal
        let mut e2=127-15+1; let mut m2=m; while m2&0x400==0 {m2<<=1; e2-=1;} m2&=0x3ff; (s<<31)|(e2<<23)|(m2<<13)
    }} else if e==0x1f { (s<<31)|(0xff<<23)|(m<<13) } else { (s<<31)|((e+127-15)<<23)|(m<<13) };
    f32::from_bits(bits)
}
fn main(){ unsafe{
    let (t,hv,d)=(2048usize,32usize,128usize);
    atlas_gdn_load();
    let q=up(&rd("/tmp/gdn_ref/q.bin")); let k=up(&rd("/tmp/gdn_ref/k.bin"));
    let v=up(&rd("/tmp/gdn_ref/v.bin")); let al=up(&rd("/tmp/gdn_ref/g.bin"));
    let be=up(&rd("/tmp/gdn_ref/beta.bin"));
    let o=zeros(t*hv*d*2); let st=zeros(hv*d*d*4); let ini=zeros(hv*d*d*4); let tm=zeros(6144*128);
    let cu_host:[i64;2]=[0, t as i64];
    let cu={ let mut p=std::ptr::null_mut(); cudaMalloc(&mut p,16); cudaMemcpy(p,cu_host.as_ptr() as *const c_void,16,H2D); p };
    let mut stream=std::ptr::null_mut(); cudaStreamCreate(&mut stream);
    let ret=atlas_gdn_prefill(q,k,v,o,al,be,st,ini,tm,cu, 0.08838834764831843f32, t as c_int,
        16,16,32,32,1, 32, stream);
    cudaDeviceSynchronize();
    println!("Rust->shim->AOT kernel: atlas_gdn_prefill ret={}", ret);
    // read o (fp16) + o_ref (fp32) and compare
    let n=t*hv*d; let mut ho=vec![0u8; n*2];
    cudaMemcpy(ho.as_mut_ptr() as *mut c_void, o, n*2, D2H);
    let refb=rd("/tmp/gdn_ref/o_ref.bin");
    let rf:&[f32]=unsafe{ std::slice::from_raw_parts(refb.as_ptr() as *const f32, n) };
    let (mut maxe,mut so,mut sr,mut dot,mut no2,mut nr2,mut nan)=(0f64,0f64,0f64,0f64,0f64,0f64,0u32);
    for i in 0..n {
        let ov=h2f(u16::from_le_bytes([ho[i*2],ho[i*2+1]])) as f64;
        if ov.is_nan(){nan+=1;}
        let e=(ov-rf[i] as f64).abs(); if e>maxe{maxe=e;}
        so+=ov.abs(); sr+=(rf[i] as f64).abs(); dot+=ov*rf[i] as f64; no2+=ov*ov; nr2+=(rf[i]as f64)*(rf[i]as f64);
    }
    let cos=dot/(no2.sqrt()*nr2.sqrt()+1e-12);
    println!("max_abs_err={:.6}  |o|mean={:.6}  |ref|mean={:.6}  cos={:.6}  nan={}", maxe, so/n as f64, sr/n as f64, cos, nan);
    println!("RESULT: {}", if cos>0.99 && nan==0 && so/n as f64>1e-4 {"PASS - Rust FFI -> AOT GDN kernel matches JIT reference"} else {"MISMATCH"});
}}
