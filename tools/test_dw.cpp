#include "acfun.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>

static void dw_naive(const float* in, int N,int H,int W,int C,const float* w,const float* b,
                     int KH,int KW,int st,int pd,float* out,int act){
    int OH=(H+2*pd-KH)/st+1, OW=(W+2*pd-KW)/st+1;
    for(int n=0;n<N;++n)for(int oc=0;oc<C;++oc){
        const float* wch=w+(size_t)oc*KH*KW;
        for(int oh=0;oh<OH;++oh)for(int ow=0;ow<OW;++ow){
            float s=b?b[oc]:0.0f;
            for(int kh=0;kh<KH;++kh){int ih=oh*st-pd+kh; if(ih<0||ih>=H)continue;
                for(int kw=0;kw<KW;++kw){int iw=ow*st-pd+kw; if(iw<0||iw>=W)continue;
                    size_t idx=((size_t)n*H+ih)*W+iw;
                    s+=in[idx*C+oc]*wch[kh*KW+kw];}}
            if(act==1&&s<0)s=0;
            size_t odx=((size_t)n*OH+oh)*OW+ow;
            out[odx*C+oc]=s;
        }
    }
}

static void gn_naive(const float* in, int N,int H,int W,int Cin,const float* w,const float* b,
                     int KH,int KW,int Cout,int st,int pd,int G,float* out,int act){
    int Cin_p=Cin/G, Cout_p=Cout/G;
    int OH=(H+2*pd-KH)/st+1, OW=(W+2*pd-KW)/st+1;
    for(int g=0;g<G;++g){
        const float* wg=w+(size_t)g*Cout_p*KH*KW*Cin_p;   /* 组内标准 conv 权重块 [(kh,kw,ic)][oc] */
        const float* bg=b?(b+(size_t)g*Cout_p):nullptr;
        for(int n=0;n<N;++n)for(int oc=0;oc<Cout_p;++oc){
            for(int oh=0;oh<OH;++oh)for(int ow=0;ow<OW;++ow){
                float s=bg?bg[oc]:0.0f;
                for(int kh=0;kh<KH;++kh){int ih=oh*st-pd+kh; if(ih<0||ih>=H)continue;
                    for(int kw=0;kw<KW;++kw){int iw=ow*st-pd+kw; if(iw<0||iw>=W)continue;
                        for(int ic=0;ic<Cin_p;++ic){
                            size_t idx=((size_t)n*H+ih)*W+iw;
                            size_t iaddr=((size_t)g*Cin_p) + idx*(size_t)Cin + ic;  /* acfun 抽取后的组输入通道 */
                            s+=in[iaddr]*wg[((kh*KW+kw)*Cin_p+ic)*Cout_p+oc];}}}
                if(act==1&&s<0)s=0;
                size_t odx=((size_t)n*OH+oh)*OW+ow;
                out[odx*(size_t)Cout+ (size_t)g*Cout_p+oc]=s;
            }
        }
    }
}

static float rnd(){ return (float)(rand()%200-100)/50.0f; }

int main(){
    int fails=0;
    struct D{int H,W,C,KH,KW,st,pd,act;} dws[]={
        {112,112,16,3,3,1,1,1},{56,56,32,3,3,1,1,0},{224,224,3,3,3,1,1,1},
        {56,56,128,3,3,2,1,1},{28,28,24,3,3,1,1,0},{13,13,512,3,3,1,1,1},
        {64,64,64,1,1,1,0,1},{33,31,12,3,3,1,1,0},{16,16,8,3,3,1,1,1},
    };
    for(auto&d:dws){
        int OH=(d.H+2*d.pd-d.KH)/d.st+1, OW=(d.W+2*d.pd-d.KW)/d.st+1;
        std::vector<float> in((size_t)d.H*d.W*d.C), w((size_t)d.C*d.KH*d.KW), b(d.C,0.1f);
        std::vector<float> on((size_t)OH*OW*d.C), oa((size_t)OH*OW*d.C);
        for(auto&v:in)v=rnd(); for(auto&v:w)v=rnd();
        dw_naive(in.data(),1,d.H,d.W,d.C,w.data(),b.data(),d.KH,d.KW,d.st,d.pd,on.data(),d.act);
        acfun_conv2d_dw(in.data(),1,d.H,d.W,d.C,w.data(),b.data(),d.KH,d.KW,d.st,d.pd,oa.data(),d.act);
        float md=0; for(size_t i=0;i<oa.size();++i) md=std::max(md,std::fabs(on[i]-oa[i]));
        bool ok=md<1e-2f; if(!ok)fails++;
        printf("DW %dx%dx%d k%d s%d p%d act%d  max|Δ|=%9.3e  %s\n",d.H,d.W,d.C,d.KH,d.st,d.pd,d.act,md,ok?"OK":"FAIL");
    }
    struct G{int H,W,Cin,Cout,KH,KW,st,pd,Gg; int act;} gs[]={
        {56,56,16,32,3,3,1,1,4,1},{28,28,32,64,1,1,1,0,8,0},{14,14,24,48,3,3,2,1,6,1},
    };
    for(auto&g:gs){
        int OH=(g.H+2*g.pd-g.KH)/g.st+1, OW=(g.W+2*g.pd-g.KW)/g.st+1;
        int Cin_p=g.Cin/g.Gg, Cout_p=g.Cout/g.Gg;
        std::vector<float> in((size_t)g.H*g.W*g.Cin), w((size_t)g.Gg*Cout_p*g.KH*g.KW*Cin_p), b(g.Cout,0.1f);
        for(auto&v:in)v=rnd(); for(auto&v:w)v=rnd();
        std::vector<float> on((size_t)OH*OW*g.Cout), oa((size_t)OH*OW*g.Cout);
        gn_naive(in.data(),1,g.H,g.W,g.Cin,w.data(),b.data(),g.KH,g.KW,g.Cout,g.st,g.pd,g.Gg,on.data(),g.act);
        acfun_conv2d_g(in.data(),1,g.H,g.W,g.Cin,w.data(),b.data(),g.KH,g.KW,g.Cout,g.st,g.pd,g.Gg,oa.data(),g.act);
        float md=0; for(size_t i=0;i<oa.size();++i) md=std::max(md,std::fabs(on[i]-oa[i]));
        bool ok=md<1e-2f; if(!ok)fails++;
        printf("G  %dx%dx%d->%d k%d g%d  max|Δ|=%9.3e  %s\n",g.H,g.W,g.Cin,g.Cout,g.KH,g.Gg,md,ok?"OK":"FAIL");
    }
    printf(fails==0?"DW/GROUPED ALL OK\n":"DW/GROUPED FAIL\n");
    return fails?1:0;
}
